"""S8 on the tiny backbone: multi-depth router features, router architectures / objectives / pooling, hard-negative mining
and the head recipe knobs, inference arms (k / stride / tail pad), the arm registry, Table 4, Fig 1, s8ctl dry runs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from lgs1.lgcore import adapter as A
from lgs1.lgcore.model import LGConfig, LandmarkGuard, Router, RouterMLP, make_router, router_in_dim, trunk_windows, trunk_windows_multi
from lgs4 import backbone as BB
from lgs4 import config as C4
from lgs4 import hardneg as HN
from lgs4 import head_train as HT
from lgs4 import router_train as RT
from lgs4 import sampling as SM
from lgs4.prep import PrepSet
from lgs5 import config as C5
from lgs5 import run as R5
from lgs5 import tables as T5
from lgs5.runners import make_runner, write_rows
from lgs8 import arms as ARMS_MOD
from lgs8 import table4 as T4
from lgs8 import fig1 as F1
from lgs8.arms import ARMS, Arm, arms_of, cli_flags, resolve_head_flags
from lgs8.config import ABL_DEPTHS, S8Paths, eval_args, subset_row
from tests.test_s4_train import s4root  # noqa: F401  (fixture)
from tests.test_s5_eval import s5root  # noqa: F401  (fixture)


def _ids(n, seed=3, vocab=512):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g)


# ------------------------------------------------------------------------------------------------
# lgcore: multi-depth trunk, router architectures
# ------------------------------------------------------------------------------------------------
def test_multi_depth_trunk_and_router(tiny):
    ids = _ids(300)
    from lgs1.lgcore.windows import make_windows
    wins = make_windows(300, 8)
    states, valid = trunk_windows_multi(tiny, ids, wins, [1, 2, 3], pad_id=0)
    for e in (1, 2, 3):
        h, v = trunk_windows(tiny, ids, wins, e, 0)
        assert torch.allclose(states[e], h, atol=1e-4) and torch.equal(v, valid), e     # CPU eager attention: thread-order noise
    # LandmarkGuard with the router reading depths 1 and 3 (concatenated), the head from ell=2
    d = 64
    cfg = LGConfig(b=8, ell=2, k=3, pool="mean", router_ells=[1, 3])
    assert cfg.router_depths == [1, 3] and cfg.trunk_depth == 3 and router_in_dim(cfg, d) == 2 * d
    torch.manual_seed(0)
    router = Router(d, in_dim=2 * d)
    with torch.no_grad():
        router.proj.weight.normal_(0, 0.5)
    lg = LandmarkGuard(tiny, cfg, router).eval()
    full = lg(ids)
    st = lg.forward_stream(ids, chunk_windows=4)
    assert sorted(st.selected.tolist()) == sorted(full.selected.tolist()) and abs(st.score() - full.score()) < 1e-4
    assert full.token_scores.shape == (len(wins), 16)
    r = lg.route(ids)
    assert r["hidden"].shape[-1] == d                            # the head state stays d-wide
    # the router's scores depend on the depth-3 state: a d-wide default router on the same model gives other scores
    lg_def = LandmarkGuard(tiny, LGConfig(b=8, ell=2, k=3, pool="mean"), Router(d)).eval()
    assert lg_def(ids).token_scores.shape == full.token_scores.shape
    # MLP router
    mlp = make_router(d, "mlp", in_dim=2 * d, hidden=16)
    assert isinstance(mlp, RouterMLP) and mlp.in_dim == 2 * d and mlp.arch == "mlp"
    out = mlp(torch.randn(3, 16, 2 * d))
    assert out.shape == (3, 16)
    with pytest.raises(ValueError):
        make_router(d, "conv")


# ------------------------------------------------------------------------------------------------
# S5 spec options and suffix-safe derivations
# ------------------------------------------------------------------------------------------------
def test_spec_options_and_derivations():
    s = C5.parse_spec("lg:tiny_s0:k4:s16:tp8")
    assert s.stride == 16 and s.tail_pad == 8 and s.method == "lg_k4_s16_tp8" and str(s) == "lg:tiny_s0:k4:s16:tp8"
    assert C5.parse_spec("sliding:tiny_s0:s16").method == "sliding_s16" and C5.parse_spec("oracle:tiny_s0").method == "oracle"
    assert C5.split_lg_method("lg_k32_s256") == (32, "_s256") and C5.split_lg_method("lg_k8") == (8, "") and C5.split_lg_method("sliding") is None
    with pytest.raises(SystemExit):
        C5.parse_spec("dense:tiny_s0:tp8")
    rows = [{"method": "lg_k4_tp8", "model": "m", "deep_probs": [0.9, 0.1, 0.2, 0.3], "selected": [5, 1, 2, 3], "needle_windows": [1], "label": 1},
            {"method": "lg_k4", "model": "m", "deep_probs": [0.2, 0.8, 0.1, 0.1], "selected": [7, 1, 2, 3], "needle_windows": [1], "label": 1}]
    dk = T5.derive_k(rows, (2,))
    assert sorted(r["method"] for r in dk) == ["lg_k2", "lg_k2_tp8"]         # the option suffix survives the derivation
    assert {r["method"] for r in T5.derive_agg(rows, k=4)} == {"lg_k4_rg3", "lg_k4_tp8_rg3"}


# ------------------------------------------------------------------------------------------------
# hard negatives, recipe filters, head / router training knobs, load_lg round trip
# ------------------------------------------------------------------------------------------------
def test_hardneg_recipes_and_router_variants(s5root, monkeypatch):
    P, tok, repo = s5root
    P5 = C5.S5Paths.load().ensure()
    ps = PrepSet(P, "train", "main")
    # ---- mining with the base tiny run ----
    out_dir = HN.mine(P, "tiny", 0, sets=[("train", "main")], device="cpu", m_router=2, dtype=torch.float32)
    z = np.load(out_dir / "train.main.npz")
    assert len(z["row_ptr"]) == len(ps) + 1 and len(z["win_start"]) == int(z["row_ptr"][-1]) > 0
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["m_router"] == 2 and meta["b"] == C4.WINDOW_B and meta["sets"]["train.main"]["rows"] == len(ps)
    hard = SM.HardNegIndex.load(out_dir, ["train.main"])
    ws = hard.windows("train.main", 0)
    assert ws and all(ws[i][2] >= ws[i + 1][2] for i in range(len(ws) - 1))          # sorted by head prob desc
    span = ps.meta[0]["needle_tok_span"]
    assert all(not (s <= span[0] and span[1] <= e) for s, e, _, _ in ws)              # never a needle window
    # cached second call
    assert HN.mine(P, "tiny", 0, sets=[("train", "main")], device="cpu", m_router=2, dtype=torch.float32) == out_dir
    # ---- sampling: hard negatives replace random negatives; filters ----
    import random
    units = SM.row_units(ps, 0, 0, random.Random(0), neg_per_row=4, near_neg=1, hard=hard, hard_per_row=2)
    kinds = [u.kind for u in units]
    # up to 2 mined windows (fewer when a mined window is also a near window), the random negatives fill up to neg_per_row
    assert 1 <= kinds.count("hard") <= 2 and kinds.count("hard") + kinds.count("neg") <= 4 and all(u.label == 0 for u in units if u.kind == "hard")
    assert len({(u.start, u.end) for u in units}) == len(units)                       # no window twice
    hard_spans = {(s, e) for s, e, _, _ in hard.windows("train.main", 0)}
    assert all((u.start, u.end) in hard_spans for u in units if u.kind == "hard")
    f = SM.make_row_filter(max_L=700)
    assert all(f(m) for m in ps.meta if m["L_target"] == 700) and not any(f(m) for m in ps.meta if m["L_target"] == 1100)
    assert SM.make_row_filter() is None
    f2 = SM.make_row_filter(needle_frac=0.5, seed=0)
    kept = [m for m in ps.meta if f2(m)]
    assert 0 < len(kept) < len(ps.meta)
    pairs = {}
    for m in ps.meta:
        pairs.setdefault(m["pair_id"], set()).add(f2(m))
    assert all(len(v) == 1 for v in pairs.values())                                   # both rows of a pair on the same side
    f3 = SM.make_row_filter(twin=False)
    assert not any(f3(m) for m in ps.meta if m["label"] == 0) and all(f3(m) for m in ps.meta if m["label"] == 1)
    assert not any(SM.make_row_filter(domains=["arxiv"])(m) for m in ps.meta)         # tiny rows are all govreport
    ep = SM.epoch_units([ps], [], 0, 0, 2, 1, 0.0, row_filter=SM.make_row_filter(max_L=700), hard=hard, hard_per_row=1)
    assert ep and all(ps.meta[u.row]["L_target"] == 700 for u in ep) and any(u.kind == "hard" for u in ep)
    # ---- head: hardneg_ft-style run (init from the base head, hard negatives, 1 epoch) ----
    base = P.run_dir("tiny", 0)
    hp = {"batch_windows": 8, "log_every": 2, "eval_every": 0, "epochs": 1, "hard_neg": str(out_dir), "hard_per_row": 2,
          "init_from": str(base / "head.pt"), "max_L": 1100}
    res = HT.train(P, "tiny", 0, hp, device="cpu", max_steps=3, allow_unsealed=True, dtype=torch.float32, val_max_rows=4, variant="hnft")
    hd = P.run_dir("tiny", 0, "hnft")
    hc = BB.load_ckpt(hd / "head.pt")
    assert res["steps"] == 3 and hc["meta"]["init_from"]["kind"] == "lg_head" and hc["meta"]["init_from"]["probe"] is True
    assert hc["meta"]["epoch0_kinds"].get("hard", 0) > 0 and hc["meta"]["recipe"]["hard_per_row"] == 2 and hc["meta"]["b"] == C4.WINDOW_B
    # dense init + b=64 (the mined negatives were mined at b=128 -> refused with b=64)
    hp2 = {"batch_windows": 8, "log_every": 2, "eval_every": 0, "epochs": 1, "init_from": str(base / "dense_ft.pt"), "b": 64}
    res2 = HT.train(P, "tiny", 0, hp2, device="cpu", max_steps=2, allow_unsealed=True, dtype=torch.float32, val_max_rows=4, variant="b64d")
    hc2 = BB.load_ckpt(P.run_dir("tiny", 0, "b64d") / "head.pt")
    assert res2["steps"] == 2 and hc2["cfg"]["b"] == 64 and hc2["meta"]["init_from"]["kind"] == "dense_ft" and hc2["meta"]["tokens_per_step"] == 8 * 128
    with pytest.raises(SystemExit):
        HT.train(P, "tiny", 0, dict(hp2, hard_neg=str(out_dir), hard_per_row=1), device="cpu", max_steps=1, allow_unsealed=True,
                 dtype=torch.float32, val_max_rows=4, variant="b64bad")
    # ---- router variants on a run dir with a SYMLINKED base head (router arm layout) ----
    import s8ctl
    arm = Arm("pm", "E6-4 pooling", "router", "mean pool + mlp + token_bce + depths 1,2", router={"pool": "mean", "arch": "mlp", "objective": "token_bce", "router-ells": "1,2"})
    link = s8ctl._link_base_head(arm, P, "tiny", 0)
    assert link.is_symlink() and BB.sha256_file(link) == BB.sha256_file(base / "head.pt")
    rr = RT.train(P, "tiny", 0, {"epochs": 1, "batch_rows": 4, "log_every": 2, "pool": "mean", "arch": "mlp", "mlp_hidden": 16,
                                 "objective": "token_bce", "router_ells": "1,2"},
                  device="cpu", allow_unsealed=True, val_max_rows=4, dtype=torch.float32, variant="pm")
    rd = P.run_dir("tiny", 0, "pm")
    rc = BB.load_ckpt(rd / "router.pt")
    assert rr["steps"] > 0 and rc["arch"] == "mlp" and rc["in_dim"] == 2 * 64 and rc["cfg"]["pool"] == "mean" and rc["cfg"]["router_ells"] == [1, 2]
    assert rc["hp"]["target"] == "label" and rc["hp"]["lambda_rank"] == 0.0 and rc["meta"]["objective"] == "token_bce"
    model, _, head, router, cfg, info = BB.load_lg(rd, device="cpu", k=2, dtype=torch.float32)
    assert isinstance(router, RouterMLP) and cfg.pool == "mean" and cfg.router_ells == [1, 2] and cfg.ell == 2
    # the arm evaluates end to end and its rows carry the variant run name
    spec = C5.parse_spec("lg:tiny_pm_s0:k2")
    rows, man = R5.rows_for(P5, "e2", spec, [700, 1100], depths=ABL_DEPTHS, max_pairs=200, calib_main=False)
    assert rows and not any(r["split"] == "calib" and r["subset"] == "main" for r in rows)
    n = write_rows(make_runner(P5, spec, device="cpu"), spec, rows, P5.rows_path("e2", spec), manifest_sha=man, log_every=1000)
    assert n == len(rows)
    # other objectives / pools run too (linear, one epoch each)
    for i, (obj, pool) in enumerate((("distill", "top2"), ("window_bce", "lse"))):
        arm_i = Arm(f"o{i}", "E6-5 objective", "router", "", router={"objective": obj, "pool": pool})
        s8ctl._link_base_head(arm_i, P, "tiny", 0)
        r_i = RT.train(P, "tiny", 0, {"epochs": 1, "batch_rows": 4, "log_every": 2, "objective": obj, "pool": pool}, device="cpu",
                       allow_unsealed=True, val_max_rows=4, dtype=torch.float32, variant=f"o{i}")
        assert r_i["steps"] > 0 and BB.load_ckpt(P.run_dir("tiny", 0, f"o{i}") / "router.pt")["cfg"]["pool"] == pool
    with pytest.raises(SystemExit):
        RT.objective_settings({"objective": "bogus", "lambda_rank": 1.0})


# ------------------------------------------------------------------------------------------------
# inference arms (tail pad, stride, k), Table 4, Fig 1, s8ctl
# ------------------------------------------------------------------------------------------------
def test_infer_arms_table4_fig1_and_ctl(s5root, monkeypatch, capsys):
    P, tok, repo = s5root
    P5 = C5.S5Paths.load().ensure()
    Ls = [700, 1100]
    # baseline rows (k=4) with the E2-7 twins (the full e2 pass) -> Table 4 restricts them to the subset
    base = C5.parse_spec("lg:tiny_s0:k4")
    rows, man = R5.rows_for(P5, "e2", base, Ls)
    write_rows(make_runner(P5, base, device="cpu"), base, rows, P5.rows_path("e2", base), manifest_sha=man, log_every=1000)
    assert any(r["split"] == "calib" and r["subset"] == "main" for r in rows)
    assert all(subset_row(r) for r in rows if r["subset"] == "benign_long")
    assert not any(subset_row(r) for r in rows if r["split"] == "calib" and r["subset"] == "main")
    # tail pad + stride arms on the same run
    tp = C5.parse_spec("lg:tiny_s0:k4:tp8")
    runner = make_runner(P5, tp, device="cpu")
    r0 = next(r for r in rows if r["subset"] == "main")
    ids_plain, _ = make_runner(P5, base, device="cpu")._ids(r0["text"], [r0["needle_char_span"]])
    ids_tp, spans_tp = runner._ids(r0["text"], [r0["needle_char_span"]])
    assert ids_tp.numel() == ids_plain.numel() + 8 and torch.equal(ids_tp[: ids_plain.numel()], ids_plain)
    s = runner.score_row(r0)
    assert s["tail_pad"] == 8 and s["n_tok"] == ids_plain.numel() + 8
    rows_tp, man_tp = R5.rows_for(P5, "e2", tp, Ls, depths=ABL_DEPTHS, max_pairs=200, calib_main=False)
    assert write_rows(runner, tp, rows_tp, P5.rows_path("e2", tp), manifest_sha=man_tp, log_every=1000) == len(rows_tp)
    st = C5.parse_spec("lg:tiny_s0:k4:s256")
    rs = make_runner(P5, st, device="cpu")
    assert rs.cfg.stride == 256 and rs.score_row(r0)["n_windows"] < s["n_windows"] and rs.score_row(r0)["stride"] == 256
    # a k=3 arm evaluated on 2 pairs only (an arm mid-evaluation, or the 2026-09-27 arms before their 0.2 / 0.8 rows)
    k3 = C5.parse_spec("lg:tiny_s0:k3")
    rows_k3, man_k3 = R5.rows_for(P5, "e2", k3, Ls, depths=ABL_DEPTHS, max_pairs=2, calib_main=False)
    assert 0 < len([r for r in rows_k3 if r["subset"] == "main"]) < len([r for r in rows if r["split"] == "test" and r["subset"] == "main"])
    assert write_rows(make_runner(P5, k3, device="cpu"), k3, rows_k3, P5.rows_path("e2", k3), manifest_sha=man_k3, log_every=1000) == len(rows_k3)
    # a second seed of the baseline recipe = the s0 rows under the model name tiny_s1 (Table 4 shows it as the noise floor)
    s1_path = P5.rows_path("e2", C5.parse_spec("lg:tiny_s1:k4"))
    with open(P5.rows_path("e2", base), encoding="utf-8") as f_in, open(s1_path, "w", encoding="utf-8") as f_out:
        for ln in f_in:
            f_out.write(ln.replace('"model": "tiny_s0"', '"model": "tiny_s1"'))
    # ---- Table 4 with a custom arm list (tiny names) ----
    arms = [Arm("tp8", "tailpad", "infer", "8 neutral suffix tokens", spec_opts=":k4:tp8"),
            Arm("k2", "E6-1 k", "infer", "top-2 (own pass, not run here)", spec_opts=":k2"),
            Arm("k3", "E6-1 k", "infer", "top-3 on two pairs", spec_opts=":k3"),
            Arm("pm", "E6-4 pooling", "router", "mean pool mlp", router={"pool": "mean"}, spec_opts=":k2")]
    # the pm arm's rows come from test_hardneg_recipes_and_router_variants when it ran first; here it may be absent
    P8 = S8Paths.load().ensure()
    res = T4.build(P8, Ls, backbone="tiny", seed=0, arms=arms, base_k=4, derived_ks=(2,))
    by_arm = {r["arm"]: r for r in res["rows"]}
    assert by_arm["base"]["n_rows"] and by_arm["base"].get("R@128k") is not None       # 'R@128k' = the largest L of the grid
    assert by_arm["k2"]["kind"] == "derived" and by_arm["tp8"]["kind"] == "infer" and by_arm["tp8"].get("R@128k") is not None
    assert "d_R@128k" in by_arm["tp8"] and by_arm["tp8"]["status"] == "ok" and by_arm["pm"]["status"] in ("ok", "no rows")
    assert by_arm["k3"]["status"].startswith("partial ") and "d_R@128k" in by_arm["k3"] and by_arm["k3"]["n_rows"] < by_arm["base"]["n_rows"]
    n_main = len({r["id"] for r in rows_k3 if r["subset"] == "main"})
    assert by_arm["k3"]["status"].startswith(f"partial {n_main:,}/")
    md = (P8.reports / "table4_ablation.md").read_text()
    assert res["subset"]["depths"] == ABL_DEPTHS and "partial n/N" in md and "| G2(a) |" in md and "| tau@128k |" in md
    assert by_arm["s1"]["kind"] == "seed" and by_arm["s1"]["status"] == "ok" and abs(by_arm["s1"]["d_R@128k"]) < 1e-9
    assert [r["arm"] for r in res["rows"]][:2] == ["base", "s1"] and by_arm["base"].get("tau@128k") is not None
    assert by_arm["base"].get("twin_fpr@28k") is not None and by_arm["base"].get("test_fpr@5pct@128k") is not None
    s1_path.unlink()
    # derived rows come from the k=4 pass only (the partial k=3 pass of the same run adds no second copy), and every row
    # carries the G2 (a) verdict against its own R@1k
    assert by_arm["k2"]["n_rows"] == by_arm["base"]["n_rows"] and by_arm["base"]["g2a"] in (True, False) and by_arm["base"]["R@1k"] is not None
    # the arm's rows re-read with rg3 / top-2: combo rows right after the arm, derived from the arm's own pass
    assert by_arm["tp8+rg3"]["method"] == "lg_k4_tp8_rg3" and by_arm["tp8+rg3"]["kind"] == "derived" and by_arm["tp8+rg3"]["status"] == "ok"
    assert by_arm["tp8+k2"]["method"] == "lg_k2_tp8" and by_arm["tp8+k2"].get("R@128k") is not None
    assert "k3+rg3" not in by_arm and [r["arm"] for r in res["rows"]].index("tp8+rg3") == [r["arm"] for r in res["rows"]].index("tp8") + 1
    md = (P8.reports / "table4_ablation.md").read_text()
    assert "| tailpad | tp8 | infer |" in md and "| baseline | base | base |" in md and "no rows" in md or "| E6-4 pooling | pm |" in md
    assert (P8.reports / "fig10_k_sweep.json").exists() and "4" in res["fig10"]["recall"]
    # ---- Fig 1 from the S5 table2 + a fake S7 table5 ----
    T5.build(P5, Ls)
    t2 = json.loads((P5.reports / "table2_le28k.json").read_text())
    t2["rows"] = [dict(r, per_L={"28672": r["per_L"][str(Ls[0])], "131072": r["per_L"][str(Ls[-1])]}) for r in t2["rows"]]
    (P5.reports / "table2_allL.json").write_text(json.dumps(t2))
    s7 = P5.out / "reports" / "s7"
    s7.mkdir(parents=True, exist_ok=True)
    t5 = {"rtxa5000": [{"method": "lg_k4+c16+h8", "model": "tiny_s0", "p50_ms@1024": 10.0, "p50_ms@28672": 100.0, "p50_ms@131072": 400.0, "peak_mb@131072": 900},
                       {"method": "dense_ft_yarn", "model": "tiny_s0", "p50_ms@28672": 1000.0, "p50_ms@131072": 9000.0, "peak_mb@131072": 5000}]}
    (s7 / "table5_cost.json").write_text(json.dumps(t5))
    res1 = F1.build(P5, gpu="rtxa5000")
    pts = {(p["method"], p["model"]): p for p in res1["points"]}
    assert pts[("lg_k4", "tiny_s0")]["cost_method"] == "lg_k4+c16+h8" and pts[("lg_k4", "tiny_s0")]["ms@28k"] == 100.0
    assert pts[("lg_k4", "tiny_s0")]["usd_per_1M_docs@28k"] > 0 and pts[("lg_k4", "tiny_s0")]["peak_mb@128k"] == 900
    assert pts[("lg_k4_tp8", "tiny_s0")]["cost_source"] == "eval ms_mean" and pts[("lg_k4_tp8", "tiny_s0")]["cost_method"] is None
    assert any(p["pareto@28k"] for p in res1["points"]) and (s7 / "fig1_pareto.md").exists()
    assert F1.match_cost("qwen3guard_gen_8b/chunk_or_512_256", "qwen3guard_gen_8b", [{"method": "qwen3guard_gen_8b/cd512o256", "model": "qwen3guard_gen_8b"}])
    assert F1.match_cost("qwen3guard_gen_8b/native", "qwen3guard_gen_8b", [{"method": "qwen3guard_gen_8b/yarn", "model": "qwen3guard_gen_8b"}])
    # rows with eval-side options map to the same-option cost row; the rg3 aggregation costs what its source rows cost
    t5x = [{"method": "lg_k32_tp64+c16+h8", "model": "m"}, {"method": "lg_k32+c16+h8", "model": "m"}]
    assert F1.match_cost("lg_k32_tp64_rg3", "m", t5x)["method"] == "lg_k32_tp64+c16+h8" and F1.match_cost("lg_k32_tp64", "m", t5x)["method"] == "lg_k32_tp64+c16+h8"
    assert F1.match_cost("lg_k32_rg3", "m", t5x)["method"] == "lg_k32+c16+h8" and F1.match_cost("lg_k8_tp64", "m", t5x) is None
    from lgs7.tables import g2_cost
    g2 = g2_cost([{"method": "lg_k32_tp64+c16+h8", "model": "m", "p50_ms@131072": 1.0, "peak_mb@1024": 1, "peak_mb@131072": 1, "mem_growth_1k_to_128k": 0.0},
                  {"method": "sliding+c16", "model": "m", "p50_ms@131072": 4.0}])
    assert g2["lg_k32_tp64+c16+h8|m"]["b_latency"]["ratio"] == 0.25 and g2["lg_k32_tp64+c16+h8|m"]["c_memory"]["pass_c"]
    assert ARMS["ep1"].head == {"epochs": 1} and ARMS["ep1"].kind == "head" and [a.name for a in arms_of(axis="E6-10")] == ["ep1", "ep2"]
    # ---- registry ----
    assert len(ARMS) >= 30 and {a.kind for a in ARMS.values()} == {"infer", "router", "head"}
    assert ARMS["hardneg"].spec_str() == "lg:qwen3guard_0.6b_hardneg_s0" and ARMS["k64"].spec_str() == "lg:qwen3guard_0.6b_s0:k64"
    assert ARMS["kall"].spec().method == "sliding" and ARMS["tp32"].spec().method == "lg_k32_tp32" and ARMS["stride256"].spec().method == "lg_k32_s256"
    assert ARMS["hardneg_tp32"].spec_str() == "lg:qwen3guard_0.6b_hardneg_s0:tp32" and ARMS["hardneg_tp32"].variant == "hardneg" and ARMS["hardneg"].variant == "hardneg"
    assert [a.name for a in arms_of(axis="E6-4")] == ["pool_mean", "pool_top2", "pool_top4", "pool_lse"]
    with pytest.raises(SystemExit):
        arms_of(["nope"])
    P4 = C4.S4Paths.load()
    fl = resolve_head_flags(ARMS["hardneg_ft"], P4)
    assert fl["hard-neg"].endswith("qwen3guard_0.6b_s0/hardneg") and fl["init-from"].endswith("qwen3guard_0.6b_s0/head.pt") and fl["epochs"] == 1
    assert cli_flags({"b": 64, "no-twin": True, "domains": "govreport", "x": None}) == " --b 64 --no-twin --domains govreport"
    assert eval_args() == "--L all --depths 0.0,0.2,0.5,0.8,1.0 --max-pairs 200 --no-calib-main"
    # ---- s8ctl dry runs ----
    import s8ctl
    cfg = {"qos": "normal", "gres": "gpu:1", "cpus": "4", "mem": "32G", "nodelist": "", "cpu_partitions": "", "cpu_qos": "", "cpu_gres": ""}
    monkeypatch.setattr(s8ctl.s1ctl, "_slurm_cfg", lambda: cfg)
    monkeypatch.setattr(s8ctl.s1ctl, "partitions_for", lambda gb, bf16=None: "A5000")
    assert s8ctl.main(["arms", "--axis", "E6-6"]) == 0
    out = capsys.readouterr().out
    assert "hardneg_ft" in out and "init_dense" in out
    assert s8ctl.main(["run", "--arms", "k64,tp32,pool_mean,hardneg,hardneg_tp32", "--sbatch", "--dry-run-sbatch"]) == 0
    out = capsys.readouterr().out
    assert "--model lg:qwen3guard_0.6b_hardneg_s0:tp32 " in out
    assert "-m lgs4.hardneg --backbone qwen3guard_0.6b --seed 0" in out                       # mining job first (not mined yet)
    assert "-m lgs4.head_train --backbone qwen3guard_0.6b --seed 0 --variant hardneg --hard-neg" in out and "--hard-per-row 4" in out
    assert "#SBATCH --signal=B:TERM@1500" in out and "-m lgs4.router_train --backbone qwen3guard_0.6b --seed 0 --variant pool_mean --pool mean" in out
    assert "-m lgs5.run --exp e2 --model lg:qwen3guard_0.6b_s0:k64 --L all --depths 0.0,0.2,0.5,0.8,1.0 --max-pairs 200 --no-calib-main" in out
    assert "--model lg:qwen3guard_0.6b_s0:tp32 " in out and "--model lg:qwen3guard_0.6b_pool_mean_s0 " in out and "--model lg:qwen3guard_0.6b_hardneg_s0 " in out
    assert "#SBATCH --signal=B:TERM@300" in out                                                    # eval jobs resume by id
    assert s8ctl.main(["train", "--axis", "E6-3", "--sbatch", "--dry-run-sbatch"]) == 0
    out = capsys.readouterr().out
    assert "--variant ell6_12 --router-ells 6,12" in out and "lgs4.head_train" not in out and "--exclude" not in out
    # LG_EXCLUDE=n44 -> every sbatch gets #SBATCH --exclude=n44 (a node whose GPU throws uncorrectable ECC errors)
    monkeypatch.setattr(s8ctl.s1ctl, "_slurm_cfg", lambda: dict(cfg, exclude="n44"))
    assert s8ctl.main(["eval", "--arms", "b256_tp64", "--sbatch", "--dry-run-sbatch", "--force"]) == 0
    out = capsys.readouterr().out
    assert "#SBATCH --exclude=n44" in out and "--model lg:qwen3guard_0.6b_b256_s0:tp64 " in out and "excluding nodes n44" in out
    monkeypatch.setattr(s8ctl.s1ctl, "_slurm_cfg", lambda: cfg)
    assert s8ctl.main(["tables", "--sbatch", "--dry-run-sbatch"]) == 0
    assert "-m lgs8.table4" in capsys.readouterr().out
    # `table`: only the asked arms (with their combo rows) after the baseline / seed rows, from the JSON of the build above
    assert s8ctl.main(["table", "--arms", "tp8,k3,nope"]) == 0
    out = capsys.readouterr().out
    names = [c for c in (l.split("|")[1].strip() for l in out.splitlines() if l.startswith("| ")) if c not in ("arm", "---")]
    assert names[:2] == ["base", "s1"] and "tp8" in names and "tp8+rg3" in names and "tp8+k2" in names and "k3" in names
    assert "pm" not in names and "k2" not in names and "partial" in out and "(no rows in the last tables run for: ['nope'])" in out
    assert s8ctl.main(["table", "--arms", "k3", "--no-base"]) == 0
    out = capsys.readouterr().out
    assert "| base |" not in out and "| k3 |" in out
    assert s8ctl.main(["status", "--backbone", "tiny"]) == 0
    out = capsys.readouterr().out
    assert "hard negatives:" in out and "baseline rows" in out
    # eval status from the logs: b256's log is s8_eval_b256_<jid>.out only (not b256_tp64's), a started-but-empty log with
    # no job left in the queue is a startup failure (9/30: b256_tp64 job 1037593 vanished, status said "queued")
    logs = P5.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "s8_eval_b256_777.out").write_text("[s5] e2 lg:x done=24,000 todo=0\n")
    (logs / "s8_eval_b256_tp64_778.out").write_text("")
    monkeypatch.setattr(s8ctl.s5ctl, "_queued_names", lambda: set())
    assert s8ctl.main(["status", "--backbone", "tiny", "--axis", "E6-2"]) == 0 and s8ctl.main(["status", "--backbone", "tiny", "--axis", "tailpad"]) == 0
    out = capsys.readouterr().out
    b256 = [l for l in out.splitlines() if l.strip().startswith("b256 ")][0]
    tp = [l for l in out.splitlines() if l.strip().startswith("b256_tp64 ")][0]
    assert b256.rstrip().endswith("done") and "FAILED before the first rows" in tp and "s8_eval_b256_tp64_778.err" in tp
    with pytest.raises(SystemExit):
        s8ctl.main(["run", "--sbatch", "--dry-run-sbatch"])                                        # no arms given


# ------------------------------------------------------------------------------------------------
# cfg_hash: the pre-S8 form is the form (2026-09-27 LineageMismatch), the 9/27 files resume under their own hash
# ------------------------------------------------------------------------------------------------
def test_cfg_hash_form_and_20260927_alias(s5root, capsys):
    from lgs1.lineage import LineageMismatch, canonical_hash, iter_rows
    from lgs2.config import RECIPE_V2, depth_warning, missing_depths
    from lgs5.runners import HARNESS, lg_cfg_dict, lg_cfg_dict_20260927
    # LGConfig.to_dict(): the seven pre-S8 keys, router_ells only when set
    assert LGConfig(b=8, ell=2, k=3).to_dict() == {"b": 8, "stride": None, "ell": 2, "k": 3, "pool": "max", "tau": 0.5, "pad_id": 0}
    assert LGConfig(b=8, ell=2, k=3, router_ells=[3, 1]).to_dict()["router_ells"] == [3, 1]
    assert LGConfig(**LGConfig(b=8, ell=2, k=3).to_dict()).router_ells is None
    P, tok, repo = s5root
    P5 = C5.S5Paths.load().ensure()
    Ls = [700, 1100]
    spec = C5.parse_spec("lg:tiny_s0:k4")
    runner = make_runner(P5, spec, device="cpu")
    cfg = runner.cfg
    # the hash every S5/S6 LG rows file carries: exactly the pre-9/27 dict (k twice: cfg.k and spec.k, equal)
    pre_s8 = dict(b=cfg.b, stride=cfg.stride, ell=cfg.ell, pool=cfg.pool, tau=cfg.tau, pad_id=cfg.pad_id,
                  method="lg_k4", harness=HARNESS, k=4)
    assert cfg.k == 4
    assert runner.lineage.cfg_hash == canonical_hash(pre_s8) == canonical_hash(lg_cfg_dict(cfg, "lg_k4", 4, 0))
    assert "tail_pad" not in lg_cfg_dict(cfg, "lg_k4", 4, 0) and lg_cfg_dict(cfg, "lg_k4_tp8", 4, 8)["tail_pad"] == 8
    alias = runner.lineage_aliases[0]
    assert alias.cfg_hash == canonical_hash(dict(pre_s8, router_ells=None, tail_pad=0)) != runner.lineage.cfg_hash
    assert alias.model_sha == runner.lineage.model_sha and alias.router_sha == runner.lineage.router_sha
    assert lg_cfg_dict_20260927(cfg, "lg_k4", 4, 0) == dict(pre_s8, router_ells=None, tail_pad=0)
    # a rows file written by the 9/27 code (its hash on every row) is resumed, and stays under that hash
    k3 = C5.parse_spec("lg:tiny_s0:k3")
    rows, man = R5.rows_for(P5, "e2", k3, Ls, calib_main=False)
    r3 = make_runner(P5, k3, device="cpu")
    old_lineage, legacy = r3.lineage, r3.lineage_aliases[0]
    r3.lineage = legacy
    path = P5.rows_path("e2", k3)
    if path.exists():
        path.unlink()
    assert write_rows(r3, k3, rows[: len(rows) // 2], path, manifest_sha=man, log_every=1000) == len(rows) // 2
    r3.lineage = old_lineage
    assert write_rows(r3, k3, rows, path, manifest_sha=man, log_every=1000) == len(rows) - len(rows) // 2
    assert "2026-09-27 cfg_hash" in capsys.readouterr().out
    assert {r["cfg_hash"] for r in iter_rows(path)} == {legacy.cfg_hash} and len(list(iter_rows(path))) == len(rows)
    assert write_rows(r3, k3, rows, path, manifest_sha=man) == 0                             # complete: nothing to do
    # without a matching alias the mismatch is still refused
    r3.lineage_aliases = []
    with pytest.raises(LineageMismatch):
        write_rows(r3, k3, rows + [dict(rows[0], id="extra-row")], path, manifest_sha=man)
    path.unlink()
    # depth subsets: the 0.1-step test grid has no 0.25 / 0.75; ABL_DEPTHS are grid values; the runners warn on a miss
    grid = RECIPE_V2["sets"]["test"]["positions"]
    assert set(ABL_DEPTHS) <= set(grid) and len(ABL_DEPTHS) == 5 and missing_depths([{"depth": d} for d in grid], [0, 0.25, 0.5, 0.75, 1]) == [0.25, 0.75]
    assert not missing_depths([{"depth": d} for d in grid], ABL_DEPTHS) and depth_warning([{"depth": 0.5}], [0.5], "x") == ""
    assert "0.25" in depth_warning([{"depth": 0.5}, {"depth": None}], [0.25, 0.5], "x")
    R5.rows_for(P5, "e2", spec, Ls, depths=[0.25, 0.5])                                     # tiny rows sit at depth 0.5
    assert "[depths] WARNING" in capsys.readouterr().out
    R5.rows_for(P5, "e2", spec, Ls, depths=[0.5])
    assert "[depths]" not in capsys.readouterr().out


def test_derivations_read_one_native_pass():
    """S8 E6-1: a k=64 pass next to the k=32 rows of the same run must not double the derived lg_k8 / lg_k32_rg3 rows."""
    def row(method, i, k):
        return {"id": f"r{i}", "split": "test", "subset": "main", "L_target": 1024, "depth": 0.5, "label": 1, "method": method, "model": "m",
                "deep_probs": [0.9] + [0.1] * (k - 1), "selected": list(range(k)), "needle_windows": [0], "score": 0.9, "verdict": True,
                "sel_scores": [1.0] * k, "n_windows": 40, "n_deep": k}
    rows = [row("lg_k32", i, 32) for i in range(3)] + [row("lg_k64", i, 64) for i in range(2)]
    assert T5.native_source_k(rows) == {("m", ""): 32}
    d = T5.derive_k(rows, (8,))
    assert len(d) == 3 and {r["derived_from"] for r in d} == {"lg_k32"} and {r["method"] for r in d} == {"lg_k8"}
    a = T5.derive_agg(rows, k=32)
    assert len(a) == 3 and {r["derived_from"] for r in a} == {"lg_k32"} and {r["method"] for r in a} == {"lg_k32_rg3"}
    # the larger pass is used when it is the one with more rows (a complete k=64 file next to a partial k=32 file)
    rows2 = [row("lg_k32", i, 32) for i in range(2)] + [row("lg_k64", i, 64) for i in range(3)]
    assert T5.native_source_k(rows2) == {("m", ""): 64} and {r["derived_from"] for r in T5.derive_k(rows2, (8,))} == {"lg_k64"}
    # options are separate sources (tp32 rows derive their own lg_k8_tp32), ties go to the smaller k
    rows3 = rows + [row("lg_k32_tp32", i, 32) for i in range(3)] + [row("lg_k64", 9, 64)]
    src = T5.native_source_k(rows3)
    assert src == {("m", ""): 32, ("m", "_tp32"): 32}
    assert sorted({r["method"] for r in T5.derive_k(rows3, (8,))}) == ["lg_k8", "lg_k8_tp32"]


def test_table2_leaves_subset_files_to_table4():
    """Table 2 / g2_check read the full-grid e2 files only; the S8 ablation-subset files (5 of 11 depths) are Table 4's."""
    def rows(depths):
        return [{"split": "test", "subset": "main", "depth": d} for d in depths] + [{"split": "test", "subset": "benign_long", "depth": None}]
    grid = [round(0.1 * i, 1) for i in range(11)]
    files = {"lg_k32__m_s0": rows(grid), "lg_k32_tp64__m_s0": rows(grid), "lg_k32_tp32__m_s0": rows([0, .2, .5, .8, 1]),
             "lg_k32__m_arm_s0": rows([0, .2, .5, .8, 1]), "needle__m_s0": rows([]), "lg_k32__m8_s0": rows(grid[:9])}
    assert sorted(T5.subset_files(files)) == ["lg_k32__m_arm_s0", "lg_k32_tp32__m_s0"]      # 9 of 11 (a pass mid-way) stays
    assert T5.subset_files({"a": rows([0.5]), "b": rows([0.5])}) == []                       # one depth everywhere (tiny data)
