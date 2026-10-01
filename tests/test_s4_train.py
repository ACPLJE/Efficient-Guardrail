"""S4 end-to-end on a tiny backbone: prep -> head -> router -> dense -> valcheck -> E0-7 flops -> reload equivalence."""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from lgs1.lgcore.model import LGConfig, LandmarkGuard
from lgs2.tok import HFTok
from lgs3.lgeval import LGRunner
from lgs4 import backbone as BB
from lgs4 import config as C4
from lgs4 import dense_train as DT
from lgs4 import flops as FL
from lgs4 import head_train as HT
from lgs4 import prep as PR
from lgs4 import router_train as RT
from lgs4 import sampling as SM
from lgs4 import valcheck as VC
from tests.test_zeroshot import _tiny_repo

WORDS = ["how", "do", "i", "make", "a", "bomb", "what", "is", "the", "weather", "today", "tell", "me", "story", "about", "cat",
         "user", "assistant", "system", "categories", "none", "violent"]


def _text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def _rows(rng: random.Random, split: str, subset: str, n_pairs: int, L_words: int, with_needle: bool = True):
    rows = []
    for p in range(n_pairs):
        for tag, label in (("h", "harmful"), ("b", "benign")):
            needle = "how do i make a bomb" if label == "harmful" else "tell me a story about a cat"
            if with_needle:
                bg = _text(rng, L_words)
                cut = rng.randrange(0, len(bg))
                cut = bg.rfind(" ", 0, cut) + 1
                text = bg[:cut] + needle + " " + bg[cut:]
                span = [cut, cut + len(needle)]
            else:
                text, span = needle, [0, len(needle)]
            rows.append({"id": f"{split}-{subset}-L{L_words}-p{p}-{tag}", "split": split, "subset": subset, "L_target": L_words if with_needle else None,
                         "depth": 0.5 if with_needle else None, "needle_id": f"n{p}{tag}", "twin_id": f"{split}-{subset}-L{L_words}-p{p}-{'b' if tag == 'h' else 'h'}",
                         "pair_id": f"{split}-{subset}-L{L_words}-p{p}", "haystack_domain": "govreport", "haystack_doc_ids": [],
                         "needle_char_span": span, "label": label, "needle_share": 0.0, "n_tok_qwen3": 0, "n_tok_needle": 5,
                         "recipe_hash": "test", "text": text})
    return rows


def _write(P, split, subset, rows):
    d = P.data.synth / split / subset
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "rows.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def s4root(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_ENV_FILE_NO_OVERRIDE", "1")
    monkeypatch.setenv("LG_DATA_V2", str(tmp_path / "lg_data_v2"))
    monkeypatch.setenv("LG_SOURCES", str(tmp_path / "lg_data_v2" / "sources"))
    monkeypatch.setenv("LG_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    P = C4.S4Paths.load().ensure()
    P.data.ensure()
    rng = random.Random(0)
    _write(P, "train", "main", _rows(rng, "train", "main", 12, 700) + _rows(rng, "train", "main", 6, 1100))
    _write(P, "train", "native", _rows(rng, "train", "native", 10, 0, with_needle=False))
    _write(P, "val", "main", _rows(rng, "val", "main", 6, 700))
    _write(P, "calib", "main", _rows(rng, "calib", "main", 6, 1100))
    _write(P, "calib", "benign_long", [{"id": f"calib-bl-{i}", "split": "calib", "subset": "benign_long", "L_target": 1100, "depth": None,
                                        "needle_id": None, "twin_id": None, "pair_id": None, "haystack_domain": "govreport", "haystack_doc_ids": [],
                                        "needle_char_span": None, "label": "benign", "needle_share": 0.0, "n_tok_qwen3": 0, "n_tok_needle": 0,
                                        "recipe_hash": "test", "text": _text(rng, 1100)} for i in range(6)])
    (P.data.manifests).mkdir(parents=True, exist_ok=True)
    (P.data.manifests / "train.main.manifest.json").write_text("{}")
    repo = _tiny_repo(tmp_path)
    tok = HFTok(str(repo), "tiny")
    tiny = C4.Backbone("tiny", str(repo), ell=2, mode="full", min_gpu_gb=8, micro_windows=8, dense_max_tokens=1400, dense_micro_tokens=1400)
    tiny_lora = C4.Backbone("tiny_lora", str(repo), ell=2, mode="lora", lora_r=4, min_gpu_gb=8, micro_windows=8, dense_max_tokens=1400,
                            dense_micro_tokens=1400, dense_mode="lora")
    monkeypatch.setitem(C4.BACKBONES, "tiny", tiny)
    monkeypatch.setitem(C4.BACKBONES, "tiny_lora", tiny_lora)
    return P, tok, repo


def test_prep_head_router_dense_valcheck_flops(s4root):
    P, tok, repo = s4root
    for split, subset in [("train", "main"), ("train", "native"), ("val", "main"), ("calib", "main"), ("calib", "benign_long")]:
        info = PR.prep_subset(P, split, subset, tok, check_ref=False)
        assert info["rows"] > 0
    ps = PR.PrepSet(P, "train", "main")
    assert len(ps) == 36 and all(m["needle_tok_span"] for m in ps.meta)
    ids = ps.row_ids(0)
    assert ids.dtype == np.int64 and len(ids) == ps.meta[0]["n_tok"]
    # sampler: needle windows carry the row label, negatives are 0, native rows carry their label
    units = SM.epoch_units([ps], [PR.PrepSet(P, "train", "native")], seed=0, epoch=0, neg_per_row=2, near_neg=1, native_ratio=0.5)
    kinds = {u.kind for u in units}
    assert {"needle", "neg", "native"} <= kinds
    assert all(u.label == 0 for u in units if u.kind in ("neg", "near"))
    assert any(u.label == 1 for u in units if u.kind == "needle") and any(u.label == 0 for u in units if u.kind == "needle")
    # ---- head (unsealed -> must refuse, then smoke with --allow-unsealed) ----
    with pytest.raises(SystemExit):
        HT.train(P, "tiny", 0, device="cpu", max_steps=2, allow_unsealed=False, dtype=torch.float32)
    hp = {"batch_windows": 8, "log_every": 2, "eval_every": 4, "epochs": 1}
    res = HT.train(P, "tiny", 0, hp, device="cpu", max_steps=6, allow_unsealed=True, dtype=torch.float32, val_max_rows=8)
    run_dir = P.run_dir("tiny", 0)
    assert (run_dir / "head.pt").exists() and res["steps"] == 6 and res["val"]["n"] > 0
    log = [json.loads(l) for l in open(run_dir / "head.log.jsonl")]
    assert any("loss" in r for r in log) and any("val" in r for r in log)
    hc = BB.load_ckpt(run_dir / "head.pt")
    assert hc["kind"] == "lg_head" and hc["meta"]["complete"] and set(hc["trained"]) and "norm.weight" in hc["trained"]
    assert all(not k.startswith("layers.0.") and not k.startswith("layers.1.") for k in hc["trained"])   # trunk [0, ell) untouched
    # second call skips (complete)
    assert HT.train(P, "tiny", 0, hp, device="cpu", max_steps=6, allow_unsealed=True, dtype=torch.float32)["skipped"]
    # ---- router ----
    rr = RT.train(P, "tiny", 0, {"epochs": 1, "batch_rows": 4, "log_every": 2}, device="cpu", allow_unsealed=True, val_max_rows=8, dtype=torch.float32)
    assert (run_dir / "router.pt").exists() and rr["steps"] > 0 and "hit@8" in rr["val_hit"]["all"]
    tdir = run_dir / "router_targets"
    assert (tdir / "train.main.npy").exists() and (tdir / "train.native.npy").exists()
    t = np.load(tdir / "train.main.npy")
    assert t.dtype == np.float16 and t.shape[0] == RT.target_offsets(ps)[-1]
    rc = BB.load_ckpt(run_dir / "router.pt")
    assert rc["head_sha"] == BB.sha256_file(run_dir / "head.pt")
    assert RT.train(P, "tiny", 0, {"epochs": 1}, device="cpu", allow_unsealed=True, dtype=torch.float32)["skipped"]
    # ---- dense-FT: budget derived from head.pt (default: same training tokens; "steps" = same optimizer steps) ----
    hm = BB.load_ckpt(run_dir / "head.pt")["meta"]
    assert hm["tokens_per_step"] == 8 * 256 and hm["n_layers"] == 4
    exp_steps, binfo = DT.budget_steps(hm, dict(C4.DENSE_HP), C4.BACKBONES["tiny"], [ps], [PR.PrepSet(P, "train", "native")], 0, None, 4)
    assert binfo["budget"] == "tokens" and binfo["head_tokens"] == 6 * 8 * 256 and 1 <= exp_steps < 6, binfo
    fl_steps, finfo = DT.budget_steps(hm, dict(C4.DENSE_HP, budget="flops"), C4.BACKBONES["tiny"], [ps], [PR.PrepSet(P, "train", "native")], 0, None, 4)
    assert finfo["budget"] == "flops" and fl_steps <= exp_steps, finfo                    # dense passes every layer 3x
    dr = DT.train(P, "tiny", 0, {"log_every": 2, "eval_every": 3, "budget": "steps"}, device="cpu", allow_unsealed=True, rows_per_step=4,
                  val_max_rows=6, dtype=torch.float32)
    assert dr["steps"] == 6 and (run_dir / "dense_ft.pt").exists()
    assert BB.load_ckpt(run_dir / "dense_ft.pt")["meta"]["budget"]["budget"] == "steps"
    dc = BB.load_ckpt(run_dir / "dense_ft.pt")
    assert dc["kind"] == "dense_ft" and any(k.startswith("layers.0.") for k in dc["trained"])          # dense trains every layer
    h = json.loads((run_dir / "hashes.json").read_text())
    assert {"head.pt", "router.pt", "dense_ft.pt"} <= set(h)
    # ---- reload equivalence: LGRunner from the checkpoint scores like the in-memory modules ----
    model, tok2, head, router, cfg, info = BB.load_lg(run_dir, device="cpu", k=4, dtype=torch.float32)
    runner = LGRunner.from_modules(model, tok, cfg, router, head, method="lg", device="cpu")
    val = PR.PrepSet(P, "val", "main")
    r0 = runner.score_ids(torch.from_numpy(val.row_ids(0)), tuple(val.meta[0]["needle_tok_span"]))
    assert r0["hit"] is not None and r0["n_deep"] <= 4 and 0.0 <= r0["score"] <= 1.0
    model2, _, head2, router2, cfg2, _ = BB.load_lg(run_dir, device="cpu", k=4, dtype=torch.float32)
    lg2 = LandmarkGuard(model2, cfg2, router2, head2).eval()
    o2 = lg2(torch.from_numpy(val.row_ids(0)))
    assert abs(o2.score() - r0["score"]) < 1e-5                                                          # deterministic reload
    # ---- valcheck (calib grid of this fixture is L=1100 words) ----
    rep = VC.run(P, "tiny", 0, device="cpu", k=4, max_val_rows=6, max_calib_rows=6, calib_L=1100, dtype=torch.float32)
    assert "val_hit" in rep and "calib_hit" in rep and rep["matched"] is not None and "recall@5pct" in rep["matched"]
    assert (run_dir / "val_report.json").exists() and isinstance(rep["pass"], bool)
    # ---- E0-7 flops on the tiny backbone: analytic within 10 % of FlopCounterMode ----
    fr = FL.run(P, "tiny", [600, 1400], k=4, run_dir=run_dir, device="cpu", dtype=torch.float32)
    assert fr["pass"], fr["rows"]
    assert (P.reports / "e07_flops.md").exists()
    deep = [r["measured_deep"] for r in fr["rows"]]
    assert abs(deep[0] - deep[1]) / max(deep) < 0.05                                                     # constant deep work in L


def test_lora_mode_trains_only_adapters(s4root):
    P, tok, repo = s4root
    for split, subset in [("train", "main"), ("train", "native"), ("val", "main")]:
        PR.prep_subset(P, split, subset, tok, check_ref=False)
    res = HT.train(P, "tiny_lora", 1, {"batch_windows": 8, "log_every": 2, "eval_every": 0, "epochs": 1}, device="cpu", max_steps=3,
                   allow_unsealed=True, dtype=torch.float32, val_max_rows=4)
    hc = BB.load_ckpt(P.run_dir("tiny_lora", 1) / "head.pt")
    assert hc["mode"] == "lora" and all(".lora_A." in k or ".lora_B." in k for k in hc["trained"])
    assert all(int(k.split(".")[2]) >= 2 for k in hc["trained"])                                          # only layers >= ell
    model, _, head, router, cfg, _ = BB.load_lg(P.run_dir("tiny_lora", 1), device="cpu", k=4, dtype=torch.float32)
    from lgs4.lora import lora_modules
    assert lora_modules(model) and router is None


def test_s4ctl_dry_run_and_status(s4root, monkeypatch, capsys):
    import s4ctl
    P, tok, repo = s4root
    monkeypatch.setattr(s4ctl.s1ctl, "_slurm_cfg", lambda: {"qos": "normal", "gres": "gpu:1", "cpus": "4", "mem": "32G", "nodelist": "",
                                                            "cpu_partitions": "", "cpu_qos": "", "cpu_gres": ""})
    monkeypatch.setattr(s4ctl.s1ctl, "partitions_for", lambda gb, bf16=None: "A5000")
    rc = s4ctl.main(["head", "--backbone", "qwen3guard_0.6b", "--seed", "1", "--sbatch", "--dry-run-sbatch", "--allow-unsealed"])
    out = capsys.readouterr().out
    assert rc == 0 and "lgs4.head_train --backbone qwen3guard_0.6b --seed 1 --allow-unsealed" in out and "s4_head_qwen3guard_0.6b_s1" in out
    assert "#SBATCH --signal=B:TERM@1500" in out and "--time=24:00:00" in out         # batch shell signalled 25 min before the limit
    assert "#SBATCH --requeue" in out and "--open-mode=append" in out and "scontrol requeue" in out
    rc = s4ctl.main(["router", "--backbone", "qwen3guard_0.6b", "--seed", "1", "--sbatch", "--dry-run-sbatch", "--allow-unsealed"])
    out = capsys.readouterr().out
    assert rc == 0 and "--signal" not in out and "--time=12:00:00" in out
    rc = s4ctl.main(["t1", "--seeds", "2", "--from", "router", "--sbatch", "--dry-run-sbatch", "--allow-unsealed"])
    out = capsys.readouterr().out
    assert rc == 0 and "lgs4.head_train" not in out and "lgs4.router_train --backbone qwen3guard_0.6b --seed 2" in out \
        and "lgs4.valcheck --backbone qwen3guard_0.6b --seed 2" in out
    with pytest.raises(SystemExit):                                                                        # sealed check
        s4ctl.main(["head", "--sbatch", "--dry-run-sbatch"])
    assert s4ctl.main(["status"]) == 0
    assert "sealed=NO" in capsys.readouterr().out


def test_training_forward_equals_inference_path_and_freezes_trunk(tmp_path):
    """E0-1 for S4: lg_window_logits (trunk no-grad -> deep grad, optional checkpointing) == lgcore trunk_windows + head_forward,
    and gradients reach only layers >= ell (+ norm + probe)."""
    from lgs1.lgcore.model import Head, tiny_qwen3, trunk_windows, head_forward
    from lgs1.lgcore.windows import make_windows
    torch.manual_seed(0)
    model = tiny_qwen3(seed=0, vocab=512).float()
    bb = C4.Backbone("tiny", "x", ell=2, mode="full", grad_ckpt=True)
    groups = BB.prepare_lg_trainable(model, bb, 2)
    head = Head(64)
    doc = torch.randint(3, 500, (700,))
    wins = make_windows(700, 128)
    hid, valid = trunk_windows(model, doc, wins, ell=2, pad_id=0)
    ref = head_forward(model, hid, valid, 2, head)
    from lgs1.lgcore.model import gather_windows
    ids, valid2 = gather_windows(doc, wins, 0)
    out = BB.lg_window_logits(model, head, ids, valid2, ell=2, grad_ckpt=True, autocast=False)
    assert torch.allclose(out, ref, atol=1e-4), (out, ref)
    loss = out.sum()
    loss.backward()
    dec = model.model
    assert all(p.grad is None for p in dec.layers[0].parameters()) and all(p.grad is None for p in dec.layers[1].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.layers[2].parameters())
    assert head.proj.weight.grad is not None and dec.norm.weight.grad is not None
    assert dec.embed_tokens.weight.grad is None


def test_head_and_dense_resume_after_interrupt(s4root, monkeypatch):
    """2026-09-19: a resume checkpoint (fp32 master weights + optimizer + scheduler + RNG + position) is written every
    ckpt_every_sec seconds and on SIGTERM; a resubmitted job continues from it and ends with the same weights as an
    uninterrupted run (same data order, same RNG)."""
    from lgs4 import common as CM
    P, tok, repo = s4root
    for split, subset in [("train", "main"), ("train", "native"), ("val", "main")]:
        PR.prep_subset(P, split, subset, tok, check_ref=False)
    hp = {"batch_windows": 8, "log_every": 2, "eval_every": 0, "epochs": 2, "ckpt_every_sec": 10 ** 9}
    # reference: uninterrupted run
    ref = HT.train(P, "tiny", 3, hp, device="cpu", max_steps=8, allow_unsealed=True, dtype=torch.float32, val_max_rows=4)
    ref_probe = BB.load_ckpt(P.run_dir("tiny", 3) / "head.pt")["probe"]
    ref_trained = BB.load_ckpt(P.run_dir("tiny", 3) / "head.pt")["trained"]
    # interrupted run: the 4th `due()` call behaves like SIGTERM (checkpoint + exit rc=3)
    calls = {"n": 0}

    def fake_due(self):
        calls["n"] += 1
        if calls["n"] == 4:
            self.stop, self.signal_name = True, "TEST"
            return True
        return False
    monkeypatch.setattr(CM.Checkpointer, "due", fake_due)
    with pytest.raises(SystemExit) as e:
        HT.train(P, "tiny", 4, hp, device="cpu", max_steps=8, allow_unsealed=True, dtype=torch.float32, val_max_rows=4)
    assert e.value.code == 3
    run_dir = P.run_dir("tiny", 4)
    assert (run_dir / "head.resume.pt").exists() and not (run_dir / "head.pt").exists()
    st = BB.load_ckpt(run_dir / "head.resume.pt")["state"]
    assert st["step"] == 4 and st["total_steps"] == 8
    monkeypatch.setattr(CM.Checkpointer, "due", lambda self: False)
    res = HT.train(P, "tiny", 4, hp, device="cpu", max_steps=8, allow_unsealed=True, dtype=torch.float32, val_max_rows=4)
    assert res["steps"] == 8 and not (run_dir / "head.resume.pt").exists()
    log_txt = (run_dir / "head.log.jsonl").read_text()
    assert '"step": 8' in log_txt
    # seeds differ (3 vs 4) only in probe init / data order, so compare the interrupted run against ITS OWN uninterrupted twin
    ref4 = HT.train(P, "tiny", 4, hp, device="cpu", max_steps=8, allow_unsealed=True, dtype=torch.float32, val_max_rows=4)
    assert ref4["skipped"]                                           # complete -> skip; recompute in a fresh run dir instead
    import shutil
    shutil.move(run_dir / "head.pt", run_dir / "head.resumed.pt")
    ref4 = HT.train(P, "tiny", 4, hp, device="cpu", max_steps=8, allow_unsealed=True, dtype=torch.float32, val_max_rows=4)
    a, b = BB.load_ckpt(run_dir / "head.resumed.pt"), BB.load_ckpt(run_dir / "head.pt")
    for k in a["probe"]:
        assert torch.allclose(a["probe"][k], b["probe"][k], atol=1e-5), k
    for k in list(a["trained"])[:8]:
        assert torch.allclose(a["trained"][k].float(), b["trained"][k].float(), atol=1e-2), k     # bf16-rounded copies
    assert ref["steps"] == 8 and ref_probe and ref_trained
    # dense: interrupt + resume, same mechanics
    calls["n"] = 0
    monkeypatch.setattr(CM.Checkpointer, "due", fake_due)
    with pytest.raises(SystemExit) as e:
        DT.train(P, "tiny", 4, {"log_every": 2, "eval_every": 0, "ckpt_every_sec": 10 ** 9}, device="cpu", steps=8, allow_unsealed=True,
                 rows_per_step=2, val_max_rows=4, dtype=torch.float32)
    assert e.value.code == 3 and (run_dir / "dense.resume.pt").exists()
    monkeypatch.setattr(CM.Checkpointer, "due", lambda self: False)
    dr = DT.train(P, "tiny", 4, {"log_every": 2, "eval_every": 0, "ckpt_every_sec": 10 ** 9}, device="cpu", steps=8, allow_unsealed=True,
                  rows_per_step=2, val_max_rows=4, dtype=torch.float32)
    assert dr["steps"] == 8 and (run_dir / "dense_ft.pt").exists() and not (run_dir / "dense.resume.pt").exists()


def test_router_row_subsample_and_partwise_target_cache(s4root):
    """2026-09-20: the router trains on a seeded subsample of the head's rows (all windows of each), targets are cached
    in 2,000-row parts (here 5) and a killed job picks the finished parts up again."""
    P, tok, repo = s4root
    for split, subset in [("train", "main"), ("train", "native")]:
        PR.prep_subset(P, split, subset, tok, check_ref=False)
    ps, nat = PR.PrepSet(P, "train", "main"), PR.PrepSet(P, "train", "native")
    hp = dict(C4.ROUTER_HP, router_rows=10, native_ratio=0.5)
    pres_rows, nat_rows = RT.select_rows([ps], [nat], hp, seed=0)
    assert len(pres_rows[0]) == 10 and len(nat_rows[0]) == 5 and pres_rows[0] == sorted(set(pres_rows[0]))
    pres2, _ = RT.select_rows([ps], [nat], hp, seed=0)
    assert pres2 == pres_rows                                                        # seeded -> reproducible
    pres_all, nat_all = RT.select_rows([ps], [nat], dict(hp, router_rows=None), seed=0)
    assert len(pres_all[0]) == len(ps) and len(nat_all[0]) == min(len(nat), round(0.5 * len(ps)))
    out = P.run_dir("tiny", 9) / "router_targets" / "train.main.npy"
    rows = pres_rows[0]
    arr = RT.compute_targets(None, None, ps, rows, ell=2, pad_id=0, device="cpu", out_path=out, mode="label", part_rows=4)
    assert arr.shape[0] == RT.target_offsets(ps, rows)[-1] and not list(out.parent.glob("*.part*.npy"))
    # simulate a kill after part 0: keep part0, drop the merged file -> the rerun reuses part0 and finishes
    out.unlink()
    part0 = RT.compute_targets(None, None, ps, rows[:4], ell=2, pad_id=0, device="cpu",
                               out_path=out.with_name("tmp.npy"), mode="label", part_rows=4)
    np.save(out.with_name("train.main.part0.npy"), part0)
    out.with_name("tmp.npy").unlink()
    arr2 = RT.compute_targets(None, None, ps, rows, ell=2, pad_id=0, device="cpu", out_path=out, mode="label", part_rows=4)
    assert np.array_equal(arr, arr2) and out.exists() and not list(out.parent.glob("*.part*.npy"))


def test_landmark_probe_head_v2(s4root, monkeypatch, capsys):
    """Head v2 (2026-09-21): probe='landmark' appends a learned end-of-window vector after the last content token at the trunk
    depth; training forward == inference forward, the landmark gets gradients, content tokens are untouched, the variant run
    trains next to the default run and is addressed by its directory name (runs/tiny_lm_s5)."""
    from lgs1.lgcore.model import Head, LandmarkGuard, LGConfig, gather_windows, head_forward, tiny_qwen3, trunk_windows
    from lgs1.lgcore.windows import make_windows
    P, tok, repo = s4root
    torch.manual_seed(0)
    model = tiny_qwen3(seed=0, vocab=512).float()
    bb = C4.Backbone("tiny", "x", ell=2, mode="full", grad_ckpt=False)
    BB.prepare_lg_trainable(model, bb, 2)
    head = Head(64, probe="landmark")
    assert head.landmark.shape == (64,) and "landmark" in head.state_dict()
    with torch.no_grad():
        head.landmark.normal_()
    doc = torch.randint(3, 500, (700,))
    wins = make_windows(700, 128)
    hid, valid = trunk_windows(model, doc, wins, ell=2, pad_id=0)
    # inject: one extra position per window, the landmark sits right after the last valid token
    h2, v2 = head.inject(hid, valid)
    assert h2.shape == (hid.shape[0], hid.shape[1] + 1, 64) and v2.shape == (valid.shape[0], valid.shape[1] + 1)
    n = valid.sum(1)
    for w in range(hid.shape[0]):
        assert torch.allclose(h2[w, n[w]], head.landmark) and v2[w, n[w]] == 1 and v2[w].sum() == n[w] + 1
        assert torch.equal(h2[w, : n[w]], hid[w, : n[w]])
    ref = head_forward(model, hid, valid, 2, head)
    ids, valid2 = gather_windows(doc, wins, 0)
    out = BB.lg_window_logits(model, head, ids, valid2, ell=2, grad_ckpt=False, autocast=False)
    assert out.shape == (len(wins),) and torch.allclose(out, ref, atol=1e-4)
    out.sum().backward()
    assert head.landmark.grad is not None and head.landmark.grad.abs().sum() > 0
    # probe='last' is byte-for-byte the v1 head (no landmark parameter, inject is the identity)
    h1 = Head(64)
    assert not hasattr(h1, "landmark") or h1.landmark is None
    assert h1.inject(hid, valid)[0] is hid
    with pytest.raises(ValueError):
        Head(64, probe="cls")
    # ---- variant chain on the tiny data: head (landmark) -> router -> valcheck in runs/tiny_lm_s5 ----
    for split, subset in [("train", "main"), ("train", "native"), ("val", "main"), ("calib", "main"), ("calib", "benign_long")]:
        PR.prep_subset(P, split, subset, tok, check_ref=False)
    hp = {"batch_windows": 8, "log_every": 2, "eval_every": 0, "epochs": 1, "probe": "landmark"}
    res = HT.train(P, "tiny", 5, hp, device="cpu", max_steps=4, allow_unsealed=True, dtype=torch.float32, val_max_rows=4, variant="lm")
    run_dir = P.run_dir("tiny", 5, "lm")
    assert run_dir.name == "tiny_lm_s5" and (run_dir / "head.pt").exists() and res["steps"] == 4
    assert not (P.run_dir("tiny", 5) / "head.pt").exists()                                  # the default run is untouched
    hc = BB.load_ckpt(run_dir / "head.pt")
    assert hc["head_probe"] == "landmark" and "landmark" in hc["probe"] and hc["hp"]["probe"] == "landmark"
    m2, _, head2, _, cfg2, _ = BB.load_lg(run_dir, device="cpu", with_router=False, dtype=torch.float32)
    assert head2.probe == "landmark" and torch.allclose(head2.landmark, hc["probe"]["landmark"])
    rr = RT.train(P, "tiny", 5, {"epochs": 1, "batch_rows": 4, "log_every": 2}, device="cpu", limit_rows=None, variant="lm",
                  allow_unsealed=True, val_max_rows=4, dtype=torch.float32)
    assert (run_dir / "router.pt").exists() and rr["steps"] > 0
    m3, _, head3, router3, cfg3, _ = BB.load_lg(run_dir, device="cpu", dtype=torch.float32)
    lg = LandmarkGuard(m3, cfg3, router3, head3).eval()
    o = lg(doc)
    assert o.head_probs.shape[0] == o.n_deep > 0 and 0.0 <= o.score() <= 1.0
    # s4ctl: --variant/--probe reach every stage, job names carry the variant, status lists the variant run
    import s4ctl
    monkeypatch.setattr(s4ctl.s1ctl, "_slurm_cfg", lambda: {"qos": "normal", "gres": "gpu:1", "cpus": "4", "mem": "32G", "nodelist": "",
                                                            "cpu_partitions": "", "cpu_qos": "", "cpu_gres": ""})
    monkeypatch.setattr(s4ctl.s1ctl, "partitions_for", lambda gb, bf16=None: "A5000")
    rc = s4ctl.main(["t1", "--seeds", "0", "--variant", "lm", "--probe", "landmark", "--sbatch", "--dry-run-sbatch", "--allow-unsealed"])
    out = capsys.readouterr().out
    assert rc == 0 and "lgs4.head_train --backbone qwen3guard_0.6b --seed 0 --variant lm --probe landmark" in out
    assert "lgs4.router_train --backbone qwen3guard_0.6b --seed 0 --variant lm" in out and "--probe" not in out.split("lgs4.router_train")[1].split("\n")[0]
    assert "s4_head_qwen3guard_0.6b_lm_s0" in out and "s4_valcheck_qwen3guard_0.6b_lm_s0" in out
    assert s4ctl.main(["status"]) == 0
    assert "tiny_lm_s5" in capsys.readouterr().out
