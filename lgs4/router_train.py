"""T1/T3 — router training by head-distillation on the FT checkpoint (plan §3: `--model $LG`, never the base model).

    python -m lgs4.router_train --backbone qwen3guard_0.6b --seed 0            # needs runs/v2/<bb>_s<seed>/head.pt
    python -m lgs4.router_train --backbone qwen3guard_0.6b --seed 0 --limit-rows 300 --epochs 1   (smoke)
    python -m lgs4.router_train --backbone qwen3guard_0.6b --seed 0 --variant pool_mean --pool mean          # E6-4
    python -m lgs4.router_train --backbone qwen3guard_0.6b --seed 0 --variant ell6_12 --router-ells 6,12     # E6-3
    python -m lgs4.router_train --backbone qwen3guard_0.6b --seed 0 --variant mlp_token --arch mlp --objective token_bce   # E6-5

Stage A (targets): every window of the router's training rows through the trained deep head -> P_head(window), cached
        in runs/v2/<run>/router_targets/<split>.<subset>.npy (float16, windows in grid order, rows concatenated).
        The router's rows are a seeded uniform subsample of the head's rows (ROUTER_HP.router_rows presence rows +
        native_ratio x that many native rows): 2026-09-20 the all-rows version (2.1M windows = 550M tokens through the
        full model, then 2 epochs of trunk passes) needed 8-10 h on an RTX 3090 and died at the 4 h limit; a 3k-parameter
        router does not need 2M windows.  Windows are packed across rows into 256-window forwards; the cache is written
        in parts of 2,000 rows so a killed job resumes.
Stage B (fit):   router = LayerNorm -> Linear (or the E6-5 MLP) on the FROZEN trunk features (layers [0, ell), block-local;
        E6-3: the state at other / several depths, concatenated), window score = in-window pooling (max by default; E6-4
        mean / top2 / top4 / lse); objective (E6-5):
            distill_listwise (default)  BCE(score, 1[P_head > 0.5]) + lambda * KL(softmax(logit P_head / T) || softmax(scores)) per row
            distill                     the BCE term alone
            window_bce                  BCE(score, 1[window contains the needle] * label)
            token_bce                   BCE on every token score: 1 on the needle tokens of a harmful row, 0 elsewhere
        Validation: hit@{8,16,32} on val/main (needle window in the top-k).
Output: router.pt (+ router.log.jsonl, hashes.json updated).  The checkpoint carries pool / router_ells / arch so
        `lgs4.backbone.load_lg` rebuilds exactly this router (an S8 arm lives in runs/v2/<bb>_<variant>_s<seed>/ next to a
        symlinked head.pt of the run whose head it routes for).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lgs1.lgcore.model import Head, LGConfig, Router, _inner, make_router, pool_scores, router_in_dim
from lgs1.lgcore.windows import make_windows, windows_containing
from lgs4 import backbone as BB
from lgs4.common import JsonlLog, device_auto, gpu_mem_gb, run_meta, set_seed
from lgs4.config import LG_K_REPORT, NATIVE_SETS, PRESENCE_SETS, ROUTER_HP, VAL_SETS, WINDOW_B, S4Paths, backbone
from lgs4.prep import PrepSet

ROOT = Path(__file__).resolve().parents[1]

# objective -> (target mode, rank weight, token-level loss)
OBJECTIVES: Dict[str, Tuple[str, Optional[float], bool]] = {
    "distill_listwise": ("head", None, False),      # None = ROUTER_HP.lambda_rank (or --lambda-rank)
    "distill": ("head", 0.0, False),
    "window_bce": ("label", 0.0, False),
    "token_bce": ("label", 0.0, True),
}


def objective_settings(hp: Dict) -> Tuple[str, float, bool]:
    obj = hp.get("objective", "distill_listwise")
    if obj not in OBJECTIVES:
        raise SystemExit(f"router objective must be one of {list(OBJECTIVES)}, got {obj!r}")
    mode, lam, token_level = OBJECTIVES[obj]
    return mode, (float(hp["lambda_rank"]) if lam is None else lam), token_level


def parse_ells(s: Optional[str]) -> Optional[List[int]]:
    if s is None or str(s).strip() in ("", "none", "None"):
        return None
    return sorted({int(x) for x in str(s).split(",") if x.strip()})


# ------------------------------------------------------------------------------------------------
# window tensors for one row
# ------------------------------------------------------------------------------------------------
def row_windows(ps: PrepSet, row: int, pad_id: int, b: int = WINDOW_B) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    ids_np = ps.row_ids(row)
    wins = make_windows(len(ids_np), b)
    L = 2 * b
    ids = torch.full((len(wins), L), pad_id, dtype=torch.long)
    valid = torch.zeros((len(wins), L), dtype=torch.long)
    for i, (s, e) in enumerate(wins):
        ids[i, : e - s] = torch.from_numpy(np.asarray(ids_np[s:e], dtype=np.int64))
        valid[i, : e - s] = 1
    return ids, valid, wins


def token_labels(ps: PrepSet, row: int, wins: Sequence[Tuple[int, int]], L: int) -> torch.Tensor:
    """token_bce targets [W, L]: 1 on the needle tokens of a harmful row (0 on a twin's needle tokens and everywhere
    else); a native row (the needle text alone) carries its label on every token."""
    m = ps.meta[row]
    y = torch.zeros((len(wins), L), dtype=torch.float32)
    lab = float(m["label"])
    span = m.get("needle_tok_span")
    if ps.subset == "native" or span is None:
        for i, (s, e) in enumerate(wins):
            y[i, : e - s] = lab
        return y
    if lab == 0.0:
        return y
    a, b_ = int(span[0]), int(span[1])
    for i, (s, e) in enumerate(wins):
        lo, hi = max(a, s), min(b_, e)
        if hi > lo:
            y[i, lo - s:hi - s] = 1.0
    return y


@torch.no_grad()
def trunk_features(model, ids: torch.Tensor, valid: torch.Tensor, ell, micro: int = 64) -> torch.Tensor:
    """Frozen block-local trunk features [W, L, d * n_depths] (bf16 autocast on GPU).  `ell` is one depth or a list of
    depths whose states are concatenated along the hidden dimension (E6-3); one segment-wise pass either way."""
    depths = sorted({int(e) for e in (ell if isinstance(ell, (list, tuple)) else [ell])})
    dec = _inner(model)
    dev_type = "cuda" if ids.is_cuda else "cpu"
    outs = []
    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=dev_type == "cuda"):
        for i in range(0, ids.shape[0], micro):
            v = valid[i:i + micro]
            h = dec.embed_tokens(ids[i:i + micro])
            prev, parts = 0, []
            for e in depths:
                if e > prev:
                    h = BB.run_layers(model, h, v, prev, e, apply_norm=False)
                parts.append(h)
                prev = e
            outs.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1))
    return torch.cat(outs, 0)


# ------------------------------------------------------------------------------------------------
# stage A: head targets
# ------------------------------------------------------------------------------------------------
@torch.no_grad()
def _head_probs_packed(model, head: Head, ps: PrepSet, rows: Sequence[int], ell: int, pad_id: int, device: str,
                       micro: int = 256, b: int = WINDOW_B) -> List[np.ndarray]:
    """P_head for every window of `rows`, windows packed across rows into forwards of ~`micro` windows."""
    out: List[np.ndarray] = [None] * len(rows)          # type: ignore[list-item]
    buf_ids, buf_valid, owners = [], [], []              # owners: (row index in `rows`, n_windows)

    def flush():
        if not buf_ids:
            return
        ids = torch.cat(buf_ids, 0).to(device)
        valid = torch.cat(buf_valid, 0).to(device)
        probs = torch.sigmoid(BB.lg_window_logits(model, head, ids, valid, ell)).float().cpu().numpy().astype(np.float16)
        pos = 0
        for j, n in owners:
            out[j] = probs[pos:pos + n]
            pos += n
        buf_ids.clear(); buf_valid.clear(); owners.clear()

    n_buf = 0
    for j, r in enumerate(rows):
        ids, valid, wins = row_windows(ps, r, pad_id, b)
        buf_ids.append(ids); buf_valid.append(valid); owners.append((j, len(wins)))
        n_buf += len(wins)
        if n_buf >= micro:
            flush()
            n_buf = 0
    flush()
    return out


@torch.no_grad()
def compute_targets(model, head: Head, ps: PrepSet, rows: Sequence[int], ell: int, pad_id: int, device: str, out_path: Path,
                    micro: int = 256, mode: str = "head", part_rows: int = 2000, b: int = WINDOW_B) -> np.ndarray:
    """P_head (or the label target) for every window of `rows` (in that order), cached at `out_path`; while running,
    parts of `part_rows` rows are written next to it (`<name>.partK.npy`) and picked up again after a kill."""
    need = sum(len(make_windows(ps.meta[r]["n_tok"], b)) for r in rows)
    if out_path.exists():
        arr = np.load(out_path)
        if arr.shape[0] == need:
            print(f"[router] targets cached {out_path.name} ({arr.shape[0]:,} windows)", flush=True)
            return arr
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parts: List[np.ndarray] = []
    n_parts = -(-len(rows) // part_rows) if rows else 0
    for k in range(n_parts):
        part_path = out_path.with_name(f"{out_path.stem}.part{k}.npy")
        chunk = rows[k * part_rows:(k + 1) * part_rows]
        if part_path.exists():
            arr_k = np.load(part_path)
            exp = sum(len(make_windows(ps.meta[r]["n_tok"], b)) for r in chunk)
            if arr_k.shape[0] == exp:
                parts.append(arr_k)
                continue
        if mode == "label":                            # ablation: target = needle windows (label) only
            outs = []
            for r in chunk:
                m = ps.meta[r]
                wins = make_windows(m["n_tok"], b)
                sp = m.get("needle_tok_span")
                nw = set(windows_containing((sp[0], sp[1]), wins)) if sp else set(range(len(wins)))
                outs.append(np.asarray([float(m["label"]) if i in nw else 0.0 for i in range(len(wins))], dtype=np.float16))
        else:
            outs = _head_probs_packed(model, head, ps, chunk, ell, pad_id, device, micro=micro, b=b)
        arr_k = np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float16)
        np.save(part_path, arr_k)
        parts.append(arr_k)
        print(f"   targets {ps.split}/{ps.subset}: {min(len(rows), (k + 1) * part_rows):,}/{len(rows):,} rows  {time.time() - t0:.0f}s", flush=True)
    arr = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float16)
    np.save(out_path, arr)
    for k in range(n_parts):
        pp = out_path.with_name(f"{out_path.stem}.part{k}.npy")
        if pp.exists():
            pp.unlink()
    print(f"[router] targets {ps.split}/{ps.subset}: {arr.shape[0]:,} windows over {len(rows):,} rows, "
          f"mean P={float(arr.astype(np.float32).mean()) if arr.size else 0:.3f} {time.time() - t0:.0f}s", flush=True)
    return arr


def target_offsets(ps: PrepSet, rows: Optional[Sequence[int]] = None, b: int = WINDOW_B) -> np.ndarray:
    """Window offsets of `rows` (default: every row of the set) inside a concatenated target array."""
    rows = range(len(ps)) if rows is None else rows
    counts = [len(make_windows(ps.meta[r]["n_tok"], b)) for r in rows]
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def select_rows(presence: Sequence[PrepSet], native: Sequence[PrepSet], hp: Dict, seed: int,
                limit_rows: Optional[int] = None) -> Tuple[List[List[int]], List[List[int]]]:
    """Seeded uniform subsample: up to hp['router_rows'] presence rows spread over the presence sets in proportion to
    their size (all windows of each chosen row), plus native_ratio x that many native rows.  `limit_rows` (smoke
    tests) first restricts every set to its first rows."""
    rng = random.Random(f"{seed}:router:rows")
    pres_all = [list(range(min(len(ps), limit_rows) if limit_rows else len(ps))) for ps in presence]
    n_total = sum(len(x) for x in pres_all)
    cap = hp.get("router_rows") or n_total
    if cap < n_total:
        pres_rows = []
        for x in pres_all:
            n_take = max(1, round(cap * len(x) / n_total)) if x else 0
            pres_rows.append(sorted(rng.sample(x, min(n_take, len(x)))))
    else:
        pres_rows = pres_all
    n_pres = sum(len(x) for x in pres_rows)
    nat_all = [list(range(min(len(ps), limit_rows) if limit_rows else len(ps))) for ps in native]
    n_nat_total = sum(len(x) for x in nat_all)
    n_nat = min(n_nat_total, int(round(hp["native_ratio"] * n_pres)))
    nat_rows = []
    for x in nat_all:
        n_take = round(n_nat * len(x) / n_nat_total) if n_nat_total else 0
        nat_rows.append(sorted(rng.sample(x, min(n_take, len(x)))))
    return pres_rows, nat_rows


# ------------------------------------------------------------------------------------------------
# stage B: fit
# ------------------------------------------------------------------------------------------------
def router_loss(scores: torch.Tensor, p_head: torch.Tensor, hp: Dict, lambda_rank: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """scores [W] router window scores; p_head [W] head probabilities (or 0/1 labels).  Returns (total, bce, rank)."""
    lam = float(hp["lambda_rank"]) if lambda_rank is None else float(lambda_rank)
    y = (p_head > hp["pos_threshold"]).float()
    bce = F.binary_cross_entropy_with_logits(scores, y)
    if scores.numel() > 1 and lam > 0:
        eps = 1e-4
        head_logit = torch.log(p_head.clamp(eps, 1 - eps) / (1 - p_head.clamp(eps, 1 - eps)))
        target = F.softmax(head_logit / hp["temperature"], dim=0)
        rank = F.kl_div(F.log_softmax(scores, dim=0), target, reduction="sum")
    else:
        rank = torch.zeros((), device=scores.device)
    return bce + lam * rank, bce, rank


def token_loss(token_scores: torch.Tensor, y_tok: torch.Tensor, valid: torch.Tensor, pos_weight_cap: float = 8.0) -> torch.Tensor:
    """token_bce: BCE over the valid tokens of a row's windows, positives (needle tokens) re-weighted n_neg / n_pos (capped)."""
    v = valid.bool()
    s, y = token_scores[v], y_tok[v]
    if s.numel() == 0:
        return token_scores.sum() * 0.0
    n_pos = float(y.sum())
    pw = torch.tensor(min(pos_weight_cap, (y.numel() - n_pos) / n_pos) if n_pos > 0 else 1.0, device=s.device)
    return F.binary_cross_entropy_with_logits(s, y, pos_weight=pw)


@torch.no_grad()
def hit_at_k(model, router, val: PrepSet, ell, pad_id: int, device: str, ks: Sequence[int] = LG_K_REPORT,
             max_rows: int = 400, seed: int = 0, pool: str = "max", b: int = WINDOW_B) -> Dict:
    rng = random.Random(seed)
    rows = [i for i, m in enumerate(val.meta) if m["label"] == 1 and m.get("needle_tok_span")]
    rng.shuffle(rows)
    rows = rows[:max_rows]
    per_L: Dict = {}
    for r in rows:
        m = val.meta[r]
        ids, valid, wins = row_windows(val, r, pad_id, b)
        h = trunk_features(model, ids.to(device), valid.to(device), ell)
        sc = pool_scores(router(h.float()), valid.to(device), pool)
        nw = windows_containing((m["needle_tok_span"][0], m["needle_tok_span"][1]), wins)
        order = torch.argsort(sc, descending=True).tolist()
        key = m["L_target"]
        rec = per_L.setdefault(key, {f"hit@{k}": 0 for k in ks} | {"n": 0})
        rec["n"] += 1
        for k in ks:
            if any(w in order[:k] for w in nw):
                rec[f"hit@{k}"] += 1
    out = {}
    for L, rec in sorted(per_L.items(), key=lambda kv: kv[0] or 0):
        out[str(L)] = {f"hit@{k}": rec[f"hit@{k}"] / rec["n"] for k in ks} | {"n": rec["n"]}
    n_all = sum(v["n"] for v in per_L.values())
    out["all"] = {f"hit@{k}": sum(v[f"hit@{k}"] for v in per_L.values()) / max(1, n_all) for k in ks} | {"n": n_all}
    return out


def train(P: S4Paths, bb_name: str, seed: int, hp: Optional[Dict] = None, device: str = "auto", limit_rows: Optional[int] = None, variant: str = "",
          repo_override: Optional[str] = None, allow_unsealed: bool = False, val_max_rows: int = 400,
          dtype: torch.dtype = torch.bfloat16) -> Dict:
    hp = dict(ROUTER_HP, **(hp or {}))
    bb = backbone(bb_name)
    if not P.sealed() and not allow_unsealed:
        raise SystemExit(f"{P.data.root}/SEALED missing (G0); pass --allow-unsealed for a smoke run only")
    device = device_auto(device)
    run_dir = P.run_dir(bb_name, seed, variant)
    head_path = run_dir / BB.HEAD_FILE
    if not head_path.exists():
        raise SystemExit(f"{head_path} missing: train the head first (python s4ctl.py head --backbone {bb_name} --seed {seed})")
    router_path = run_dir / BB.ROUTER_FILE
    tmode, lam, token_level = objective_settings(hp)
    hp["target"], hp["lambda_rank"] = tmode, lam
    router_ells = parse_ells(hp.get("router_ells")) if not isinstance(hp.get("router_ells"), list) else sorted(set(hp["router_ells"]))
    hp["router_ells"] = router_ells
    if router_path.exists():
        prev = BB.load_ckpt(router_path)
        if prev.get("meta", {}).get("complete") and prev.get("head_sha") == BB.sha256_file(head_path):
            print(f"[router] {router_path} already trained on this head -> skip (delete it to retrain)")
            return {"skipped": True, "path": str(router_path)}
    set_seed(seed)
    model, tok, head, _, cfg, info = BB.load_lg(run_dir, device=device, repo_override=repo_override, with_router=False, dtype=dtype)
    cfg = dataclasses.replace(cfg, pool=hp.get("pool", "max"), router_ells=router_ells)
    ell, pad_id, d, b = cfg.ell, cfg.pad_id, BB.hidden_size(model), int(cfg.b)
    if cfg.trunk_depth >= BB.n_layers(model):
        raise SystemExit(f"router depth {cfg.trunk_depth} >= n_layers {BB.n_layers(model)}")
    feat_ells = cfg.router_depths
    presence = [PrepSet(P, s, u) for s, u in PRESENCE_SETS if (P.prep / s / u / "ids.npy").exists()]
    native = [PrepSet(P, s, u) for s, u in NATIVE_SETS if (P.prep / s / u / "ids.npy").exists()]
    val = PrepSet(P, *VAL_SETS[0]) if (P.prep / VAL_SETS[0][0] / VAL_SETS[0][1] / "ids.npy").exists() else None
    sets = presence + native
    pres_rows, nat_rows = select_rows(presence, native, hp, seed, limit_rows)
    set_rows = pres_rows + nat_rows
    n_win = sum(len(make_windows(ps.meta[r]["n_tok"], b)) for ps, rows in zip(sets, set_rows) for r in rows)
    print(f"[router] rows: presence {sum(len(x) for x in pres_rows):,} of {sum(len(ps) for ps in presence):,}, "
          f"native {sum(len(x) for x in nat_rows):,} of {sum(len(ps) for ps in native):,} -> {n_win:,} windows "
          f"(router_rows={hp.get('router_rows')}, native_ratio={hp['native_ratio']})", flush=True)
    # ---- stage A ----
    tdir = run_dir / "router_targets"
    targets = [compute_targets(model, head, ps, rows, ell, pad_id, device, tdir / f"{ps.split}.{ps.subset}.npy", mode=hp["target"], b=b)
               for ps, rows in zip(sets, set_rows)]
    offs = [target_offsets(ps, rows, b) for ps, rows in zip(sets, set_rows)]
    pos_in_set = [{r: j for j, r in enumerate(rows)} for rows in set_rows]
    # ---- stage B ----
    set_seed(seed)
    router = make_router(d, hp.get("arch", "linear"), router_in_dim(cfg, d), int(hp.get("mlp_hidden", 256))).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    log = JsonlLog(run_dir / "router.log.jsonl")
    units = [(si, r) for si, rows in enumerate(set_rows) for r in rows]
    steps_per_epoch = -(-len(units) // hp["batch_rows"])
    total = steps_per_epoch * hp["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 if total <= 1 else max(0.05, 1.0 - s / total))
    print(f"[router] {bb_name} s{seed} variant={variant or '-'} rows={len(units):,} batch_rows={hp['batch_rows']} steps={total:,} "
          f"objective={hp.get('objective', 'distill_listwise')} (target={hp['target']} lambda_rank={hp['lambda_rank']} token={token_level}) "
          f"pool={cfg.pool} arch={getattr(router, 'arch', 'linear')} depths={feat_ells} b={b} params={sum(p.numel() for p in router.parameters()):,} "
          f"device={device}", flush=True)
    t0 = time.time()
    step = 0
    acc_loss = acc_bce = acc_rank = 0.0
    acc_n = 0
    hits: Dict = {}
    L = 2 * b
    for epoch in range(hp["epochs"]):
        rng = random.Random(f"{seed}:router:{epoch}")
        order = list(units)
        rng.shuffle(order)
        for bi in range(0, len(order), hp["batch_rows"]):
            batch = order[bi:bi + hp["batch_rows"]]
            opt.zero_grad(set_to_none=True)
            # one trunk pass for all windows of the batch's rows (windows packed across rows), then per-row losses
            packed = [row_windows(sets[si], r, pad_id, b) for si, r in batch]
            ids_all = torch.cat([x[0] for x in packed], 0).to(device)
            valid_all = torch.cat([x[1] for x in packed], 0).to(device)
            h_all = trunk_features(model, ids_all, valid_all, feat_ells)
            tok_all = router(h_all.float())
            scores_all = pool_scores(tok_all, valid_all, cfg.pool)
            tot = torch.zeros((), device=device)
            pos = 0
            for (si, r), (ids, valid, wins) in zip(batch, packed):
                scores = scores_all[pos:pos + len(wins)]
                j = pos_in_set[si][r]
                p = torch.from_numpy(targets[si][offs[si][j]:offs[si][j + 1]].astype(np.float32)).to(device)
                if p.numel() != scores.numel():
                    raise RuntimeError(f"target/window count mismatch row {sets[si].meta[r]['id']}: {p.numel()} vs {scores.numel()}")
                if token_level:
                    y_tok = token_labels(sets[si], r, wins, L).to(device)
                    loss = token_loss(tok_all[pos:pos + len(wins)], y_tok, valid.to(device))
                    bce, rank = loss.detach(), torch.zeros((), device=device)
                else:
                    loss, bce, rank = router_loss(scores, p, hp)
                pos += len(wins)
                tot = tot + loss / len(batch)
                acc_loss += float(loss.detach()); acc_bce += float(bce.detach()); acc_rank += float(rank.detach()); acc_n += 1
            tot.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % hp["log_every"] == 0 or step == total:
                rec = {"step": step, "epoch": epoch, "loss": acc_loss / max(1, acc_n), "bce": acc_bce / max(1, acc_n),
                       "rank": acc_rank / max(1, acc_n), "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
                log.write(**rec)
                print(f"   step {step}/{total} loss={rec['loss']:.4f} bce={rec['bce']:.4f} rank={rec['rank']:.4f} {rec['sec']}s", flush=True)
                acc_loss = acc_bce = acc_rank = 0.0
                acc_n = 0
        if val is not None:
            hits = hit_at_k(model, router, val, feat_ells, pad_id, device, max_rows=val_max_rows, seed=seed, pool=cfg.pool, b=b)
            log.write(step=step, epoch=epoch, val_hit=hits)
            print(f"   val hit@k after epoch {epoch}: {hits.get('all')}", flush=True)
    meta = run_meta(ROOT, {"backbone": bb_name, "seed": seed, "variant": variant, "ell": ell, "router_ells": router_ells, "pool": cfg.pool,
                           "arch": getattr(router, "arch", "linear"), "objective": hp.get("objective", "distill_listwise"),
                           "steps": step, "rows": len(units), "windows": n_win, "router_rows": hp.get("router_rows"),
                           "seconds": round(time.time() - t0, 1), "val_hit": hits,
                           "targets": {f"{ps.split}/{ps.subset}": int(t.shape[0]) for ps, t in zip(sets, targets)},
                           "gpu_gb": gpu_mem_gb(), "complete": True})
    BB.save_router_ckpt(router_path, router, bb, cfg, head_path, hp, meta)
    BB.write_hashes(run_dir)
    log.write(step=step, done=True)
    log.close()
    print(f"[router] saved {router_path} ({step} steps, {meta['seconds']}s) hit={hits.get('all')}", flush=True)
    return {"path": str(router_path), "steps": step, "val_hit": hits}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S4 router training (head-distillation)")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--target", default=None, choices=[None, "head", "label"], help="(legacy) head = distil, label = window_bce")
    ap.add_argument("--objective", default=None, choices=[None] + list(OBJECTIVES), help="E6-5 (default distill_listwise)")
    ap.add_argument("--lambda-rank", type=float, default=None)
    ap.add_argument("--router-rows", type=int, default=None, help="presence rows for the router (default ROUTER_HP.router_rows)")
    ap.add_argument("--pool", default=None, choices=[None, "max", "mean", "top2", "top4", "lse"], help="E6-4 in-window pooling (default max)")
    ap.add_argument("--arch", default=None, choices=[None, "linear", "mlp"], help="E6-5 router capacity (default linear)")
    ap.add_argument("--mlp-hidden", type=int, default=None)
    ap.add_argument("--router-ells", default=None, help="E6-3 trunk depths the router reads, e.g. 6 or 6,12 (default: the head's ell)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--allow-unsealed", action="store_true")
    ap.add_argument("--variant", default="", help="run-dir suffix (see head_train --variant)")
    a = ap.parse_args(argv)
    P = S4Paths.load().ensure().for_backbone(a.backbone)      # prep arrays of the backbone's tokenizer family (T5: llama3)
    hp = {k: v for k, v in {"epochs": a.epochs, "lambda_rank": a.lambda_rank, "router_rows": a.router_rows, "pool": a.pool,
                            "arch": a.arch, "mlp_hidden": a.mlp_hidden, "router_ells": a.router_ells, "objective": a.objective}.items()
          if v is not None}
    if a.target and not a.objective:                               # legacy flag
        hp["objective"] = "window_bce" if a.target == "label" else "distill_listwise"
    train(P, a.backbone, a.seed, hp, a.device, a.limit_rows, a.variant, a.repo, a.allow_unsealed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
