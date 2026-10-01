"""Hard-negative mining for the LG head (S8 E6-6 arm `hardneg`).

    python -m lgs4.hardneg --backbone qwen3guard_0.6b --seed 0                 # base run -> runs/v2/<run>/hardneg/{train.main,train.main16k}.npz
    python -m lgs4.hardneg --backbone qwen3guard_0.6b --seed 0 --m-router 8 --sets train/main,train/main16k

Why: the 128k drop of the 0.6B head (S6, 9/23) is threshold inflation -- the k=32 max over a long benign document rises with
L because the router hands the head the 32 most suspicious background windows, and the head (trained on random background
windows) scores some of them high.  The head never saw those windows as negatives.  Mining makes them training negatives:
for every presence row (train/main, train/main16k) the BASE run's router ranks the non-needle windows, the top `m_router`
go through the base head, and the (window, head_prob, router_score) triples are stored.  `lgs4.sampling.row_units`
then draws `hard_per_row` of them (highest head probability first) in place of random negatives.

Cost: one block-local trunk pass over all windows (~1.2M windows for the 0.6B presence rows) + the deep head on 8 windows
per row (~270k) -- about 30-40 min on an RTX 6000 Ada.  Written in parts of `part_rows` rows so a killed job resumes.

Output per set: <run>/hardneg/<split>.<subset>.npz with row_ptr [n_rows+1], win_start / win_end [M], head_prob / router_score
[M] (float16), plus hardneg/meta.json (run, shas, m_router, b, pool, depths).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from lgs1.lgcore.model import pool_scores
from lgs1.lgcore.windows import make_windows, windows_containing
from lgs4 import backbone as BB
from lgs4.common import device_auto
from lgs4.config import PRESENCE_SETS, S4Paths, backbone
from lgs4.prep import PrepSet
from lgs4.router_train import row_windows, trunk_features

DEFAULT_M_ROUTER = 8
HARDNEG_DIR = "hardneg"


@torch.no_grad()
def mine_set(model, head, router, cfg, ps: PrepSet, device: str, out_path: Path, m_router: int = DEFAULT_M_ROUTER,
             micro: int = 256, part_rows: int = 2000, limit_rows: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Mine one prepped set (row order = set order).  Resumable through <name>.partK.npz files."""
    b = int(cfg.b)
    n_rows = min(len(ps), limit_rows) if limit_rows else len(ps)
    rows = list(range(n_rows))
    if out_path.exists():
        z = np.load(out_path)
        if len(z["row_ptr"]) == n_rows + 1:
            print(f"[hardneg] cached {out_path.name} ({n_rows:,} rows, {len(z['win_start']):,} windows)", flush=True)
            return {k: z[k] for k in z.files}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_parts = -(-n_rows // part_rows) if n_rows else 0
    parts: List[Dict[str, np.ndarray]] = []
    depths = cfg.router_depths

    def mine_rows(chunk: Sequence[int]) -> Dict[str, np.ndarray]:
        counts, ws, we, hp_, rs_ = [], [], [], [], []
        # pass 1: router scores for every window of the chunk (packed forwards), keep the top-m non-needle windows per row
        pend: List[Tuple[int, torch.Tensor, torch.Tensor, List[Tuple[int, int]], set]] = []
        sel_ids: List[torch.Tensor] = []
        sel_valid: List[torch.Tensor] = []
        sel_meta: List[Tuple[int, int, int, float]] = []     # (row position in chunk, start, end, router score)
        per_row_sel: Dict[int, List[Tuple[int, int, float]]] = {}

        def flush():
            if not pend:
                return
            ids = torch.cat([x[1] for x in pend], 0).to(device)
            valid = torch.cat([x[2] for x in pend], 0).to(device)
            h = trunk_features(model, ids, valid, depths)
            sc = pool_scores(router(h.float()), valid, cfg.pool).float().cpu()
            pos = 0
            for j, ids_r, valid_r, wins, needle in pend:
                s_r = sc[pos:pos + len(wins)]
                pos += len(wins)
                cand = [(float(s_r[i]), i) for i in range(len(wins)) if i not in needle]
                cand.sort(reverse=True)
                chosen = cand[:m_router]
                per_row_sel[j] = [(wins[i][0], wins[i][1], s) for s, i in chosen]
                for s, i in chosen:
                    sel_ids.append(ids_r[i:i + 1]); sel_valid.append(valid_r[i:i + 1]); sel_meta.append((j, wins[i][0], wins[i][1], s))
            pend.clear()

        n_buf = 0
        for j, r in enumerate(chunk):
            m = ps.meta[r]
            ids_r, valid_r, wins = row_windows(ps, r, cfg.pad_id, b)
            span = m.get("needle_tok_span")
            needle = set(windows_containing((span[0], span[1]), wins)) if span else set()
            if span and not needle:
                needle = {i for i, (s, e) in enumerate(wins) if s < span[1] and span[0] < e}
            pend.append((j, ids_r, valid_r, wins, needle))
            n_buf += len(wins)
            if n_buf >= micro:
                flush()
                n_buf = 0
        flush()
        # pass 2: head probability of the selected windows (packed)
        probs = np.zeros((len(sel_meta),), dtype=np.float32)
        for i in range(0, len(sel_meta), micro):
            ids = torch.cat(sel_ids[i:i + micro], 0).to(device)
            valid = torch.cat(sel_valid[i:i + micro], 0).to(device)
            probs[i:i + ids.shape[0]] = torch.sigmoid(BB.lg_window_logits(model, head, ids, valid, cfg.ell)).float().cpu().numpy()
        by_row: Dict[int, List[Tuple[int, int, float, float]]] = {}
        for (j, s, e, sc), p in zip(sel_meta, probs):
            by_row.setdefault(j, []).append((s, e, float(p), sc))
        for j in range(len(chunk)):
            lst = sorted(by_row.get(j, []), key=lambda t: (-t[2], -t[3]))
            counts.append(len(lst))
            for s, e, p, sc in lst:
                ws.append(s); we.append(e); hp_.append(p); rs_.append(sc)
        return {"counts": np.asarray(counts, dtype=np.int64), "win_start": np.asarray(ws, dtype=np.int32), "win_end": np.asarray(we, dtype=np.int32),
                "head_prob": np.asarray(hp_, dtype=np.float16), "router_score": np.asarray(rs_, dtype=np.float16)}

    for k in range(n_parts):
        pp = out_path.with_name(f"{out_path.stem}.part{k}.npz")
        chunk = rows[k * part_rows:(k + 1) * part_rows]
        if pp.exists():
            z = np.load(pp)
            if len(z["counts"]) == len(chunk):
                parts.append({kk: z[kk] for kk in z.files})
                continue
        part = mine_rows(chunk)
        np.savez(pp, **part)
        parts.append(part)
        print(f"   hardneg {ps.split}/{ps.subset}: {min(n_rows, (k + 1) * part_rows):,}/{n_rows:,} rows  {time.time() - t0:.0f}s", flush=True)
    counts = np.concatenate([p["counts"] for p in parts]) if parts else np.zeros((0,), dtype=np.int64)
    out = {"row_ptr": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)}
    for kk in ("win_start", "win_end", "head_prob", "router_score"):
        out[kk] = np.concatenate([p[kk] for p in parts]) if parts else np.zeros((0,), dtype=np.float16 if kk in ("head_prob", "router_score") else np.int32)
    np.savez(out_path, **out)
    for k in range(n_parts):
        pp = out_path.with_name(f"{out_path.stem}.part{k}.npz")
        if pp.exists():
            pp.unlink()
    hp = out["head_prob"].astype(np.float32)
    print(f"[hardneg] {ps.split}/{ps.subset}: {n_rows:,} rows, {len(hp):,} mined windows, head_prob mean {hp.mean() if hp.size else 0:.3f} "
          f"(>0.5: {(hp > 0.5).mean() if hp.size else 0:.3f}) {time.time() - t0:.0f}s", flush=True)
    return out


def mine(P: S4Paths, bb_name: str, seed: int, variant: str = "", sets: Sequence[Tuple[str, str]] = PRESENCE_SETS, device: str = "auto",
         m_router: int = DEFAULT_M_ROUTER, limit_rows: Optional[int] = None, repo_override: Optional[str] = None,
         dtype: torch.dtype = torch.bfloat16, out_dir: Optional[Path] = None) -> Path:
    device = device_auto(device)
    bb = backbone(bb_name)
    run_dir = P.run_dir(bb_name, seed, variant)
    if not (run_dir / BB.ROUTER_FILE).exists():
        raise SystemExit(f"{run_dir / BB.ROUTER_FILE} missing: mining needs the base run's head AND router")
    model, tok, head, router, cfg, info = BB.load_lg(run_dir, device=device, repo_override=repo_override, dtype=dtype)
    out_dir = Path(out_dir) if out_dir else run_dir / HARDNEG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    for split, subset in sets:
        if not (P.prep / split / subset / "ids.npy").exists():
            print(f"[hardneg] {split}/{subset}: not prepped -> skipped", flush=True)
            continue
        ps = PrepSet(P, split, subset)
        arr = mine_set(model, head, router, cfg, ps, device, out_dir / f"{split}.{subset}.npz", m_router=m_router, limit_rows=limit_rows)
        done[f"{split}.{subset}"] = {"rows": int(len(arr["row_ptr"]) - 1), "windows": int(len(arr["win_start"]))}
    meta = {"run": run_dir.name, "backbone": bb_name, "seed": seed, "variant": variant, "m_router": m_router, "b": cfg.b, "pool": cfg.pool,
            "router_ells": cfg.router_ells, "ell": cfg.ell, "head_sha": BB.sha256_file(run_dir / BB.HEAD_FILE),
            "router_sha": BB.sha256_file(run_dir / BB.ROUTER_FILE), "sets": done, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[hardneg] {out_dir}: {done}", flush=True)
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S8 hard-negative mining with the base LG run")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="", help="mine with this run variant's head + router (default: the base run)")
    ap.add_argument("--sets", default=",".join(f"{s}/{u}" for s, u in PRESENCE_SETS))
    ap.add_argument("--m-router", type=int, default=DEFAULT_M_ROUTER)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--out", default=None, help="output dir (default runs/v2/<run>/hardneg)")
    a = ap.parse_args(argv)
    P = S4Paths.load().ensure().for_backbone(a.backbone)
    sets = [tuple(x.split("/", 1)) for x in a.sets.split(",") if x]
    mine(P, a.backbone, a.seed, a.variant, sets, a.device, a.m_router, a.limit_rows, a.repo, out_dir=Path(a.out) if a.out else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
