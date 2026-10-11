"""Score the large negative pool (bigneg.jsonl.gz) with the frozen probes -- the low-FPR regime of plan E0.3/E1.1
(window FPR 2e-4 needs >= 25k negatives for 5 expected false positives, 5e-5 needs 100k).

Features are never stored: each batch is pooled, standardized with the train statistics and multiplied by every
probe (tasks x seeds x layers) on the device.  Output <run>/probes/<bb>/<framing>/<pool>.bigneg_scores.f32, a float32
memmap [n_big, T, S, L+1], with bigneg_ids.json and a resumable progress file.

    python -m agdepth.score --run NAME --backbone qwen3guard_0.6b [--framings raw] [--poolings mean,last]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Dict, Optional, Sequence

import numpy as np

from .config import DEFAULT_POOLINGS, AGPaths, backbone

SHARD = 4096


def score(run: str, bb_key: str, framings: Sequence[str] = ("raw",), poolings: Sequence[str] = DEFAULT_POOLINGS,
          device: Optional[str] = None, dtype: Optional[str] = None, budget: Optional[int] = None,
          P: Optional[AGPaths] = None, M: Optional[Dict] = None) -> None:
    import torch
    from .extract import iter_pooled, load
    from .windows import read_jsonl_gz
    P = P or AGPaths.load()
    R = P.run(run)
    if not R.bigneg.exists():
        print(f"[score] {R.bigneg} not built (profile without bigneg): nothing to do")
        return
    card = json.loads(R.datacard.read_text())
    max_len = int(card["recipe"]["profile"]["max_len"])
    rows = read_jsonl_gz(R.bigneg)
    bb = backbone(bb_key)
    M = M or load(bb.repo, device, dtype)
    budget = budget or bb.tokens_per_batch
    dev = M["device"]
    N = len(rows)
    for framing in framings:
        D = R.probes(bb.key, framing)
        probes = {}
        for p in poolings:
            f = D / f"{p}.probes.npz"
            if not f.exists():
                raise SystemExit(f"{f} missing: run fit first")
            z = np.load(f)
            probes[p] = {k: torch.from_numpy(z[k].astype(np.float32)).to(dev) for k in ("W", "b", "mu", "sd")}
        T, S, L1, d = probes[poolings[0]]["W"].shape
        mms, done, psha = {}, {}, {}
        for p in poolings:
            out_p, prog_p = D / f"{p}.bigneg_scores.f32", D / f"{p}.bigneg_progress.json"
            psha[p] = hashlib.sha1((D / f"{p}.probes.npz").read_bytes()).hexdigest()[:16]
            prog = json.loads(prog_p.read_text()) if prog_p.exists() else {}
            ok = prog.get("n") == N and prog.get("probes_sha") == psha[p] and out_p.exists()   # re-fit probes -> rescore
            mms[p] = np.memmap(out_p, dtype=np.float32, mode="r+" if ok else "w+", shape=(N, T, S, L1))
            done[p] = set(prog.get("done", [])) if ok else set()
        (D / "bigneg_ids.json").write_text(json.dumps([r["id"] for r in rows]))
        shards = [list(range(s, min(N, s + SHARD))) for s in range(0, N, SHARD)]
        todo = [si for si in range(len(shards)) if any(si not in done[p] for p in poolings)]
        t0 = time.time()
        for k, si in enumerate(todo):
            shard = shards[si]
            sub = [rows[i] for i in shard]
            for loc, feats in iter_pooled(M, sub, framing, list(poolings), max_len, budget):
                gidx = np.asarray([shard[j] for j in loc])
                for p, x in feats.items():
                    pr = probes[p]
                    xt = (torch.from_numpy(x).to(dev) - pr["mu"]) / pr["sd"]            # [B, L1, d]
                    sc = torch.einsum("bld,tsld->btsl", xt, pr["W"]) + pr["b"][None]
                    mms[p][gidx] = sc.cpu().numpy()
            for p in poolings:
                mms[p].flush()
                done[p].add(si)
                (D / f"{p}.bigneg_progress.json").write_text(json.dumps({"n": N, "probes_sha": psha[p], "done": sorted(done[p]),
                                                                         "complete": len(done[p]) == len(shards)}))
            rate = sum(len(shards[j]) for j in todo[:k + 1]) / max(1e-6, time.time() - t0)
            print(f"[score] {bb.key}/{framing} shard {si + 1}/{len(shards)}  {rate:.1f} windows/s", flush=True)
        print(f"[score] {bb.key}/{framing}: {N:,} bigneg windows -> {D}")


def load_bigneg(R, bb: str, framing: str, pool: str):
    D = R.probes(bb, framing)
    prog_p = D / f"{pool}.bigneg_progress.json"
    if not prog_p.exists() or not json.loads(prog_p.read_text()).get("complete"):
        return None, None
    ids = json.loads((D / "bigneg_ids.json").read_text())
    z = np.load(D / f"{pool}.probes.npz")
    T, S, L1, _ = z["W"].shape
    return np.memmap(D / f"{pool}.bigneg_scores.f32", dtype=np.float32, mode="r", shape=(len(ids), T, S, L1)), ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="score the big negative pool with the frozen probes")
    ap.add_argument("--run", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--framings", default="raw")
    ap.add_argument("--poolings", default=",".join(DEFAULT_POOLINGS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--tokens-per-batch", type=int, default=None)
    a = ap.parse_args(argv)
    score(a.run, a.backbone, [f for f in a.framings.split(",") if f], [p for p in a.poolings.split(",") if p],
          a.device, a.dtype, a.tokens_per_batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
