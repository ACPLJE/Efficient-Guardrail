"""E0-7 — analytic FLOPs of Landmark Guard vs torch's FlopCounterMode (plan S4 item 2; report Table 5 input).

    python -m lgs4.flops --backbone qwen3guard_0.6b [--run runs/v2/qwen3guard_0.6b_s0] --L 4096,28672,131072 --k 32

Analytic (per document of L tokens, windows of 2b tokens, W = |make_windows(L)| ~ L/b):
    trunk  = W  * 2b * ell     * c_layer       c_layer = 2 * P_layer + 4 * 2b * d_q     (linear + attention over the window)
    deep   = k  * 2b * (N-ell) * c_layer
    total  = trunk + deep (+ router/probe, negligible)   -> constant-work claim: `deep` does not depend on L
Measured: FlopCounterMode around LandmarkGuard.forward on random tokens (batch 1, skipping on = only k windows go deep).
Pass: |measured - analytic| / analytic <= 10 % for every L.  Output $LG_OUT/reports/e07_flops.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from lgs1.lgcore.model import Head, LGConfig, LandmarkGuard, Router, _inner
from lgs1.lgcore.windows import make_windows
from lgs4 import backbone as BB
from lgs4.common import device_auto
from lgs4.config import LG_K_DEFAULT, WINDOW_B, S4Paths, backbone


def layer_params(model) -> int:
    dec = _inner(model)
    return sum(p.numel() for p in dec.layers[0].parameters())


def q_dim(model) -> int:
    cfg = _inner(model).config
    heads = int(getattr(cfg, "num_attention_heads"))
    hd = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // heads)
    return heads * hd


def analytic(model, L: int, ell: int, k: int, b: int = WINDOW_B) -> Dict:
    T = 2 * b
    W = len(make_windows(L, b))
    c_layer_tok = 2 * layer_params(model) + 4 * T * q_dim(model)     # FLOPs per token per layer (window of T tokens)
    n = len(_inner(model).layers)
    kk = min(k, W)
    trunk = W * T * ell * c_layer_tok
    deep = kk * T * (n - ell) * c_layer_tok
    return {"L": L, "windows": W, "k_eff": kk, "flops_trunk": trunk, "flops_deep": deep, "flops_total": trunk + deep,
            "c_layer_per_token": c_layer_tok}


@torch.no_grad()
def measure(lg: LandmarkGuard, ids: torch.Tensor) -> Dict:
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    t0 = time.perf_counter()
    with fc:
        lg.route(ids)
    t1 = time.perf_counter()
    route_flops = fc.get_total_flops()
    fc2 = FlopCounterMode(display=False)
    with fc2:
        lg(ids)
    t2 = time.perf_counter()
    total = fc2.get_total_flops()
    return {"measured_route": route_flops, "measured_total": total, "measured_deep": total - route_flops,
            "ms_route": round((t1 - t0) * 1000, 1), "ms_total": round((t2 - t1) * 1000, 1)}


def run(P: S4Paths, bb_name: str, Ls: List[int], k: int = LG_K_DEFAULT, run_dir: Optional[Path] = None, device: str = "auto",
        repo_override: Optional[str] = None, ell_override: Optional[int] = None, tol: float = 0.10, seed: int = 0,
        dtype: torch.dtype = torch.bfloat16) -> Dict:
    device = device_auto(device)
    bb = backbone(bb_name)
    # eager attention: QK^T / AV are plain matmuls, so FlopCounterMode sees them on every backend (fused SDPA kernels
    # are not counted on some CPU/CUDA builds -> a spurious 0.5x).  The maths (and hence the FLOPs) are identical.
    if run_dir and (Path(run_dir) / BB.HEAD_FILE).exists():
        model, tok, head, router, cfg, _ = BB.load_lg(Path(run_dir), device=device, k=k, repo_override=repo_override, dtype=dtype, attn="eager")
        cfg = LGConfig(**{**cfg.to_dict(), "k": k})
        source = str(run_dir)
    else:
        model, tok = BB.load_backbone(repo_override or bb, device=device, dtype=dtype, attn="eager")
        d = BB.hidden_size(model)
        head, router = Head(d).to(device), Router(d).to(device)
        cfg = LGConfig(b=WINDOW_B, ell=ell_override or bb.ell, k=k, pad_id=BB.pad_id_of(model, tok))
        source = "base backbone (untrained probe/router: FLOPs do not depend on weights)"
    if ell_override:
        cfg = LGConfig(**{**cfg.to_dict(), "ell": ell_override})
    lg = LandmarkGuard(model, cfg, router, head).to(device).eval()
    g = torch.Generator().manual_seed(seed)
    vocab = int(_inner(model).config.vocab_size)
    rows = []
    for L in Ls:
        ids = torch.randint(3, min(vocab, 30000), (L,), generator=g).to(device)
        m = measure(lg, ids)
        a = analytic(model, L, cfg.ell, k)
        ratio = m["measured_total"] / max(1, a["flops_total"])
        rows.append({**a, **m, "ratio_total": ratio, "ratio_deep": m["measured_deep"] / max(1, a["flops_deep"]),
                     "ratio_trunk": m["measured_route"] / max(1, a["flops_trunk"]), "pass": abs(ratio - 1) <= tol})
        print(f"[flops] L={L:>7} W={a['windows']:>5} analytic={a['flops_total']/1e12:.3f} TF measured={m['measured_total']/1e12:.3f} TF "
              f"ratio={ratio:.3f} deep={m['measured_deep']/1e12:.3f} TF ({m['ms_total']} ms) {'PASS' if rows[-1]['pass'] else 'FAIL'}", flush=True)
    rep = {"backbone": bb_name, "source": source, "cfg": cfg.to_dict(), "k": k, "tol": tol, "rows": rows, "pass": all(r["pass"] for r in rows),
           "attn_implementation": "eager (matmul-decomposed attention so the counter sees QK^T and AV)",
           "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch": torch.__version__,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    P.reports.mkdir(parents=True, exist_ok=True)
    (P.reports / "e07_flops.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = ["# E0-7 FLOPs: analytic vs measured", "", f"backbone {bb_name}, ell={cfg.ell}, b={cfg.b}, k={k}, tolerance ±{tol:.0%}", "",
          "| L | windows | analytic TF | measured TF | ratio | deep analytic TF | deep measured TF | ms | pass |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(f"| {r['L']} | {r['windows']} | {r['flops_total']/1e12:.3f} | {r['measured_total']/1e12:.3f} | {r['ratio_total']:.3f} | "
                  f"{r['flops_deep']/1e12:.3f} | {r['measured_deep']/1e12:.3f} | {r['ms_total']} | {'PASS' if r['pass'] else 'FAIL'} |")
    md += ["", f"overall: {'PASS' if rep['pass'] else 'FAIL'} (the deep stage is constant in L: k x 2b x (N-ell) layers)"]
    (P.reports / "e07_flops.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[flops] {'PASS' if rep['pass'] else 'FAIL'} -> {P.reports / 'e07_flops.md'}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E0-7 analytic vs measured FLOPs")
    ap.add_argument("--backbone", default="qwen3guard_0.6b")
    ap.add_argument("--run", default=None, help="run dir with head.pt/router.pt (default: base backbone)")
    ap.add_argument("--L", default="4096,28672,131072")
    ap.add_argument("--k", type=int, default=LG_K_DEFAULT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--ell", type=int, default=None)
    a = ap.parse_args(argv)
    P = S4Paths.load().ensure()
    rep = run(P, a.backbone, [int(x) for x in a.L.split(",") if x], a.k, Path(a.run) if a.run else None, a.device, a.repo, a.ell)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
