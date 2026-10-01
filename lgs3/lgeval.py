"""Landmark Guard evaluation core (E2-1 / E3 / E7 row producer) on top of lgs1.lgcore.

The same code scores the tiny pilot backbone (E0-5) and, from S5 on, trained checkpoints:
    runner = LGRunner.from_pretrained(backbone_repo, head=..., router=..., cfg=LGConfig(k=32))
    row = runner.score_text(text, needle_char_span)   -> score, verdict, routed windows, hit@k, timings
Rows are written with the E0-2 lineage (model / router / cfg / recipe / data-manifest hashes) and summarised
with lgs1.metrics.summarize_matched (strict matched-FPR thresholds from the calibration benign rows).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from lgs1.lgcore.model import Head, LGConfig, LandmarkGuard, Router, sliding_scan
from lgs1.lgcore.windows import windows_containing
from lgs1.lineage import Lineage, RowsWriter, canonical_hash, code_commit, dir_fingerprint, done_ids, sha256_file
from lgs1.metrics import assert_unique_ids, summarize_matched
from lgs2.tok import token_span_for_chars

ROW_KEYS = ["id", "split", "subset", "L_target", "depth", "label", "method", "score", "verdict", "theta", "n_windows",
            "n_deep", "selected", "needle_windows", "hit", "needle_tok_span", "n_tok", "ms_route", "ms_head", "ms_total"]


@dataclass
class LGRunner:
    lg: LandmarkGuard
    tok: object                      # lgs2.tok HFTok-like (encode_with_offsets)
    method: str
    device: str = "cpu"
    sliding: bool = False            # B5: every window through the head

    @classmethod
    def from_modules(cls, model, tok, cfg: LGConfig, router: Optional[Router] = None, head: Optional[Head] = None,
                     method: str = "lg", device: str = "cpu", sliding: bool = False) -> "LGRunner":
        lg = LandmarkGuard(model, cfg, router, head).to(device)
        return cls(lg=lg, tok=tok, method=method, device=device, sliding=sliding)

    @torch.no_grad()
    def score_ids(self, ids: torch.Tensor, needle_tok_span: Optional[Tuple[int, int]] = None) -> Dict:
        ids = ids.to(self.device)
        t0 = time.perf_counter()
        if self.sliding:
            out = sliding_scan(self.lg, ids)
            t1 = t2 = time.perf_counter()
        else:
            r = self.lg.route(ids)
            t1 = time.perf_counter()
            out = self.lg(ids)          # (route is recomputed inside; the split timing is what E4-4 needs)
            t2 = time.perf_counter()
        sel = [int(i) for i in out.selected.tolist()]
        nwins = windows_containing(needle_tok_span, out.windows) if needle_tok_span else []
        return {"score": out.score(), "verdict": bool(out.verdict_unsafe), "theta": float(out.theta),
                "n_windows": len(out.windows), "n_deep": int(out.n_deep), "selected": sel[:64],
                "needle_windows": nwins, "hit": (any(w in sel for w in nwins) if nwins else None),
                "needle_tok_span": list(needle_tok_span) if needle_tok_span else None, "n_tok": int(ids.numel()),
                "ms_route": round((t1 - t0) * 1000, 2), "ms_head": round((t2 - t1) * 1000, 2),
                "ms_total": round((t2 - t0) * 1000, 2)}

    def score_text(self, text: str, needle_char_span: Optional[Sequence[int]] = None) -> Dict:
        ids, offs = self.tok.encode_with_offsets(text)
        span = token_span_for_chars(offs, (needle_char_span[0], needle_char_span[1])) if needle_char_span else None
        return self.score_ids(torch.tensor(ids, dtype=torch.long), span)


def lineage_for(model_dir: Optional[str], router_path: Optional[str], cfg: LGConfig, method: str,
                recipe_hash: str = "none", manifest_path: Optional[str] = None) -> Lineage:
    cfg_d = dict(cfg.to_dict(), method=method, harness="s3-lgeval-v1")
    return Lineage(model_sha=dir_fingerprint(model_dir) if model_dir else "none",
                   router_sha=sha256_file(router_path) if router_path and Path(router_path).exists() else "none",
                   cfg_hash=canonical_hash(cfg_d), recipe_hash=recipe_hash,
                   data_manifest_sha=sha256_file(manifest_path) if manifest_path and Path(manifest_path).exists() else "none",
                   code_commit=code_commit(Path(__file__).resolve().parents[1]))


def run_rows(runner: LGRunner, rows: Iterable[Dict], out_path: Path, lin: Lineage, resume: bool = True,
             log_every: int = 50) -> int:
    """Score data rows (lgs2 synth schema) -> eval rows with lineage.  Resumable by id."""
    done = done_ids(out_path) if resume and out_path.exists() else set()
    n = 0
    t0 = time.time()
    with RowsWriter(out_path, lin, resume=out_path.exists()) as w:
        for r in rows:
            if r["id"] in done:
                continue
            s = runner.score_text(r["text"], r.get("needle_char_span"))
            w.write({"id": r["id"], "split": r["split"], "subset": r["subset"], "L_target": r.get("L_target"), "depth": r.get("depth"),
                     "label": 1 if r["label"] == "harmful" else 0, "method": runner.method, "haystack_domain": r.get("haystack_domain"),
                     "needle_id": r.get("needle_id"), "pair_id": r.get("pair_id"), **s})
            n += 1
            if n % log_every == 0:
                print(f"   {n} rows  {time.time() - t0:.0f}s", flush=True)
    return n


def read_eval_rows(path: Path) -> List[Dict]:
    from lgs1.lineage import iter_rows
    return list(iter_rows(path)) if path.exists() else []


def summarize(test_rows: List[Dict], calib_rows: List[Dict]) -> Dict:
    """matched-FPR recall per (method, L_target) + hit@k per L + default-threshold verdict rates."""
    assert_unique_ids(test_rows, keys=("split", "subset", "id"))
    for r in test_rows + calib_rows:
        r.setdefault("method", "lg")
    table = summarize_matched(test_rows, calib_rows, group_keys=("method", "L_target"))
    hits: Dict = {}
    for r in test_rows:
        if r.get("hit") is not None:
            k = (r["method"], r["L_target"])
            hits.setdefault(k, [0, 0])
            hits[k][0] += int(bool(r["hit"]))
            hits[k][1] += 1
    for t in table:
        h = hits.get((t["method"], t["L_target"]))
        t["hit_at_k"] = (h[0] / h[1]) if h else None
        sub = [r for r in test_rows if r["method"] == t["method"] and r["L_target"] == t["L_target"]]
        t["recall_default"] = (sum(1 for r in sub if r["label"] == 1 and r["verdict"]) / max(1, sum(1 for r in sub if r["label"] == 1)))
        t["fpr_default"] = (sum(1 for r in sub if r["label"] == 0 and r["verdict"]) / max(1, sum(1 for r in sub if r["label"] == 0)))
        t["ms_total_mean"] = sum(r["ms_total"] for r in sub) / max(1, len(sub))
    return {"table": table, "n_test": len(test_rows), "n_calib": len(calib_rows)}
