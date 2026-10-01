"""S7 cost tables from the lgs7.cost rows: Fig 7 (a) latency vs L, (b) throughput vs batch, (c) peak memory vs L, Fig 9
breakdown + k sweep, Table 5 cost columns (ms@28k / 128k, tokens/s, peak memory growth, token budget x), G2 (b)/(c).

    python -m lgs7.tables            # reads $LG_OUT/cost/<gpu>/*.rows.jsonl -> reports/s7/{fig7_latency,fig7_throughput,fig7_memory,fig9_breakdown,fig9_ksweep}.json, table5_cost.md, g2_cost.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from lgs5.config import S5Paths
from lgs5.tables import _f


def load_cost(P: S5Paths) -> Dict[str, List[Dict]]:
    """gpu slug -> rows (the last row per key wins)."""
    out: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for f in sorted(glob.glob(str(P.out / "cost" / "*" / "*.rows.jsonl"))):
        gpu = Path(f).parent.name
        for line in Path(f).read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            r["_file"] = Path(f).name
            out[gpu][r.get("key", line[:64])] = r
    return {g: list(d.values()) for g, d in out.items()}


def _mkey(r: Dict) -> str:
    return f"{r['method']}|{r['model']}"


def token_budget_x(lin: Dict, L: int) -> Optional[float]:
    """E4-7: LG deep-equivalent token budget = L*ell/N + k*2b*(1 - ell/N), as a multiple of L (CD = 1 + overlap/W)."""
    try:
        ell, N, k, b = int(lin["ell"]), int(lin["n_layers"]), int(lin["k"]), int(lin["b"])
    except (KeyError, TypeError, ValueError):
        return None
    W = max(1, math.ceil(L / b) - 1) if L > 2 * b else 1
    kk = min(k, W)
    return (L * ell / N + kk * 2 * b * (1 - ell / N)) / L


def fig7_latency(rows: List[Dict]) -> Dict:
    out: Dict = {}
    for r in rows:
        if r.get("exp") != "latency":
            continue
        e = out.setdefault(_mkey(r), {})
        e[str(r["L_target"])] = {"p50_ms": r.get("p50_ms"), "p99_ms": r.get("p99_ms"), "mean_ms": r.get("mean_ms"), "ms_total_p50": r.get("ms_total_p50"),
                                 "ms_tokenize": r.get("ms_tokenize_mean"), "n_tok": r.get("n_tok_mean"), "peak_alloc_mb": r.get("peak_alloc_mb"),
                                 "error": r.get("error"), "spec": r.get("spec")}
    return out


def fig7_throughput(rows: List[Dict]) -> Dict:
    out: Dict = {}
    for r in rows:
        if r.get("exp") != "throughput":
            continue
        out.setdefault(_mkey(r), {}).setdefault(str(r["L_target"]), {})[str(r["batch"])] = {
            "docs_per_s": r.get("docs_per_s"), "tokens_per_s": r.get("tokens_per_s"), "mean_ms": r.get("mean_ms"),
            "peak_alloc_mb": r.get("peak_alloc_mb"), "error": r.get("error")}
    return out


def fig7_memory(rows: List[Dict]) -> Dict:
    out: Dict = {}
    for r in rows:
        if r.get("exp") != "memory":
            continue
        out.setdefault(_mkey(r), {})[str(r["L_target"])] = {"peak_alloc_mb": r.get("peak_alloc_mb"), "peak_reserved_mb": r.get("peak_reserved_mb"),
                                                             "baseline_alloc_mb": r.get("baseline_alloc_mb"), "peak_delta_mb": r.get("peak_delta_mb"),
                                                             "error": r.get("error")}
    return out


def fig9(rows: List[Dict]) -> Dict:
    br: Dict = {}
    ks: Dict = {}
    for r in rows:
        if r.get("exp") == "breakdown":
            br.setdefault(_mkey(r), {})[str(r["L_target"])] = {"phases_ms": r.get("phases_ms"), "phases_pct": r.get("phases_pct"), "total_ms": r.get("total_ms")}
        elif r.get("exp") == "ksweep":
            ks.setdefault(_mkey(r), {}).setdefault(str(r["L_target"]), {})[str(r["k"])] = {"p50_ms": r.get("p50_ms"), "p99_ms": r.get("p99_ms"),
                                                                                            "peak_alloc_mb": r.get("peak_alloc_mb")}
    return {"breakdown": br, "ksweep": ks}


def table5(rows: List[Dict], Ls_report: Sequence[int] = (1024, 28672, 131072)) -> List[Dict]:
    lat, mem, thr = fig7_latency(rows), fig7_memory(rows), fig7_throughput(rows)
    lin = {}
    for r in rows:
        lin.setdefault(_mkey(r), r.get("lineage") or {})
    out = []
    for key in sorted(set(lat) | set(mem) | set(thr)):
        e = {"method": key.split("|")[0], "model": key.split("|")[1], "gpu": None}
        for L in Ls_report:
            l = lat.get(key, {}).get(str(L), {})
            e[f"p50_ms@{L}"] = l.get("p50_ms")
            e[f"p99_ms@{L}"] = l.get("p99_ms")
            m = mem.get(key, {}).get(str(L), {})
            e[f"peak_mb@{L}"] = m.get("peak_alloc_mb") if m.get("peak_alloc_mb") is not None else l.get("peak_alloc_mb")
        t32 = thr.get(key, {}).get("32768", {})
        best = max((v for v in t32.values() if v.get("tokens_per_s")), key=lambda v: v["tokens_per_s"], default=None)
        e["tokens_per_s@32k(best batch)"] = best["tokens_per_s"] if best else None
        e["docs_per_s@32k(best batch)"] = best["docs_per_s"] if best else None
        p1, p128 = e.get(f"peak_mb@{Ls_report[0]}"), e.get(f"peak_mb@{Ls_report[-1]}")
        e["mem_growth_1k_to_128k"] = ((p128 - p1) / p1) if (p1 and p128) else None
        e["budget_x@128k"] = token_budget_x(lin.get(key, {}), 131072) if str(e["method"]).startswith("lg_k") else None
        out.append(e)
    return out


def g2_cost(t5: List[Dict]) -> Dict:
    """G2 (b) 128k latency: LG <= sliding / 4 ; (c) LG peak memory growth 1k -> 128k <= +10 %."""
    import re
    # the main config and its streaming-chunk variants (lg_k32+c8 ...; G2 (c) fallback = chunk fixed at <= k windows)
    lg = [e for e in t5 if re.fullmatch(r"lg_k32(_s\d+)?(_tp\d+)?(\+c\d+)?(\+h\d+)?", str(e["method"]))]
    # sliding baselines by (model, streaming chunk): an LG variant is compared with the sliding scan of the SAME chunk
    # when it was measured (9/27: c64 sliding is itself ~20 % slower than c16, which flattered lg_k32+c16), else the default
    sl: Dict = {}
    for x in t5:
        mm = re.fullmatch(r"sliding(\+c(\d+))?", str(x["method"]))
        if mm:
            sl[(x["model"], int(mm.group(2)) if mm.group(2) else None)] = x
    out: Dict = {}
    for e in lg:
        mc = re.search(r"\+c(\d+)", str(e["method"]))
        s = sl.get((e["model"], int(mc.group(1)) if mc else None)) or sl.get((e["model"], None))
        b = {"lg_p50_ms@128k": e.get("p50_ms@131072"), "sliding_p50_ms@128k": s.get("p50_ms@131072") if s else None,
             "sliding_method": s["method"] if s else None}
        if b["lg_p50_ms@128k"] and b["sliding_p50_ms@128k"]:
            b["ratio"] = b["lg_p50_ms@128k"] / b["sliding_p50_ms@128k"]
            b["pass_b"] = b["ratio"] <= 0.25
        c = {"peak_mb@1k": e.get("peak_mb@1024"), "peak_mb@128k": e.get("peak_mb@131072"), "growth": e.get("mem_growth_1k_to_128k")}
        if c["growth"] is not None:
            c["pass_c"] = c["growth"] <= 0.10
        out[f"{e['method']}|{e['model']}"] = {"b_latency": b, "c_memory": c}
    return out


def table5_md(per_gpu: Dict[str, List[Dict]]) -> str:
    lines = [f"# Table 5 — cost (E4-1/2/3, E4-7)  ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "batch 1 p50 / p99 ms per document at 1k / 28k / 128k (model time; tokenization in the JSON), peak allocated MB at 1k / 128k and "
             "its growth, best-batch tokens/s at 32k, LG deep-equivalent token budget x = ell/N + k*2b*(1-ell/N)/L (E4-7).  One block per GPU.", ""]
    for gpu, rows in per_gpu.items():
        lines += [f"## {gpu}", "",
                  "| method | model | p50@1k | p50@28k | p50@128k | p99@128k | peak MB@1k | peak MB@128k | growth | tok/s@32k | budget x@128k |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for e in rows:
            lines.append(f"| {e['method']} | {e['model']} | {_f(e.get('p50_ms@1024'), 1)} | {_f(e.get('p50_ms@28672'), 1)} | {_f(e.get('p50_ms@131072'), 1)} | "
                         f"{_f(e.get('p99_ms@131072'), 1)} | {_f(e.get('peak_mb@1024'), 0)} | {_f(e.get('peak_mb@131072'), 0)} | "
                         f"{_f(e.get('mem_growth_1k_to_128k'))} | {_f(e.get('tokens_per_s@32k(best batch)'), 0)} | {_f(e.get('budget_x@128k'))} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def g2_md(per_gpu: Dict[str, Dict]) -> str:
    lines = [f"# G2 (b)(c) — 128k latency LG <= sliding / 4 ; peak memory 1k -> 128k <= +10 %  ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "| gpu | method | model | LG p50@128k | sliding p50@128k (same chunk when measured) | ratio | (b) | peak@1k | peak@128k | growth | (c) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for gpu, d in per_gpu.items():
        for key, e in d.items():
            m, mod = key.split("|", 1)
            b, c = e["b_latency"], e["c_memory"]
            sm = f" ({b['sliding_method']})" if b.get("sliding_method") and b["sliding_method"] != "sliding" else ""
            lines.append(f"| {gpu} | {m} | {mod} | {_f(b.get('lg_p50_ms@128k'), 1)} | {_f(b.get('sliding_p50_ms@128k'), 1)}{sm} | {_f(b.get('ratio'))} | "
                         f"{'PASS' if b.get('pass_b') else ('FAIL' if b.get('pass_b') is False else '')} | {_f(c.get('peak_mb@1k'), 0)} | "
                         f"{_f(c.get('peak_mb@128k'), 0)} | {_f(c.get('growth'))} | {'PASS' if c.get('pass_c') else ('FAIL' if c.get('pass_c') is False else '')} |")
    return "\n".join(lines) + "\n"


def build(P: S5Paths) -> Dict[str, str]:
    per = load_cost(P)
    if not per:
        return {}
    rep = P.out / "reports" / "s7"
    rep.mkdir(parents=True, exist_ok=True)
    written = {}
    figs = {"fig7_latency": {g: fig7_latency(r) for g, r in per.items()}, "fig7_throughput": {g: fig7_throughput(r) for g, r in per.items()},
            "fig7_memory": {g: fig7_memory(r) for g, r in per.items()}, "fig9": {g: fig9(r) for g, r in per.items()}}
    for name, d in figs.items():
        (rep / f"{name}.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
        written[name] = str(rep / f"{name}.json")
    t5 = {g: table5(r) for g, r in per.items()}
    (rep / "table5_cost.json").write_text(json.dumps(t5, indent=2), encoding="utf-8")
    (rep / "table5_cost.md").write_text(table5_md(t5), encoding="utf-8")
    written["table5"] = str(rep / "table5_cost.md")
    g2 = {g: g2_cost(t) for g, t in t5.items()}
    (rep / "g2_cost.json").write_text(json.dumps(g2, indent=2), encoding="utf-8")
    (rep / "g2_cost.md").write_text(g2_md(g2), encoding="utf-8")
    written["g2"] = str(rep / "g2_cost.md")
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S7 cost tables")
    ap.parse_args(argv)
    P = S5Paths.load().ensure()
    w = build(P)
    if not w:
        print("[s7 tables] no rows under cost/ yet")
        return 1
    for k, v in w.items():
        print(f"[s7 tables] {k}: {v}")
        if v.endswith(".md"):
            print(Path(v).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
