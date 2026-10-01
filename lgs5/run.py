"""S5 row producer CLI.

    python -m lgs5.run --exp e1 --model lg:qwen3guard_0.6b_s0                 # 7 parity benches + XSTest/OR-Bench (native)
    python -m lgs5.run --exp e2 --model lg:qwen3guard_0.6b_s0 --L le28k       # test main+benign_long <=28k + calib thresholds
    python -m lgs5.run --exp e2 --model oracle:qwen3guard_0.6b_s0             # E2-7 (ii): test/main + calib/main (needle windows)
    python -m lgs5.run --exp e2 --model needle:qwen3guard_0.6b_s0             # E2-7 (i): needles alone
    python -m lgs5.run --exp e2 --model dense:qwen3guard_0.6b_s0 --L le28k    # B4
    python -m lgs5.run --exp e1 --model base:qwen3guard_gen_0.6b              # zero-shot base (strict/loose in the tables)
    python -m lgs5.run --exp e2 --model lg:qwen3guard_0.6b_s0 --L long        # S6 E2-1: the 64k/128k rows into the SAME rows file
    python -m lgs5.run --exp e2 --model dense:qwen3guard_0.6b_s0:yarn --L long  # S6 B4: dense-FT with YaRN x4 (own rows file)
    python -m lgs5.run --exp var --model lg:qwen3guard_0.6b_s0 --subsets needle_repeat,decoy,framed,ood_domain   # S6 variants

Rows -> $LG_OUT/eval/v2/<exp>/<method>__<run>.rows.jsonl (resumable by id, E0-2 lineage on every row).
"""
from __future__ import annotations

import argparse
import sys
from typing import List

from lgs1.lineage import sha256_file
from lgs2.config import depth_warning
from lgs4.common import device_auto

from .config import VAR_L, VAR_S6, VAR_SUBSETS, S5Paths, parse_Ls, parse_spec
from .data import calib_rows, e1_rows, e27_needle_rows, e2_rows, variant_rows
from .runners import make_runner, write_rows


def rows_for(P: S5Paths, exp: str, spec, Ls: List[int], depths=None, max_pairs=None, max_rows=None, subsets=None, calib_main: bool = True):
    """(rows, manifest sha) for one experiment / model kind.  calib_main=False (S8 ablation subset) leaves out the calib/main
    twins, which only the E2-7 decomposition needs."""
    D = P.data
    if exp == "e1":
        rows = e1_rows(D)
        man = D.needles / "pool.jsonl.gz"
    elif exp == "var":
        # S6 variant subsets (test only, L in VAR_L): needle_repeat (E2-4), decoy (E3-2), framed (E3-3), ood_domain (E7-5)
        # by default; any lgs3.variants subset for the S8-S9 robustness rows.  Thresholds come from the e2 rows' calib
        # benign-long at the same L (lgs6.tables), so no calib rows are added here.
        subs = list(subsets) if subsets else list(VAR_S6)
        bad = [x for x in subs if x not in VAR_SUBSETS]
        if bad:
            raise SystemExit(f"unknown variant subset(s) {bad}; one of {VAR_SUBSETS}")
        if spec.kind in ("oracle", "needle"):
            raise SystemExit("var rows are for lg | sliding | dense models")
        rows = variant_rows(D, subs, Ls)
        mans = [D.manifests / f"test.{x}.manifest.json" for x in subs]
        man = mans[0] if len(mans) == 1 else None
        if len(mans) > 1:
            import hashlib
            h = hashlib.sha256()
            for m in mans:
                h.update((sha256_file(m) if m.exists() else "none").encode())
            man_sha = h.hexdigest()
            if max_rows:
                rows = rows[:max_rows]
            return rows, man_sha
    elif exp == "e2":
        if spec.kind == "oracle":
            rows = list(e2_rows(D, Ls, split="test", subsets=("main",), depths=depths, max_pairs=max_pairs)) + \
                   list(e2_rows(D, Ls, split="calib", subsets=("main",)))
        elif spec.kind == "needle":
            rows = e27_needle_rows(D, Ls) + [dict(r, split="calib") for r in e27_needle_rows_calib(D, Ls)]
        else:
            rows = list(e2_rows(D, Ls, split="test", depths=depths, max_pairs=max_pairs)) + list(calib_rows(D, Ls))
            if spec.kind in ("lg", "sliding") and calib_main:
                # E2-7 needs the three layers thresholded on the SAME negatives: the calib/main benign twins (the needle-only
                # and oracle layers cannot score benign-long rows).  Table 2 keeps calib/benign_long (E0-3).  Appended last so
                # a rows file written before 2026-09-20 resumes with exactly these ids as todo.
                rows += list(e2_rows(D, Ls, split="calib", subsets=("main",)))
        man = D.manifests / "test.main.manifest.json"
        w = depth_warning([r for r in rows if r.get("split") == "test"], depths, f"e2 {spec}")
        if w:
            print(w, file=sys.stderr, flush=True)
            print(w, flush=True)
    else:
        raise SystemExit(f"unknown exp {exp} (e1 | e2 | var)")
    if max_rows:
        rows = rows[:max_rows]
    return rows, (sha256_file(man) if man is not None and man.exists() else "none")


def e27_needle_rows_calib(D, Ls):
    """Needle-only rows of the calib/main pairs (thresholds for the needle-only layer)."""
    from .data import e27_needle_rows as _nr
    import lgs5.data as DATA
    seen = {}
    for r in DATA.e2_rows(D, Ls, split="calib", subsets=("main",)):
        nid = r.get("needle_id")
        if nid and nid not in seen:
            seen[nid] = r
    texts = DATA.needle_texts(D, seen)
    out = []
    for nid, r in seen.items():
        nd = texts.get(nid)
        text = nd["text"] if nd else r["text"][r["needle_char_span"][0]:r["needle_char_span"][1]]
        out.append({"id": f"needle-calib-{nid}", "split": "calib", "subset": "needle_only", "L_target": None, "depth": None,
                    "label": r["label"], "text": text, "needle_char_span": [0, len(text)], "needle_id": nid, "pair_id": r.get("pair_id"),
                    "haystack_domain": None, "meta": {}})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S5 evaluation rows")
    ap.add_argument("--exp", required=True, choices=["e1", "e2", "var"])
    ap.add_argument("--model", required=True, help="lg:<run>[:kK] | sliding:<run> | oracle:<run> | needle:<run> | dense:<run>[:head|tail|yarn] | base:<guard>")
    ap.add_argument("--L", default="le28k", help="le28k | long (64k,128k) | all | var (4k,28k,128k) | comma list")
    ap.add_argument("--subsets", default=None, help="var: comma list of lgs3.variants subsets (default needle_repeat,decoy,framed,ood_domain)")
    ap.add_argument("--depths", default=None, help="comma list of depths (subset runs)")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-tokens", type=int, default=None, help="dense: context cap (default min(model max, 32768); yarn: 131072)")
    ap.add_argument("--prefill-chunk", type=int, default=None, help="dense: KV-cache prefill chunk for rows > --prefill-min tokens (default 2048)")
    ap.add_argument("--prefill-min", type=int, default=None, help="dense: rows longer than this prefill in chunks (default 32768)")
    ap.add_argument("--stream", default="auto", choices=["auto", "on", "off"], help="LG: window streaming (auto = rows > 32k tokens)")
    ap.add_argument("--chunk-windows", type=int, default=64)
    ap.add_argument("--no-calib-main", action="store_true", help="e2: skip the calib/main twins (S8 ablation subset; E2-7 not needed)")
    a = ap.parse_args(argv)
    P = S5Paths.load().ensure()
    spec = parse_spec(a.model)
    Ls = parse_Ls(a.L) if not (a.exp == "var" and a.L == "le28k") else list(VAR_L)
    depths = [float(x) for x in a.depths.split(",")] if a.depths else None
    subsets = [x for x in a.subsets.split(",") if x] if a.subsets else None
    rows, man = rows_for(P, a.exp, spec, Ls, depths, a.max_pairs, a.max_rows, subsets=subsets, calib_main=not a.no_calib_main)
    device = device_auto(a.device)
    kw = {}
    if spec.kind == "dense":
        kw = {k: v for k, v in {"max_tokens": a.max_tokens, "prefill_chunk": a.prefill_chunk, "prefill_min": a.prefill_min}.items() if v}
    if spec.kind in ("lg", "sliding", "oracle", "needle"):
        kw.update(stream=a.stream, chunk_windows=a.chunk_windows)
    runner = make_runner(P, spec, device=device, **kw)
    write_rows(runner, spec, rows, P.rows_path(a.exp, spec), manifest_sha=man)
    return 0


if __name__ == "__main__":
    sys.exit(main())
