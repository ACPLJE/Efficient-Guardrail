"""B1 / B2 — zero-shot guards on the long-context sets (plan S3 item 7, S5 B2, S6 B1; report §6.2, §6.3).

    python -m lgs3.longbench --guard qwen3guard_gen_8b --bench v2main --policy native --L 1024,2048,4096,8192,16384,28672
    python -m lgs3.longbench --guard qwen3guard_gen_8b --bench v2main --policy chunk_or:512:256      # CD W=512, overlap 256
    python -m lgs3.longbench --guard llamaguard3_8b --bench safetyniah --policy native
    python -m lgs3.longbench --summarize --bench v2main                                              # roll-up tables

Policies for inputs longer than the guard's context (Table 3): native (fits or fails), trunc_keep_head,
trunc_keep_tail, yarn (Qwen family, factor 4) and chunk_or:W:overlap (chunk-and-OR = CD: verdict OR over chunks,
score = max, token budget = processed / document tokens).  Rows carry the E0-2 lineage and are resumable by id.
Row: id, guard, policy, bench, L_target, depth, label, pred, score, parse_fail, n_tok_doc, n_tok_input, ctx,
     truncated, n_chunks, tokens_processed, budget_x, n_decode, ms, error
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from lgs1.lineage import Lineage, RowsWriter, canonical_hash, code_commit, dedupe_rows, dir_fingerprint, done_ids, iter_rows
from lgs1.metrics import auprc, f1_binary
from lgs2.config import S2Paths, depth_warning

HARNESS = "s3-longbench-v1"
# documented context limits that are smaller than config.max_position_embeddings (report §6.3, Table 3)
CTX_DOC: Dict[str, int] = {"nemotron_safety_guard_v3": 8192, "shieldgemma_9b": 8192, "bingoguard_8b": 4096,
                           "prompt_guard_2_86m": 512, "wildguard_7b": 32768, "polyguard_qwen_7b": 32768}
YARN_FAMILY = ("qwen3guard_gen_0.6b", "qwen3guard_gen_4b", "qwen3guard_gen_8b", "qwen3guard_stream_8b", "polyguard_qwen_7b")
VARIANT_META = ("variant", "n_decoys", "n_pieces", "frame", "multi_config", "related_m", "n_repeats", "decoy_source", "needle_share",
                "n_tok_needle", "pair_id")


# ----------------------------------------------------------------------------------------------
# policies
# ----------------------------------------------------------------------------------------------
def parse_policy(s: str) -> Dict:
    parts = s.split(":")
    name = parts[0]
    if name == "chunk_or":
        W = int(parts[1]) if len(parts) > 1 else 512
        ov = int(parts[2]) if len(parts) > 2 else W // 2
        return {"name": "chunk_or", "W": W, "overlap": ov, "tag": f"cd{W}o{ov}"}
    if name not in ("native", "trunc_keep_head", "trunc_keep_tail", "yarn"):
        raise ValueError(f"unknown policy {s}")
    return {"name": name, "tag": name}


def template_overhead(spec, tok) -> int:
    if spec.build is None:
        return 16
    return len(tok(spec.build(tok, ""), add_special_tokens=False)["input_ids"]) + int(spec.max_new_tokens) + 16


def context_limit(model_name: str, model, override: Optional[int] = None) -> Optional[int]:
    if override:
        return override
    if model_name in CTX_DOC:
        return CTX_DOC[model_name]
    cfg = getattr(model, "config", None) or getattr(getattr(model, "model", None), "config", None)
    if cfg is None:
        return None
    if getattr(cfg, "text_config", None) is not None:
        cfg = cfg.text_config
    return getattr(cfg, "max_position_embeddings", None)


def fit_doc(tok, text: str, ctx: Optional[int], overhead: int, policy: Dict) -> Tuple[str, int, bool]:
    """Return (doc text to feed, n_tok_doc, truncated) for native/truncation/yarn policies."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n = len(ids)
    if ctx is None or policy["name"] in ("native", "yarn") or n + overhead <= ctx:
        return text, n, False
    keep = max(16, ctx - overhead)
    kept = ids[:keep] if policy["name"] == "trunc_keep_head" else ids[-keep:]
    return tok.decode(kept, skip_special_tokens=True), n, True


def chunk_ids(ids: List[int], W: int, overlap: int) -> List[Tuple[int, int]]:
    if len(ids) <= W:
        return [(0, len(ids))]
    step = max(1, W - overlap)
    out = []
    s = 0
    while s < len(ids):
        e = min(len(ids), s + W)
        out.append((s, e))
        if e == len(ids):
            break
        s += step
    return out


# ----------------------------------------------------------------------------------------------
# model loading + scoring (reuses the S1 zero-shot harness)
# ----------------------------------------------------------------------------------------------
def load_guard(model_name: str, dtype: str, policy: Dict):
    from lgs1.zeroshot import guards as G
    from lgs1.zeroshot.run import _local_snapshot, _transformers_ge, load_model
    spec = G.get(model_name)
    snap = _local_snapshot(spec.repo_id)
    if policy["name"] == "yarn":
        if model_name not in YARN_FAMILY or spec.kind != "causal":
            raise SystemExit(f"yarn policy is defined for the Qwen family causal guards only, not {model_name}")
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from lgs4.backbone import apply_yarn
        cfg = apply_yarn(AutoConfig.from_pretrained(spec.repo_id))         # factor 4 (x4 window), 4.x / 5.x rope dicts
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        kw = dict(dtype=td) if _transformers_ge("4.56") else dict(torch_dtype=td)
        if torch.cuda.is_available():
            kw["device_map"] = "auto"
        tok = AutoTokenizer.from_pretrained(spec.repo_id)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec.repo_id, config=cfg, **kw).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            model = model.to(dev)
        return spec, tok, model, dev, snap
    tok, model, dev = load_model(spec, dtype)
    return spec, tok, model, dev, snap


def score_texts(spec, model_name: str, tok, model, dev, texts: List[str]) -> List[Dict]:
    """[{raw, label, score, n_decode}] for a batch of (already fitted) documents, any guard kind."""
    from lgs1.zeroshot import run as R
    from lgs2.screen import label_words, score_batch
    if spec.kind == "causal":
        built = [spec.build(tok, t) for t in texts]
        try:
            words, flagged = label_words(model_name, spec)
            res = score_batch(spec, tok, model, dev, built, words, flagged)
        except ValueError:                                    # no label tokens known: verdict only
            res = R.generate_batch(spec, tok, model, dev, built, want_score=True)
        out = []
        for r in res:
            lab = spec.parse(r["raw"])
            sc = r.get("score")
            if sc is None and lab is not None:
                sc = 1.0 if lab in ("unsafe", "controversial") else 0.0
            out.append({"raw": r["raw"][:300], "label": lab, "score": sc, "n_decode": r.get("n_decode", 0)})
        return out
    if spec.kind == "yesno":
        built = [spec.build(tok, t) for t in texts]
        res = R.yesno_batch(spec, tok, model, dev, built)
        return [{"raw": r["raw"], "label": "unsafe" if r["score"] > 0.5 else "safe", "score": r["score"], "n_decode": 0} for r in res]
    if spec.kind == "stream":
        return [_stream(R, spec, tok, model, dev, t) for t in texts]
    if spec.kind == "vlm":
        out = []
        for t in texts:
            r = R.vlm_one(spec, tok, model, dev, t)
            lab = spec.parse(r["raw"])
            out.append({"raw": r["raw"][:300], "label": lab, "score": (1.0 if lab == "unsafe" else 0.0) if lab else None, "n_decode": r.get("n_decode", 0)})
        return out
    if spec.kind == "classifier":
        res = R.classifier_batch(model, texts)
        return [{"raw": r["raw"][:300], "label": r["label"], "score": r["score"], "n_decode": 0} for r in res]
    raise ValueError(spec.kind)


def score_texts_prefill(spec, model_name: str, tok, model, dev, texts: List[str], chunk: int) -> List[Dict]:
    """Causal guards only: the same verdict / score as score_texts, produced through a chunked prefill (64k/128k on 48 GB)."""
    from lgs1.zeroshot.prefill import prefill_generate_texts
    from lgs2.screen import _label_ids, label_words
    if spec.kind != "causal":
        return score_texts(spec, model_name, tok, model, dev, texts)
    built = [spec.build(tok, t) for t in texts]
    try:
        words, flagged = label_words(model_name, spec)
        res = prefill_generate_texts(spec, tok, model, dev, built, chunk, label_ids=_label_ids(tok, words), flagged_idx=list(flagged))
    except ValueError:
        res = prefill_generate_texts(spec, tok, model, dev, built, chunk, first_token=list(spec.first_token) if spec.first_token else None)
    out = []
    for r in res:
        lab = spec.parse(r["raw"])
        sc = r.get("score")
        if sc is None and lab is not None:
            sc = 1.0 if lab in ("unsafe", "controversial") else 0.0
        out.append({"raw": r["raw"][:300], "label": lab, "score": sc, "n_decode": r.get("n_decode", 0)})
    return out


def _stream(R, spec, tok, model, dev, t) -> Dict:
    r = R.stream_one(spec, tok, model, dev, t)
    lab = r.get("label")
    return {"raw": r["raw"][:300], "label": lab, "score": (1.0 if lab in ("unsafe", "controversial") else 0.0) if lab else None, "n_decode": 0}


# ----------------------------------------------------------------------------------------------
# benches
# ----------------------------------------------------------------------------------------------
def _pair_index(pair_id: Optional[str]) -> Optional[int]:
    if not pair_id:
        return None
    m = re.search(r"-p(\d+)$", pair_id)
    return int(m.group(1)) if m else None


def load_bench(P: S2Paths, bench: str, subsets: Optional[List[str]] = None) -> Iterable[Dict]:
    """v2main -> synth/test/{main,benign_long}; v2:<subset,...> -> those test subsets; v2calib -> synth/calib/benign_long
    (matched-FPR thresholds for the zero-shot / CD rows, 2026-09-20); safetyniah -> the parquet."""
    if bench == "safetyniah":
        yield from load_safetyniah(P)
        return
    split = "test"
    if bench == "v2main":
        subsets = subsets or ["main", "benign_long"]
    elif bench == "v2calib":
        split, subsets = "calib", (subsets or ["benign_long"])
    elif bench.startswith("v2:"):
        subsets = [s for s in bench[3:].split(",") if s]
    else:
        raise ValueError(bench)
    for sub in subsets:
        f = P.synth / split / sub / "rows.jsonl.gz"
        if not f.exists():
            raise SystemExit(f"{f} missing (run S3 synthesis first)")
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    meta = {"haystack_domain": r.get("haystack_domain"), "needle_id": r.get("needle_id")}
                    for k in VARIANT_META:                      # robustness axes of the variant subsets (S6 tables)
                        if r.get(k) is not None:
                            meta[k] = r[k]
                    yield {"id": r["id"], "text": r["text"], "label": r["label"], "L_target": r.get("L_target"), "depth": r.get("depth"),
                           "subset": r["subset"], "pair": _pair_index(r.get("pair_id")), "meta": meta}


def load_safetyniah(P: S2Paths, niah_type: str = "prompt") -> Iterable[Dict]:
    import pyarrow.parquet as pq
    from lgs2.splits import _parquet_files
    files = _parquet_files("safetyniah", "caskcsg/SafetyNIAH", P)
    if not files:
        raise SystemExit("SafetyNIAH parquet not in the cache (S1 download --only safetyniah)")
    cols = ["sample_idx", "dataset", "label", "type", "target_length", "relevance", "position_pct", "prompt", "response", "is_duplicate"]
    for f in files:
        pf = pq.ParquetFile(f)
        use = [c for c in cols if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=256, columns=use):
            for r in batch.to_pylist():
                if str(r.get("type")) != niah_type or str(r.get("is_duplicate")).lower() == "true":
                    continue
                yield {"id": f"safetyniah-{r['sample_idx']}", "text": r["prompt"], "label": "harmful" if str(r["label"]).lower() == "unsafe" else "benign",
                       "L_target": int(r["target_length"]), "depth": float(r["position_pct"]) / 100.0 if r.get("position_pct") is not None else None,
                       "subset": f"niah_{r.get('relevance')}", "pair": None,
                       "meta": {"dataset": r.get("dataset"), "relevance": r.get("relevance"), "target_length_words": r["target_length"]}}


def filter_rows(rows: Iterable[Dict], Ls: Optional[List[int]], depths: Optional[List[float]], max_pairs: Optional[int],
                max_rows: Optional[int], max_bl: Optional[int] = None) -> Iterable[Dict]:
    """`max_bl` caps the benign_long rows per L (the 64k/128k subset: 128 pairs x 5 depths + 250 benign-long per L)."""
    n = 0
    bl_seen: Dict = {}
    for r in rows:
        if Ls and r.get("L_target") not in Ls:
            continue
        if depths and r.get("depth") is not None and not any(abs(r["depth"] - d) < 1e-6 for d in depths):
            continue
        if max_pairs is not None and r.get("pair") is not None and r["pair"] >= max_pairs:
            continue
        if max_bl is not None and r.get("subset") == "benign_long":
            c = bl_seen.get(r.get("L_target"), 0)
            if c >= max_bl:
                continue
            bl_seen[r.get("L_target")] = c + 1
        yield r
        n += 1
        if max_rows and n >= max_rows:
            return


# ----------------------------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------------------------
def run(P: S2Paths, guard: str, bench: str, policy_s: str, Ls=None, depths=None, max_pairs=None, max_rows=None,
        dtype: str = "bfloat16", batch: int = 8, ctx_override: Optional[int] = None, subsets=None, max_bl: Optional[int] = None,
        prefill_chunk: Optional[int] = None) -> Dict:
    """`prefill_chunk` (tokens): causal guards prefill inputs longer than it in chunks (lgs1.zeroshot.prefill) -- the 64k/128k
    rows of the 8B-20B guards on 48 GB; 0 / None = single-pass prefill as before."""
    import torch
    policy = parse_policy(policy_s)
    out_dir = P.out / "longbench" / bench.replace(":", "_").replace(",", "+")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{guard}.{policy['tag']}.rows.jsonl"
    done = done_ids(rows_path)
    kept = list(filter_rows(load_bench(P, bench, subsets), Ls, depths, max_pairs, max_rows, max_bl))
    w = depth_warning(kept, depths, f"{guard} {bench}")
    if w:
        print(w, file=sys.stderr, flush=True)
        print(w, flush=True)
    todo = [r for r in kept if r["id"] not in done]
    print(f"[longbench] {guard} {bench} {policy['tag']}: done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return {"guard": guard, "bench": bench, "policy": policy["tag"], "new": 0}
    spec, tok, model, dev, snap = load_guard(guard, dtype, policy)
    # a VLM guard (Llama Guard 4) hands back its AutoProcessor as `tok`: scoring goes through the processor, but token
    # counting / truncation / chunking must use its tokenizer -- processor(text) puts the text in the `images` slot and
    # raises "You have to specify text" (b1-calib 2026-09-20: 3,000 error rows).
    ctok = getattr(tok, "tokenizer", None) or tok or getattr(model, "tokenizer", None)   # classifier pipelines carry theirs (B6 chunk-OR)
    ctx = context_limit(guard, model, ctx_override)
    overhead = template_overhead(spec, ctok) if spec.kind in ("causal", "yesno") else 16
    cfg = dict(harness=HARNESS, guard=guard, repo=spec.repo_id, policy=policy, ctx=ctx, dtype=dtype, max_new_tokens=spec.max_new_tokens)
    # the chunked prefill computes the same greedy output; it is a memory device, not a policy -> not part of cfg_hash
    lin = Lineage(model_sha=dir_fingerprint(snap) if snap else "none", router_sha="none", cfg_hash=canonical_hash(cfg),
                  recipe_hash="none", data_manifest_sha="none", code_commit=code_commit(Path(__file__).resolve().parents[1]))
    print(f"[longbench] ctx={ctx} overhead={overhead} policy={policy}", flush=True)
    t_start = time.time()
    n = 0
    from lgs1.interrupt import Stop
    Stop.install()                                       # time-limit SIGTERM -> flush + rc=3 -> the array task requeues itself
    with RowsWriter(rows_path, lin, resume=rows_path.exists()) as w:
        for r in todo:
            t0 = time.perf_counter()
            base = {"id": r["id"], "guard": guard, "policy": policy["tag"], "bench": bench, "subset": r["subset"],
                    "L_target": r["L_target"], "depth": r["depth"], "label": 1 if r["label"] == "harmful" else 0, "ctx": ctx,
                    "meta": r.get("meta", {})}
            try:
                if policy["name"] == "chunk_or":
                    ids = ctok(r["text"], add_special_tokens=False)["input_ids"]
                    spans = chunk_ids(ids, policy["W"], policy["overlap"])
                    texts = [ctok.decode(ids[s:e], skip_special_tokens=True) for s, e in spans]
                    res = []
                    for i in range(0, len(texts), batch):
                        res += score_texts(spec, guard, tok, model, dev, texts[i:i + batch])
                    labs = [x["label"] for x in res]
                    scores = [x["score"] for x in res if x["score"] is not None]
                    flagged = any(l in ("unsafe", "controversial") for l in labs)
                    parse_fail = sum(1 for l in labs if l is None)
                    row = dict(base, pred=(1 if flagged else 0) if any(l is not None for l in labs) else None,
                               score=max(scores) if scores else None, parse_fail=parse_fail, n_tok_doc=len(ids),
                               n_tok_input=sum(e - s for s, e in spans) + overhead * len(spans), truncated=False, n_chunks=len(spans),
                               tokens_processed=sum(e - s for s, e in spans), budget_x=round(sum(e - s for s, e in spans) / max(1, len(ids)), 3),
                               n_decode=sum(x["n_decode"] for x in res), chunk_flags=[l for l in labs][:512])
                else:
                    text, n_doc, truncated = fit_doc(ctok, r["text"], ctx, overhead, policy)
                    n_in = len(ctok(text, add_special_tokens=False)["input_ids"]) + overhead
                    if policy["name"] == "native" and ctx and n_in > ctx and spec.kind != "classifier":
                        raise RuntimeError(f"input {n_in} tokens exceeds ctx {ctx} under the native policy")
                    if prefill_chunk and spec.kind == "causal" and n_in > prefill_chunk:
                        x = score_texts_prefill(spec, guard, tok, model, dev, [text], prefill_chunk)[0]
                    else:
                        x = score_texts(spec, guard, tok, model, dev, [text])[0]
                    row = dict(base, pred=(1 if x["label"] in ("unsafe", "controversial") else 0) if x["label"] else None,
                               score=x["score"], parse_fail=int(x["label"] is None), n_tok_doc=n_doc, n_tok_input=n_in,
                               truncated=truncated, n_chunks=1, tokens_processed=n_in - overhead, budget_x=round((n_in - overhead) / max(1, n_doc), 3),
                               n_decode=x["n_decode"], raw=x["raw"][:200],
                               prefill_chunk=(prefill_chunk if (prefill_chunk and spec.kind == "causal" and n_in > prefill_chunk) else None))
                row["error"] = None
            except Exception as e:  # OOM / context overflow / parser errors are recorded, never fatal (Table 3 policy column)
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                row = dict(base, pred=None, score=None, parse_fail=1, n_tok_doc=None, n_tok_input=None, truncated=None, n_chunks=None,
                           tokens_processed=None, budget_x=None, n_decode=0, error=f"{type(e).__name__}: {str(e)[:200]}")
            row["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            w.write(row)
            n += 1
            if n % 20 == 0:
                el = time.time() - t_start
                print(f"   {n}/{len(todo)}  {el:.0f}s  ({el / n:.1f} s/row)", flush=True)
            Stop.check(f"{guard} {policy['tag']} {n}/{len(todo)}")
    return {"guard": guard, "bench": bench, "policy": policy["tag"], "new": n, "seconds": round(time.time() - t_start, 1)}


# ----------------------------------------------------------------------------------------------
# summaries (LongGuard-style: R_u, R_s, F1_unsafe, FPR, budget x)
# ----------------------------------------------------------------------------------------------
def summarize_rows(rows: List[Dict], group_keys=("guard", "policy", "L_target")) -> List[Dict]:
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[tuple(r.get(k) for k in group_keys)].append(r)
    out = []
    for key, rs in sorted(g.items(), key=lambda kv: [str(x) for x in kv[0]]):
        valid = [r for r in rs if r.get("pred") is not None]
        y = [r["label"] for r in valid]
        p = [r["pred"] for r in valid]
        harm = [r for r in valid if r["label"] == 1]
        ben = [r for r in valid if r["label"] == 0]
        rec = dict(zip(group_keys, key))
        f1 = f1_binary(p, y) if valid else {"f1": float("nan"), "precision": float("nan"), "recall": float("nan"), "fpr": float("nan")}
        sc = [(r["score"], r["label"]) for r in valid if r.get("score") is not None]
        rec.update(n=len(rs), n_valid=len(valid), n_error=sum(1 for r in rs if r.get("error")),
                   parse_fail_rate=(sum(1 for r in rs if r.get("pred") is None) / max(1, len(rs))),
                   R_u=(sum(1 for r in harm if r["pred"] == 1) / len(harm)) if harm else None,
                   R_s=(sum(1 for r in ben if r["pred"] == 0) / len(ben)) if ben else None,
                   FPR=(sum(1 for r in ben if r["pred"] == 1) / len(ben)) if ben else None,
                   F1_unsafe=f1["f1"], auprc=(auprc([s for s, _ in sc], [l for _, l in sc]) if sc and len({l for _, l in sc}) == 2 else None),
                   budget_x_mean=(sum(r["budget_x"] for r in valid if r.get("budget_x") is not None) / max(1, sum(1 for r in valid if r.get("budget_x") is not None))),
                   truncated_rate=(sum(1 for r in valid if r.get("truncated")) / max(1, len(valid))),
                   ms_mean=(sum(r["ms"] for r in rs) / max(1, len(rs))), n_decode_mean=(sum(r.get("n_decode", 0) for r in rs) / max(1, len(rs))))
        out.append(rec)
    return out


def summarize(P: S2Paths, bench: str) -> Path:
    out_dir = P.out / "longbench" / bench.replace(":", "_").replace(",", "+")
    rows: List[Dict] = []
    for f in sorted(glob.glob(str(out_dir / "*.rows.jsonl"))):
        rows += dedupe_rows(iter_rows(f, tolerant=True))  # files still being appended by running jobs; one row per id
    table = summarize_rows(rows)
    (out_dir / "summary.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    lines = [f"# longbench {bench} ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "| guard | policy | L | n | R_u | R_s | FPR | F1_unsafe | AUPRC | parse fail | budget× | trunc | ms | decode tok |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    f = lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    for t in table:
        lines.append(f"| {t['guard']} | {t['policy']} | {t['L_target']} | {t['n']} | {f(t['R_u'])} | {f(t['R_s'])} | {f(t['FPR'])} | "
                     f"{f(t['F1_unsafe'])} | {f(t['auprc'])} | {f(t['parse_fail_rate'])} | {f(t['budget_x_mean'])} | {f(t['truncated_rate'])} | "
                     f"{t['ms_mean']:.0f} | {t['n_decode_mean']:.1f} |")
    md = P.out / "reports" / f"longbench_{bench.replace(':', '_').replace(',', '+')}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="B1/B2 long-context baselines")
    ap.add_argument("--guard")
    ap.add_argument("--bench", default="v2main", help="v2main | v2:<subset,...> | safetyniah")
    ap.add_argument("--policy", default="native", help="native | trunc_keep_head | trunc_keep_tail | yarn | chunk_or:W:overlap")
    ap.add_argument("--L", default="", help="comma list of L_target values to keep")
    ap.add_argument("--depths", default="", help="comma list of depths to keep")
    ap.add_argument("--max-pairs", type=int, default=None, help="keep pair indices < N (dense-baseline subset: 128)")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--max-bl", type=int, default=None, help="benign_long rows kept per L (64k/128k subset: 250)")
    ap.add_argument("--prefill-chunk", type=int, default=None, help="causal guards: chunked prefill above this many input tokens (e.g. 8192)")
    ap.add_argument("--batch", type=int, default=8, help="chunk batch for chunk_or")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--ctx", type=int, default=None, help="override the context limit")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(argv)
    P = S2Paths.load().ensure()
    if a.summarize:
        print(f"[longbench] written {summarize(P, a.bench)}")
        return 0
    if not a.guard:
        ap.error("--guard")
    Ls = [int(x) for x in a.L.split(",") if x] or None
    depths = [float(x) for x in a.depths.split(",") if x] or None
    r = run(P, a.guard, a.bench, a.policy, Ls, depths, a.max_pairs, a.max_rows, a.dtype, a.batch, a.ctx, max_bl=a.max_bl,
            prefill_chunk=a.prefill_chunk)
    print(f"[longbench] {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
