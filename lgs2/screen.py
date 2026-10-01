"""D4b — background screening with two 8B guards (report §5.3).

    python -m lgs2.screen --domain govreport --model llamaguard3_8b        # one (domain, model) task (GPU)
    python -m lgs2.screen --summarize                                     # merge -> keep lists + summary (CPU)

Every document piece is cut into 512-token chunks (Qwen3 tokenizer grid); each chunk is moderated as a user
prompt by Llama Guard 3-8B and by Qwen3Guard-Gen-8B in STRICT mode (controversial counts as unsafe).  A piece
is rejected when ANY chunk is flagged by EITHER model.  Rows (one per chunk, with lineage) go to
    screen/<domain>.<model>.rows.jsonl
and `--summarize` writes screen/keep_<domain>.json (kept piece ids), screen/screen_summary.json (pass rate and
score histograms per domain/model) and screen/manual_review_50.md (50 rejected pieces for the manual check).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from lgs1.lineage import Lineage, RowsWriter, code_commit, dir_fingerprint, done_ids, iter_rows
from .config import DOMAINS, DOMAIN_BY_NAME, SCREEN, S2Paths
from .haystack import doc_text, load_docs
from .tok import chunk_by_tokens, get_tok

HARNESS = "s2-screen-v1"
QWEN_LABELS = (" Unsafe", " Safe", " Controversial")


def iter_chunks(P: S2Paths, domain: str, tok, chunk_tokens: int = SCREEN["chunk_tokens"], limit: Optional[int] = None) -> Iterator[Dict]:
    n = 0
    for rec in load_docs(P, domain):
        text = doc_text(rec)
        for c, s, e in chunk_by_tokens(text, tok, chunk_tokens):
            yield {"id": f"{rec['id']}#c{c}", "doc_id": rec["id"], "chunk": c, "text": text[s:e], "char_span": [s, e]}
            n += 1
            if limit and n >= limit:
                return


# ----------------------------------------------------------------------------------------------
# scoring: P(flag) per chunk
# ----------------------------------------------------------------------------------------------
def _label_ids(tok, words: Tuple[str, ...]) -> List[int]:
    return [tok.encode(w, add_special_tokens=False)[0] for w in words]


def label_words(model_name: str, spec) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """(label words, indices that count as flagged).  Qwen3Guard: 'Safety: Unsafe|Safe|Controversial' (strict ->
    Unsafe + Controversial); Llama Guard-like: first generated token 'unsafe' | 'safe'."""
    if model_name.startswith("qwen3guard"):
        return QWEN_LABELS, (0, 2) if SCREEN["qwen3guard_mode"] == "strict" else (0,)
    if spec.first_token:
        return tuple(spec.first_token), (0,)
    raise ValueError(f"{model_name}: no label tokens known for scoring")


def score_batch(spec, tok, model, dev, prompts: List[str], words: Tuple[str, ...], flagged_idx: Tuple[int, ...]) -> List[Dict]:
    """Greedy verdict + P(flag): at the first generated step whose token is one of `words`, softmax over the
    label tokens and sum the flagged ones.  None when no label token was generated (parse failure)."""
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc.pop("token_type_ids", None)
    enc = enc.to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=spec.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                             output_scores=True, return_dict_in_generate=True)
    L = enc["input_ids"].shape[1]
    new = out.sequences[:, L:]
    texts = tok.batch_decode(new, skip_special_tokens=False)
    ids = _label_ids(tok, words)
    res = []
    for i, t in enumerate(texts):
        p = None
        for step in range(min(len(out.scores), new.shape[1])):
            if int(new[i, step]) in ids:
                probs = torch.softmax(out.scores[step][i].float()[ids], dim=-1)
                p = float(sum(float(probs[j]) for j in flagged_idx))
                break
        res.append({"raw": t, "score": p})
    return res


def flag_of(model_name: str, label: Optional[str]) -> Optional[bool]:
    """None = unparsed.  strict: qwen3guard controversial -> flagged."""
    if label is None:
        return None
    if label == "unsafe":
        return True
    if label == "controversial":
        return SCREEN["qwen3guard_mode"] == "strict"
    return False


def run(domain: str, model_name: str, P: S2Paths, batch: Optional[int] = None, limit: Optional[int] = None,
        dtype: str = "bfloat16") -> Dict:
    from lgs1.zeroshot import guards as G
    from lgs1.zeroshot.run import _local_snapshot, load_model
    spec = G.get(model_name)
    P.ensure()
    qtok = get_tok("qwen3")
    rows_path = P.screen / f"{domain}.{model_name}.rows.jsonl"
    snap = _local_snapshot(spec.repo_id)
    cfg = dict(harness=HARNESS, domain=domain, model=model_name, repo=spec.repo_id, chunk_tokens=SCREEN["chunk_tokens"],
               mode=SCREEN["qwen3guard_mode"], max_new_tokens=spec.max_new_tokens, dtype=dtype)
    lin = Lineage(model_sha=dir_fingerprint(snap) if snap else "none", router_sha="none",
                  cfg_hash=Lineage.build(None, None, cfg, {}, None).cfg_hash, recipe_hash="none",
                  data_manifest_sha="none", code_commit=code_commit(Path(__file__).resolve().parents[1]))
    done = done_ids(rows_path)
    todo = [c for c in iter_chunks(P, domain, qtok, limit=limit) if c["id"] not in done]
    print(f"[screen] {domain} x {model_name}: done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return {"domain": domain, "model": model_name, "done": len(done), "new": 0}
    if snap is None and os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        print(f"[skip] {model_name}: {spec.repo_id} is not in the HF cache and this node is offline "
              f"(S1: bash slurm/s1_download_login.sh --only {model_name}; python -m lgs1.download --fix-refs)", file=sys.stderr)
        raise SystemExit(3)
    print(f"[load] {spec.repo_id} (snapshot={snap})", flush=True)
    tok, model, dev = load_model(spec, dtype)
    bs = batch or SCREEN["batch_size"]
    words, flagged_idx = label_words(model_name, spec)
    t0 = time.time()
    with RowsWriter(rows_path, lin, resume=rows_path.exists()) as w:
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            built = [spec.build(tok, c["text"]) for c in chunk]
            res = score_batch(spec, tok, model, dev, built, words, flagged_idx)
            for c, r in zip(chunk, res):
                label = spec.parse(r["raw"])
                w.write({"id": c["id"], "doc_id": c["doc_id"], "chunk": c["chunk"], "char_span": c["char_span"],
                         "label": label, "flag": flag_of(model_name, label), "score": r["score"],
                         "raw": r["raw"][:200], "model": model_name, "domain": domain})
            if (i // bs) % 50 == 0:
                el = time.time() - t0
                print(f"   {i + len(chunk)}/{len(todo)}  {el:.0f}s  ({(i + len(chunk)) / max(el, 1e-6):.1f} chunks/s)", flush=True)
    return {"domain": domain, "model": model_name, "done": len(done), "new": len(todo), "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------------------------
# summary / keep lists / manual review sample
# ----------------------------------------------------------------------------------------------
def _hist(scores: List[float], bins: int = 20) -> List[int]:
    h = [0] * bins
    for s in scores:
        if s is None:
            continue
        k = min(bins - 1, max(0, int(s * bins)))
        h[k] += 1
    return h


def summarize(P: S2Paths, models: Optional[List[str]] = None, n_review: int = SCREEN["manual_review_docs"],
              seed: int = 0) -> Dict:
    models = models or SCREEN["models"]
    summary: Dict = {"chunk_tokens": SCREEN["chunk_tokens"], "mode": SCREEN["qwen3guard_mode"], "models": models,
                     "domains": {}, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rejected_examples: List[Dict] = []
    for dom in DOMAINS:
        docs = {rec["id"]: rec["n_tok"] for rec in load_docs(P, dom.name)}
        if not docs:
            summary["domains"][dom.name] = {"status": "no haystack"}
            continue
        per_model: Dict[str, Dict] = {}
        flagged_docs: Dict[str, Dict] = {}          # doc_id -> first flagged chunk info
        chunks_seen: Dict[str, set] = {}
        complete = True
        for m in models:
            f = P.screen / f"{dom.name}.{m}.rows.jsonl"
            if not f.exists():
                per_model[m] = {"status": "missing"}
                complete = False
                continue
            n = n_flag = n_unparsed = 0
            fdocs = set()
            scores: List[float] = []
            seen = set()
            for r in iter_rows(f):
                n += 1
                seen.add(r["doc_id"])
                if r.get("score") is not None:
                    scores.append(float(r["score"]))
                if r.get("flag") is None:
                    n_unparsed += 1
                elif r["flag"]:
                    n_flag += 1
                    fdocs.add(r["doc_id"])
                    flagged_docs.setdefault(r["doc_id"], {"model": m, "label": r.get("label"), "score": r.get("score"),
                                                          "chunk": r.get("chunk"), "char_span": r.get("char_span")})
            chunks_seen[m] = seen
            per_model[m] = {"status": "ok", "n_chunks": n, "n_flag_chunks": n_flag, "n_flag_docs": len(fdocs),
                            "n_unparsed": n_unparsed, "n_docs_scored": len(seen), "score_hist20": _hist(scores),
                            "mean_score": (sum(scores) / len(scores)) if scores else None}
            if len(seen) < len(docs):
                complete = False
        scored_by_all = set(docs) if complete else set.intersection(*[s for s in chunks_seen.values()]) if chunks_seen else set()
        keep = sorted(d for d in scored_by_all if d not in flagged_docs)
        kept_tokens = sum(docs[d] for d in keep)
        rej = sorted(flagged_docs)
        summary["domains"][dom.name] = {
            "status": "complete" if complete else "incomplete", "n_docs": len(docs), "tokens": sum(docs.values()),
            "n_docs_scored_by_all": len(scored_by_all), "n_rejected": len(rej), "n_kept": len(keep),
            "pass_rate": (len(keep) / len(scored_by_all)) if scored_by_all else None,
            "kept_tokens": kept_tokens, "budget_tokens": dom.budget_tokens, "budget_met": kept_tokens >= dom.budget_tokens,
            "models": per_model}
        keep_path = P.screen / f"keep_{dom.name}.json"
        if complete:
            keep_path.write_text(json.dumps({"domain": dom.name, "complete": True, "keep": keep}), encoding="utf-8")
        elif keep_path.exists():
            keep_path.unlink()          # never leave a partial keep list behind: splits/synth must see 'unscreened'

        rng = random.Random(f"{seed}:{dom.name}")
        for d in rng.sample(rej, min(len(rej), max(1, n_review // len(DOMAINS) + 2))) if rej else []:
            rejected_examples.append({"domain": dom.name, "doc_id": d, **flagged_docs[d]})
    (P.screen / "screen_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_manual_review(P, rejected_examples[:n_review])
    return summary


def _write_manual_review(P: S2Paths, examples: List[Dict]) -> None:
    lines = [f"# Screening — manual check of {len(examples)} rejected pieces ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "Mark each as `agree` (harmful/sensitive content really present) or `disagree` (false rejection). "
             "Disagreements are recorded in audit.json (screening precision); nothing is un-rejected automatically.", ""]
    by_dom: Dict[str, List[Dict]] = {}
    for e in examples:
        by_dom.setdefault(e["domain"], []).append(e)
    docs_cache: Dict[str, Dict] = {}
    for dom, exs in by_dom.items():
        want = {e["doc_id"] for e in exs}
        for rec in load_docs(P, dom, keep=want):
            docs_cache[rec["id"]] = rec
        for i, e in enumerate(exs, 1):
            rec = docs_cache.get(e["doc_id"])
            text = doc_text(rec) if rec else ""
            s, t = e.get("char_span") or [0, 0]
            excerpt = text[s:t][:800].replace("\n", " ⏎ ")
            lines += [f"## {dom} · {e['doc_id']} · chunk {e.get('chunk')} · {e['model']} → {e.get('label')} (p={e.get('score')})",
                      "", f"> {excerpt}", "", "verdict: [ ] agree  [ ] disagree   note:", ""]
    (P.screen / "manual_review_50.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_keep(P: S2Paths, domain: str) -> Optional[set]:
    """Kept piece ids, or None when the domain has not been (completely) screened."""
    f = P.screen / f"keep_{domain}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if not d.get("complete", False):
        return None
    return set(d.get("keep", []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D4b background screening")
    ap.add_argument("--domain")
    ap.add_argument("--model")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N chunks only (smoke)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(argv)
    P = S2Paths.load().ensure()
    if a.summarize:
        s = summarize(P)
        for d, v in s["domains"].items():
            if v.get("status") in ("complete", "incomplete"):
                print(f"[screen] {d:10s} {v['status']:10s} docs={v['n_docs']:6d} kept={v['n_kept']:6d} "
                      f"pass={v['pass_rate'] if v['pass_rate'] is None else round(v['pass_rate'], 3)} "
                      f"kept_tokens={v['kept_tokens']:>12,} budget_met={v['budget_met']}")
            else:
                print(f"[screen] {d:10s} {v.get('status')}")
        print(f"[written] {P.screen / 'screen_summary.json'}  {P.screen / 'manual_review_50.md'}")
        return 0
    if not (a.domain and a.model):
        ap.error("--domain and --model (or --summarize)")
    r = run(a.domain, a.model, P, a.batch, a.limit, a.dtype)
    print(f"[screen] {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
