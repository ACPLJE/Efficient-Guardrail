"""D4a — background document pools, one per domain (report §5.3).

    python -m lgs2.haystack --domains all            # -> haystack/<domain>/docs.jsonl.gz + stats.json

Each record is a document PIECE (long documents are cut at paragraph boundaries into <= max_piece_tokens):
    {id, domain, source, source_id, piece, n_pieces, sep, pars: [...], par_tok: [...], n_tok, meta}
The text is `sep.join(pars)`; synthesis inserts needles only at paragraph boundaries, so paragraphs are the
unit stored.  Token counts use the Qwen3 tokenizer (the length grid's reference).  Reading stops when the
domain has budget_tokens * collect_factor tokens (screening will reject some documents).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .config import DOMAINS, DOMAIN_BY_NAME, SCREEN, Domain, S2Paths
from .textutil import split_paragraphs
from .tok import get_tok

SEP = {"blank": "\n\n", "code": "\n\n", "line": "\n"}


# ----------------------------------------------------------------------------------------------
# source readers -> (source_id, text, meta)
# ----------------------------------------------------------------------------------------------
def _read_jsonl_gz(path: Path) -> Iterator[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _hub_snapshot(repo_id: str, hf_home: Path) -> Optional[Path]:
    d = hf_home / "hub" / f"datasets--{repo_id.replace('/', '--')}" / "snapshots"
    if not d.exists():
        return None
    snaps = sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.stat().st_mtime)
    return snaps[-1] if snaps else None


def iter_stream(dom: Domain, P: S2Paths) -> Iterator[Tuple[str, str, Dict]]:
    f = P.sources / dom.source / "docs.jsonl.gz"
    if not f.exists():
        raise FileNotFoundError(f"{f} (run S1 download: bash slurm/s1_download_login.sh --only {dom.source})")
    for row in _read_jsonl_gz(f):
        meta = row.get("meta") or {}
        ok = True
        for k, want in dom.filters.items():
            if k in meta and meta[k] != want:
                ok = False
                break
        if not ok:
            continue
        yield str(row.get("id")), row.get(dom.text_field) or row.get("text") or "", meta


def iter_parquet(dom: Domain, P: S2Paths) -> Iterator[Tuple[str, str, Dict]]:
    import pyarrow.parquet as pq
    snap = None
    rev = P.sources / dom.source / "revision.json"
    if rev.exists():
        lp = json.loads(rev.read_text()).get("local_path")
        if lp and Path(lp).exists():
            snap = Path(lp)
    snap = snap or _hub_snapshot(dom.hf_repo, P.hf_home)
    if snap is None:
        raise FileNotFoundError(f"{dom.hf_repo} not in the HF cache (S1 download --only {dom.source})")
    pat = f"**/{dom.hf_config_dir}/**/*.parquet" if dom.hf_config_dir else "**/*.parquet"
    files = sorted(glob.glob(str(snap / pat), recursive=True))
    if dom.hf_config_dir and not files:
        files = sorted(glob.glob(str(snap / f"**/*{dom.hf_config_dir}*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {snap} ({pat})")
    for fp in files:
        pf = pq.ParquetFile(fp)
        if dom.text_field not in pf.schema_arrow.names:
            raise KeyError(f"{fp}: column {dom.text_field!r} not in {pf.schema_arrow.names}")
        row = 0
        for batch in pf.iter_batches(batch_size=512, columns=[dom.text_field]):
            for text in batch.column(0).to_pylist():
                yield f"{Path(fp).stem}-{row}", text or "", {"file": Path(fp).name}
                row += 1


# --- AgentDojo: benign tool outputs rendered from the suite environments --------------------------
def _render_value(v, indent: int = 0) -> List[str]:
    pad = "  " * indent
    if isinstance(v, dict):
        out = []
        for k, x in v.items():
            if isinstance(x, (dict, list)):
                out.append(f"{pad}{k}:")
                out += _render_value(x, indent + 1)
            else:
                out.append(f"{pad}{k}: {x}")
        return out
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, (dict, list)):
                out.append(f"{pad}-")
                out += _render_value(x, indent + 1)
            else:
                out.append(f"{pad}- {x}")
        return out
    return [f"{pad}{v}"]


def iter_agentdojo(dom: Domain, P: S2Paths, seed: int = 0) -> Iterator[Tuple[str, str, Dict]]:
    """Every YAML/JSON environment file of the AgentDojo suites becomes tool-output documents in three
    renderings (yaml-like listing, JSON dump, per-record 'tool call result' blocks)."""
    import yaml
    repo = P.sources / dom.source / "repo"
    if not repo.exists():
        raise FileNotFoundError(f"{repo}: AgentDojo clone missing (S1 download --only agentdojo)")
    files = sorted(list(repo.glob("src/agentdojo/data/suites/**/*.yaml")) + list(repo.glob("src/agentdojo/data/suites/**/*.yml"))
                   + list(repo.glob("src/agentdojo/data/suites/**/*.json")))
    if not files:
        files = sorted(repo.glob("**/suites/**/*.y*ml"))
    for fp in files:
        try:
            raw = fp.read_text(encoding="utf-8")
            obj = json.loads(raw) if fp.suffix == ".json" else yaml.safe_load(raw)
        except Exception as e:  # templated yaml etc.
            yield f"{fp.parent.name}-{fp.stem}-raw", raw, {"file": str(fp.relative_to(repo)), "render": "raw", "error": str(e)[:80]}
            continue
        rel = str(fp.relative_to(repo))
        yield f"{fp.parent.name}-{fp.stem}-listing", "\n".join(_render_value(obj)), {"file": rel, "render": "listing"}
        yield f"{fp.parent.name}-{fp.stem}-json", json.dumps(obj, indent=2, ensure_ascii=False, default=str), {"file": rel, "render": "json"}
        # per-record tool results (emails / events / files / transactions ...)
        if isinstance(obj, dict):
            blocks = []
            for k, v in obj.items():
                items = v if isinstance(v, list) else (list(v.values()) if isinstance(v, dict) else [v])
                for i, it in enumerate(items):
                    body = json.dumps(it, ensure_ascii=False, default=str) if isinstance(it, (dict, list)) else str(it)
                    blocks.append(f"[tool_result name={k} index={i}]\n{body}\n[/tool_result]")
            if blocks:
                yield f"{fp.parent.name}-{fp.stem}-tools", "\n".join(blocks), {"file": rel, "render": "tool_results"}


READERS = {"stream": iter_stream, "parquet": iter_parquet, "agentdojo": iter_agentdojo}


# ----------------------------------------------------------------------------------------------
# pieces
# ----------------------------------------------------------------------------------------------
def _split_long_paragraph(par: str, style: str, max_chars: int) -> List[str]:
    if len(par) <= max_chars:
        return [par]
    units = [u for u in par.split("\n") if u.strip()] if style != "blank" else [u for u in re.split(r"(?<=[.!?])\s+", par) if u.strip()]
    if len(units) <= 1:
        return [par[i:i + max_chars] for i in range(0, len(par), max_chars)]
    out, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > max_chars:
            out.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        out.append(cur)
    return out


def make_pieces(text: str, dom: Domain, tok, max_piece_tokens: Optional[int] = None) -> List[Tuple[List[str], List[int]]]:
    """Paragraphs + per-paragraph token counts, grouped into pieces of <= max_piece_tokens."""
    mx = max_piece_tokens or dom.max_piece_tokens
    pars = split_paragraphs(text, dom.paragraph)
    if not pars:
        return []
    fixed: List[str] = []
    for p in pars:
        fixed += _split_long_paragraph(p, dom.paragraph, max_chars=mx * 3)   # ~3 chars/token upper bound
    counts = tok.count(fixed)
    pieces: List[Tuple[List[str], List[int]]] = []
    cur_p, cur_c, cur_n = [], [], 0
    for p, c in zip(fixed, counts):
        if cur_p and cur_n + c > mx:
            pieces.append((cur_p, cur_c))
            cur_p, cur_c, cur_n = [], [], 0
        cur_p.append(p)
        cur_c.append(c)
        cur_n += c
    if cur_p:
        pieces.append((cur_p, cur_c))
    return pieces


def build_domain(dom: Domain, P: S2Paths, tok, budget_factor: float = SCREEN["collect_factor"],
                 limit_docs: Optional[int] = None, max_piece_tokens: Optional[int] = None) -> Dict:
    out_dir = P.haystack / dom.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "docs.jsonl.gz"
    target = int(dom.budget_tokens * budget_factor)
    n_src = n_pieces = tokens = skipped_short = 0
    t0 = time.time()
    sep = SEP[dom.paragraph]
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for source_id, text, meta in READERS[dom.kind](dom, P):
            if tokens >= target or (limit_docs and n_src >= limit_docs):
                break
            pieces = make_pieces(text, dom, tok, max_piece_tokens)
            if not pieces:
                continue
            total = sum(sum(c) for _, c in pieces)
            if total < dom.min_doc_tokens:
                skipped_short += 1
                continue
            n_src += 1
            sid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(source_id))[:80]
            for k, (pars, cnt) in enumerate(pieces):
                rec = {"id": f"{dom.name}-{sid}-p{k}", "domain": dom.name, "source": dom.source, "source_id": sid,
                       "piece": k, "n_pieces": len(pieces), "sep": sep, "pars": pars, "par_tok": cnt, "n_tok": sum(cnt),
                       "meta": {k2: v for k2, v in (meta or {}).items() if isinstance(v, (str, int, float, bool))}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pieces += 1
                tokens += sum(cnt)
                if tokens >= target:
                    break
    stats = {"domain": dom.name, "role": dom.role, "source": dom.source, "n_source_docs": n_src, "n_pieces": n_pieces,
             "tokens": tokens, "budget_tokens": dom.budget_tokens, "collect_target": target,
             "reached_target": tokens >= target, "skipped_short": skipped_short, "tokenizer": getattr(tok, "name", "?"),
             "seconds": round(time.time() - t0, 1), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_docs(P: S2Paths, domain: str, keep: Optional[set] = None) -> Iterator[Dict]:
    f = P.haystack / domain / "docs.jsonl.gz"
    if not f.exists():
        return iter(())
    def gen():
        for rec in _read_jsonl_gz(f):
            if keep is None or rec["id"] in keep:
                yield rec
    return gen()


def doc_text(rec: Dict) -> str:
    return rec["sep"].join(rec["pars"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D4a background document pools")
    ap.add_argument("--domains", default="all", help="comma list or 'all'")
    ap.add_argument("--tok", default="qwen3", help="qwen3 | whitespace (tests)")
    ap.add_argument("--limit-docs", type=int, default=None, help="stop after N source documents (smoke)")
    ap.add_argument("--budget-factor", type=float, default=SCREEN["collect_factor"])
    ap.add_argument("--max-piece-tokens", type=int, default=None)
    a = ap.parse_args(argv)
    P = S2Paths.load().ensure()
    names = [d.name for d in DOMAINS] if a.domains == "all" else [x.strip() for x in a.domains.split(",") if x.strip()]
    tok = get_tok(a.tok)
    rc = 0
    for n in names:
        dom = DOMAIN_BY_NAME[n]
        try:
            st = build_domain(dom, P, tok, a.budget_factor, a.limit_docs, a.max_piece_tokens)
            flag = "" if st["reached_target"] else "  (below target: source exhausted)"
            print(f"[haystack] {n:10s} pieces={st['n_pieces']:6d} tokens={st['tokens']:>12,} / target {st['collect_target']:>12,}{flag}", flush=True)
        except Exception as e:
            rc = 1
            print(f"[haystack] {n}: FAILED {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
