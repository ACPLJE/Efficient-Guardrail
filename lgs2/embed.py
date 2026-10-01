"""bge-m3 chunk embeddings for the `related` variant subset (report §5.5; plan S2 D4 'related용 bge-m3 임베딩').

    python -m lgs2.embed --domain govreport            # GPU: related/<domain>.emb.npy + .chunks.jsonl.gz
    python -m lgs2.embed --retrieve needles.jsonl --domains govreport,pg19 --m 8   # S3: top-m chunks per needle

Dense bge-m3 vectors = CLS token, L2-normalised (the model card's dense retrieval recipe).  Only pieces that
passed screening are embedded when screen/keep_<domain>.json exists.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import S2Paths
from .screen import iter_chunks, load_keep
from .tok import get_tok

BGE = "BAAI/bge-m3"


def _load_bge(dtype: str = "float16"):
    import torch
    import transformers
    from packaging.version import Version
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype if dev == "cuda" else "float32"]
    kw = {"dtype": td} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": td}
    try:
        tok = AutoTokenizer.from_pretrained(BGE)
        model = AutoModel.from_pretrained(BGE, **kw).to(dev).eval()
    except (AttributeError, OSError, EnvironmentError) as e:
        # offline node + snapshot without weights: BAAI/bge-m3 ships pytorch_model.bin only, which the default D3
        # model patterns ignored (2026-09-18) -> from_pretrained dies with "'NoneType' object has no attribute 'endswith'"
        raise SystemExit(f"[embed] cannot load {BGE} from the local HF cache ({type(e).__name__}: {str(e)[:120]}). "
                         f"On the login node: bash slurm/s1_download_login.sh --only bge_m3 --no-skip   "
                         f"(fetches pytorch_model.bin, 2.3 GB), then resubmit: python s2ctl.py embed --sbatch") from e
    return tok, model, dev


def embed_texts(tok, model, dev, texts: List[str], batch: int = 32, max_len: int = 512) -> np.ndarray:
    import torch
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = model(**enc).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h.float(), dim=-1)
        out.append(h.cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, 1024), dtype=np.float16)


def embed_domain(domain: str, P: S2Paths, batch: int = 32, limit: Optional[int] = None) -> Dict:
    P.ensure()
    keep = load_keep(P, domain)
    qtok = get_tok("qwen3")
    chunks = [c for c in iter_chunks(P, domain, qtok, limit=limit) if keep is None or c["doc_id"] in keep]
    emb_path, idx_path = P.related / f"{domain}.emb.npy", P.related / f"{domain}.chunks.jsonl.gz"
    if emb_path.exists() and idx_path.exists():
        n_prev = np.load(emb_path, mmap_mode="r").shape[0]
        if n_prev == len(chunks):
            return {"domain": domain, "n_chunks": len(chunks), "status": "already embedded"}
    tok, model, dev = _load_bge()
    t0 = time.time()
    emb = embed_texts(tok, model, dev, [c["text"] for c in chunks], batch)
    np.save(emb_path, emb)
    with gzip.open(idx_path, "wt", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c["id"], "doc_id": c["doc_id"], "chunk": c["chunk"], "char_span": c["char_span"]}) + "\n")
    return {"domain": domain, "n_chunks": len(chunks), "screened_only": keep is not None, "seconds": round(time.time() - t0, 1)}


def retrieve(P: S2Paths, needle_texts: List[str], domains: List[str], m: int = 8, batch: int = 32) -> List[List[Dict]]:
    """top-m most similar screened chunks (over the given domains) for every needle text."""
    tok, model, dev = _load_bge()
    q = embed_texts(tok, model, dev, needle_texts, batch).astype(np.float32)
    mats, meta = [], []
    for d in domains:
        e, i = P.related / f"{d}.emb.npy", P.related / f"{d}.chunks.jsonl.gz"
        if not e.exists():
            continue
        mats.append(np.load(e).astype(np.float32))
        with gzip.open(i, "rt", encoding="utf-8") as f:
            meta += [dict(json.loads(l), domain=d) for l in f if l.strip()]
    if not mats:
        return [[] for _ in needle_texts]
    M = np.concatenate(mats)
    out = []
    for qi in q:
        sims = M @ qi
        top = np.argpartition(-sims, min(m, len(sims) - 1))[:m]
        top = top[np.argsort(-sims[top])]
        out.append([dict(meta[j], sim=float(sims[j])) for j in top])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retrieve", help="jsonl of {id, text} needles -> prints top-m chunks (smoke)")
    ap.add_argument("--domains", default="")
    ap.add_argument("--m", type=int, default=8)
    a = ap.parse_args(argv)
    P = S2Paths.load().ensure()
    if a.retrieve:
        rows = [json.loads(l) for l in open(a.retrieve, encoding="utf-8") if l.strip()]
        res = retrieve(P, [r["text"] for r in rows], [d for d in a.domains.split(",") if d], a.m)
        for r, hits in zip(rows, res):
            print(r.get("id"), [(h["id"], round(h["sim"], 3)) for h in hits])
        return 0
    if not a.domain:
        ap.error("--domain")
    print(f"[embed] {embed_domain(a.domain, P, a.batch, a.limit)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
