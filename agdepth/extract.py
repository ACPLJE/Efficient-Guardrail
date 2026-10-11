"""Per-layer pooled hidden states for every window (embeddings = layer 0 ... last layer = L).

Framings
  raw   the window alone, no special tokens, positions from 0 -- what the LG trunk sees (block-local windows).
  task  the window as an untrusted document inside the backbone's own chat template, after a benign user task
        (16 tasks, picked by a hash of the window id, so independent of the class).  This is what a target model
        sees when it serves the request; the injection literature probes this framing (TaskTracker, PIShield).
        The window part is tokenized with split_special_tokens=True, so special-token strings inside documents stay
        plain text (the template's own special tokens are parsed normally).
Poolings
  mean  mean over the window tokens (raw: without position 0, the attention-sink token)
  last  raw: last window token; task: last prompt token (the position that emits the first answer token)
  span  mean over the inserted snippet's tokens (diagnostic: removes the dilution by background tokens)
Note: transformers returns the last layer's hidden state AFTER the final norm; every class gets the same treatment.

Output  <run>/feats/<bb>/<framing>/<pool>.f16  float16 memmap [L+1, N, d] (one contiguous block per layer),
        ids.json, aux.npz (n_tokens, n_window_tokens, n_span_tokens), meta.json, progress.json (resumable shards).

    python -m agdepth.extract --run NAME --backbone qwen3guard_0.6b [--framings raw,task] [--poolings mean,last]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .config import DEFAULT_POOLINGS, FRAMINGS, POOLINGS, PROFILES, AGPaths, backbone

TASK_PROMPTS = [
    "Summarize the following document in three bullet points.",
    "What is the main topic of the document below?",
    "List all people and organizations mentioned in the document.",
    "Extract every date that appears in the following text.",
    "Write a one-sentence summary of the document.",
    "Translate the first two sentences of the document into French.",
    "Does the document below mention any deadlines? Answer briefly.",
    "Classify the tone of the following text as formal, neutral, or informal.",
    "Who is the intended audience of this document?",
    "Draft a short reply acknowledging receipt of the following message.",
    "Give the document a concise title.",
    "List three keywords that describe the document.",
    "What problem does the following document discuss?",
    "Rewrite the key point of the document in plain language.",
    "List the factual claims made in the following text.",
    "Explain the document to a new team member in two sentences.",
]
SHARD = 1024
FP16_MAX = 65000.0
_WARNED: Dict[str, bool] = {}


def task_of(wid: str) -> str:
    return TASK_PROMPTS[int(hashlib.sha1(wid.encode()).hexdigest()[:8], 16) % len(TASK_PROMPTS)]


# ----------------------------------------------------------------------------------------------
# tokenization with window / span masks
# ----------------------------------------------------------------------------------------------
def _enc(tok, text: str, split_special: bool = False) -> Tuple[List[int], List[Tuple[int, int]]]:
    kw = dict(add_special_tokens=False, return_offsets_mapping=True, return_attention_mask=False)
    if split_special:
        try:
            e = tok(text, split_special_tokens=True, **kw)
        except TypeError:
            e = tok(text, **kw)
    else:
        e = tok(text, **kw)
    return list(e["input_ids"]), [tuple(x) for x in e["offset_mapping"]]


def _render_task(tok, window_text: str, wid: str) -> Tuple[str, str]:
    content_head = f"{task_of(wid)}\n\n<document>\n"
    content = content_head + window_text + "\n</document>"
    msgs = [{"role": "user", "content": content}]
    prompt = None
    if getattr(tok, "chat_template", None):
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = None
    if not prompt or window_text not in prompt:
        if not _WARNED.get("template"):
            _WARNED["template"] = True
            print(f"[extract] WARNING the chat template of {getattr(tok, 'name_or_path', '?')} does not embed the document "
                  f"verbatim (or failed); task framing falls back to a plain 'User: ... Assistant:' prompt",
                  file=sys.stderr, flush=True)
        prompt = f"User: {content}\nAssistant:"
    i = prompt.find(window_text)
    return prompt[:i], prompt[i + len(window_text):]


def encode(tok, row: Dict, framing: str, max_len: int) -> Dict:
    """ids + boolean masks (window tokens, snippet tokens) + index of the last-pooling token."""
    text = row["text"]
    a, b = row["snip_span"]
    if framing == "raw":
        ids, offs = _enc(tok, text, split_special=True)
        ids, offs = ids[:max_len], offs[:max_len]
        win = np.ones(len(ids), dtype=bool)
        win[0] = False                                         # position 0 = attention sink, excluded from mean
        span = np.array([(e > a and s < b) for s, e in offs], dtype=bool)
        last = len(ids) - 1
    elif framing == "task":
        pre, suf = _render_task(tok, text, row["id"])
        p_ids, _ = _enc(tok, pre)
        w_ids, w_offs = _enc(tok, text, split_special=True)
        s_ids, _ = _enc(tok, suf)
        ids = p_ids + w_ids + s_ids
        win = np.zeros(len(ids), dtype=bool)
        win[len(p_ids):len(p_ids) + len(w_ids)] = True
        span = np.zeros(len(ids), dtype=bool)
        span[len(p_ids):len(p_ids) + len(w_ids)] = [(e > a and s < b) for s, e in w_offs]
        last = len(ids) - 1
    else:
        raise ValueError(f"framing must be one of {FRAMINGS}, got {framing!r}")
    if not span.any():                                       # truncated away (should not happen with max_len >= window)
        span[last] = True
    return {"ids": ids, "win": win, "span": span, "last": last}


# ----------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------
def load(repo: str, device: Optional[str] = None, dtype: Optional[str] = None):
    import torch
    from lgs4.backbone import load_backbone
    from lgs1.lgcore.model import _inner
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
          "float32": torch.float32}[dtype] if dtype else (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    model, tok = load_backbone(repo, device=device, dtype=dt, attn="sdpa")
    dec = _inner(model)
    n_layers = len(dec.layers)
    d = int(dec.config.hidden_size)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if tok.eos_token_id is not None else 0)
    sha = getattr(model.config, "_commit_hash", None) or ""
    return {"model": model, "dec": dec, "tok": tok, "n_layers": n_layers, "d": d, "pad": int(pad), "device": device,
            "dtype": str(dt).replace("torch.", ""), "sha": sha, "repo": repo}


def _batches(encs: Sequence[Dict], order: Sequence[int], budget: int) -> Iterator[List[int]]:
    cur: List[int] = []
    tmax = 0
    for i in order:
        t = len(encs[i]["ids"])
        if cur and max(tmax, t) * (len(cur) + 1) > budget:
            yield cur
            cur, tmax = [], 0
        cur.append(i)
        tmax = max(tmax, t)
    if cur:
        yield cur


def forward_pool(M: Dict, encs: Sequence[Dict], idx: Sequence[int], poolings: Sequence[str]) -> Dict[str, np.ndarray]:
    """{pool: float32 [len(idx), L+1, d]} for one padded batch."""
    import torch
    T = max(len(encs[i]["ids"]) for i in idx)
    B = len(idx)
    ids = torch.full((B, T), M["pad"], dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    masks = {p: torch.zeros((B, T), dtype=torch.float32) for p in ("mean", "span")}
    last = torch.zeros(B, dtype=torch.long)
    for j, i in enumerate(idx):
        e = encs[i]
        n = len(e["ids"])
        ids[j, :n] = torch.tensor(e["ids"])
        att[j, :n] = 1
        masks["mean"][j, :n] = torch.from_numpy(e["win"].astype(np.float32))
        masks["span"][j, :n] = torch.from_numpy(e["span"].astype(np.float32))
        last[j] = e["last"]
    dev = M["device"]
    with torch.no_grad():
        out = M["dec"](input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    res: Dict[str, np.ndarray] = {}
    for p in poolings:
        if p == "last":
            ar = torch.arange(B, device=dev)
            stack = torch.stack([h[ar, last.to(dev)].float() for h in hs], 1)
        else:
            m = masks[p].to(dev)
            den = m.sum(1).clamp_min(1.0)[:, None]
            stack = torch.stack([(h.float() * m[:, :, None]).sum(1) / den for h in hs], 1)
        res[p] = stack.cpu().numpy()
    del out, hs
    return res


def iter_pooled(M: Dict, rows: Sequence[Dict], framing: str, poolings: Sequence[str], max_len: int,
                budget: int, encs: Optional[List[Dict]] = None) -> Iterator[Tuple[List[int], Dict[str, np.ndarray]]]:
    """Yield (row indices, pooled features) over `rows`, length-sorted batches, halving the batch budget on OOM."""
    import torch
    encs = encs if encs is not None else [encode(M["tok"], r, framing, max_len) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(encs[i]["ids"]))
    pending = list(_batches(encs, order, budget))
    while pending:
        idx = pending.pop(0)
        try:
            yield idx, forward_pool(M, encs, idx, poolings)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                raise
            h = len(idx) // 2
            print(f"[extract] OOM at batch {len(idx)} x {max(len(encs[i]['ids']) for i in idx)} tokens -> split",
                  file=sys.stderr, flush=True)
            pending[:0] = [idx[:h], idx[h:]]


# ----------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------
def _versions() -> Dict[str, str]:
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__}


def extract(run: str, bb_key: str, framings: Sequence[str] = ("raw",), poolings: Sequence[str] = DEFAULT_POOLINGS,
            max_len: Optional[int] = None, device: Optional[str] = None, dtype: Optional[str] = None,
            budget: Optional[int] = None, P: Optional[AGPaths] = None, M: Optional[Dict] = None) -> Dict:
    from .windows import read_jsonl_gz
    P = P or AGPaths.load()
    R = P.run(run)
    card = json.loads(R.datacard.read_text())
    prof = card["recipe"]["profile"]
    max_len = max_len or int(prof["max_len"])
    rows = read_jsonl_gz(R.windows)
    bb = backbone(bb_key)
    M = M or load(bb.repo, device, dtype)
    budget = budget or bb.tokens_per_batch
    L1, d, N = M["n_layers"] + 1, M["d"], len(rows)
    summary = {}
    for framing in framings:
        D = R.feats(bb.key, framing)
        D.mkdir(parents=True, exist_ok=True)
        meta_p, prog_p = D / "meta.json", D / "progress.json"
        meta = {"backbone": bb.key, "repo": bb.repo, "model_sha": M["sha"], "framing": framing, "poolings": list(poolings),
                "n": N, "n_layers": M["n_layers"], "layers": L1, "d": d, "dtype_model": M["dtype"], "store": "float16",
                "layout": "[layer, window, d]", "max_len": max_len, "recipe_hash": card["recipe_hash"],
                "versions": _versions(), "complete": False}
        old = json.loads(meta_p.read_text()) if meta_p.exists() else None
        resume = bool(old) and all(old.get(k) == meta[k] for k in ("n", "layers", "d", "recipe_hash", "repo")) \
            and set(poolings) <= set(old.get("poolings", []))
        if resume and old.get("complete"):
            print(f"[extract] {bb.key}/{framing}: complete, skipping ({D})")
            summary[framing] = old
            continue
        done = set(json.loads(prog_p.read_text())["done"]) if resume and prog_p.exists() else set()
        if resume:
            meta["poolings"] = old["poolings"]
        mode = "r+" if resume else "w+"
        mms = {p: np.memmap(D / f"{p}.f16", dtype=np.float16, mode=mode, shape=(L1, N, d)) for p in meta["poolings"]}
        (D / "ids.json").write_text(json.dumps([r["id"] for r in rows]))
        meta_p.write_text(json.dumps(meta, indent=2))
        aux = {"n_tokens": np.zeros(N, np.int32), "n_window_tokens": np.zeros(N, np.int32),
               "n_span_tokens": np.zeros(N, np.int32)}
        if resume and (D / "aux.npz").exists():
            aux = dict(np.load(D / "aux.npz"))
        t0, n_clamped = time.time(), 0
        shards = [list(range(s, min(N, s + SHARD))) for s in range(0, N, SHARD)]
        for si, shard in enumerate(shards):
            if si in done:
                continue
            sub = [rows[i] for i in shard]
            encs = [encode(M["tok"], r, framing, max_len) for r in sub]
            for loc, feats in iter_pooled(M, sub, framing, meta["poolings"], max_len, budget, encs):
                gidx = np.asarray([shard[j] for j in loc])
                for p, x in feats.items():
                    big = np.abs(x) > FP16_MAX
                    if big.any():
                        n_clamped += int(big.sum())
                        x = np.clip(x, -FP16_MAX, FP16_MAX)
                    mms[p][:, gidx, :] = x.transpose(1, 0, 2).astype(np.float16)
            for i, e in zip(shard, encs):
                aux["n_tokens"][i], aux["n_window_tokens"][i], aux["n_span_tokens"][i] = len(e["ids"]), e["win"].sum(), e["span"].sum()
            for mm in mms.values():
                mm.flush()
            np.savez(D / "aux.npz", **aux)
            done.add(si)
            prog_p.write_text(json.dumps({"done": sorted(done), "n_shards": len(shards)}))
            rate = sum(len(shards[k]) for k in done) / max(1e-6, time.time() - t0)
            print(f"[extract] {bb.key}/{framing} shard {si + 1}/{len(shards)}  {rate:.1f} windows/s", flush=True)
        meta.update(complete=True, seconds=round(time.time() - t0, 1), n_clamped_fp16=n_clamped)
        meta_p.write_text(json.dumps(meta, indent=2))
        summary[framing] = meta
        print(f"[extract] {bb.key}/{framing}: {N:,} windows x {L1} layers x d={d} -> {D}")
    return summary


def drop_feats(run: str, bb_key: str, framings: Sequence[str], poolings: Sequence[str], P: Optional[AGPaths] = None) -> int:
    """Delete the feature memmaps once the probes and scores exist (8B, full profile: ~39 GB per framing x pooling)."""
    P = P or AGPaths.load()
    R = P.run(run)
    bb = backbone(bb_key)
    freed = 0
    for framing in framings:
        D = R.feats(bb.key, framing)
        for pool in poolings:
            f = D / f"{pool}.f16"
            if f.exists() and (R.probes(bb.key, framing) / f"{pool}.test_scores.npz").exists():
                freed += f.stat().st_size
                f.unlink()
        meta_p = D / "meta.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
            meta["complete"] = False
            meta["dropped"] = sorted(set(meta.get("dropped", [])) | set(poolings))
            meta_p.write_text(json.dumps(meta, indent=2))
            (D / "progress.json").unlink(missing_ok=True)
    print(f"[extract] dropped feature memmaps of {bb.key} ({freed / 2 ** 30:.1f} GiB freed)")
    return freed


def load_feats(R, bb: str, framing: str, pool: str) -> Tuple[np.memmap, List[str], Dict]:
    D = R.feats(bb, framing)
    meta = json.loads((D / "meta.json").read_text())
    if not meta.get("complete"):
        raise SystemExit(f"{D}: extraction incomplete (run extract again; it resumes)")
    if pool not in meta["poolings"]:
        raise SystemExit(f"{D}: pooling {pool!r} not extracted (have {meta['poolings']})")
    mm = np.memmap(D / f"{pool}.f16", dtype=np.float16, mode="r", shape=(meta["layers"], meta["n"], meta["d"]))
    return mm, json.loads((D / "ids.json").read_text()), meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="per-layer pooled features")
    ap.add_argument("--run", required=True)
    ap.add_argument("--backbone", required=True, help="key from agdepth.config.BACKBONES, a repo id or a local dir")
    ap.add_argument("--framings", default="raw")
    ap.add_argument("--poolings", default=",".join(DEFAULT_POOLINGS))
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, help="bf16 | fp16 | fp32 (default: bf16 on GPU, fp32 on CPU)")
    ap.add_argument("--tokens-per-batch", type=int, default=None)
    a = ap.parse_args(argv)
    fr = [f for f in a.framings.split(",") if f]
    po = [p for p in a.poolings.split(",") if p]
    bad = [f for f in fr if f not in FRAMINGS] + [p for p in po if p not in POOLINGS]
    if bad:
        raise SystemExit(f"unknown framing/pooling: {bad}")
    extract(a.run, a.backbone, fr, po, a.max_len, a.device, a.dtype, a.tokens_per_batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
