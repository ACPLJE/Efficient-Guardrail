"""Chunked prefill for the causal zero-shot guards at 64k/128k (plan S6 B1: "8B 128k는 chunked prefill", S7 A100-40GB).

A 128k prompt through `model.generate` prefills all positions in one forward: the attention workspace and the transient
activations of that single pass are what OOM on a 48 GB card, not the KV cache (Qwen3-8B: ~150 KB / token -> 19 GB at
128k).  Here the prompt minus its last token is pushed through the model `chunk` tokens at a time into a DynamicCache,
and `generate` then continues from that cache (the HF prefix-caching pattern), so the peak is one chunk's activations
plus the cache.  Greedy decoding on the full cache is mathematically the same computation as the single-pass prefill.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def _cache(model):
    from transformers import DynamicCache
    try:
        return DynamicCache(config=model.config)          # transformers >= 4.56 wants the config for sliding-window layers
    except TypeError:
        return DynamicCache()


def generate_with_prefill(model, tok, input_ids, chunk: int, gen_kw: Dict):
    """input_ids [1, L] on the model's device -> the `generate` output (sequences [+ scores]) after a chunked prefill."""
    import torch
    L = int(input_ids.shape[1])
    cache = _cache(model)
    with torch.no_grad():
        s = 0
        while s < L - 1:                                   # keep the last prompt token for generate (>= 1 uncached token)
            e = min(L - 1, s + int(chunk))
            out = model(input_ids=input_ids[:, s:e], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            s = e
        return model.generate(input_ids=input_ids, past_key_values=cache, **gen_kw)


def prefill_generate_texts(spec, tok, model, dev, prompts: List[str], chunk: int, label_ids: Optional[List[int]] = None,
                           flagged_idx: Optional[List[int]] = None, first_token: Optional[List[str]] = None) -> List[Dict]:
    """One prompt at a time (batch 1, no padding): [{raw, score, n_decode}] with the same score semantics as
    lgs2.screen.score_batch (label_ids given: softmax over the label tokens at the first label step, flagged mass) or
    lgs1.zeroshot.run.generate_batch (first_token given: P(unsafe) / (P(unsafe) + P(safe)) at step 0)."""
    import torch
    out: List[Dict] = []
    want_scores = bool(label_ids) or bool(first_token)
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = ids.to(getattr(model, "device", dev))
        gen_kw = dict(max_new_tokens=spec.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        if want_scores:
            gen_kw.update(output_scores=True, return_dict_in_generate=True)
        g = generate_with_prefill(model, tok, ids, chunk, gen_kw)
        seqs = g.sequences if hasattr(g, "sequences") else g
        new = seqs[:, ids.shape[1]:]
        raw = tok.batch_decode(new, skip_special_tokens=False)[0]
        score = None
        scores = getattr(g, "scores", None)
        if scores:
            if label_ids:
                for step in range(min(len(scores), new.shape[1])):
                    if int(new[0, step]) in label_ids:
                        probs = torch.softmax(scores[step][0].float()[label_ids], dim=-1)
                        score = float(sum(float(probs[j]) for j in (flagged_idx or [0])))
                        break
            elif first_token:
                probs = torch.softmax(scores[0].float(), dim=-1)[0]
                u = tok.encode(first_token[0], add_special_tokens=False)[0]
                s_ = tok.encode(first_token[1], add_special_tokens=False)[0]
                pu, ps = float(probs[u]), float(probs[s_])
                score = pu / max(pu + ps, 1e-9)
        n_new = int((new[0] != tok.pad_token_id).sum()) if tok.pad_token_id is not None else int(new.shape[1])
        out.append({"raw": raw, "score": score, "n_decode": n_new})
        del g, seqs
    return out
