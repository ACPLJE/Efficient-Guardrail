"""Window dataset: background span + one inserted snippet, label = snippet class.

Construction (identical for every class, so the only systematic difference between classes is the snippet):
  1. snippets outside the length band `profile.snip_tok` (reference-tokenizer tokens) are dropped; mined paragraphs are
     first trimmed at sentence boundaries;
  2. splits: snippet groups and background documents are hashed into train/val/test (SPLIT_FRACTIONS); mined paragraphs
     keep the split of their document; split_mode=source sends holdout sources (LG public test sets, LLMail-Inject) to
     test only;
  3. per class and split, snippets are drawn stratified by length bin so every class follows the same length
     histogram (the intersection of the two positive classes), sources weighted equally within a class, each snippet
     used at most `max_reuse` times;
  4. a contiguous background span of (window - snippet) tokens is cut from a random document of the same split; the
     snippet goes in as its own paragraph at a relative depth ~ U(DEPTH_RANGE), snapped to the nearest line or
     sentence boundary.

    python -m agdepth.windows --run NAME --profile quick [--split-mode group|source] [--tok qwen3]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (BG_DOMAINS, BIGNEG_CLASSES, CLASSES, DEPTH_RANGE, PROFILES, SEED, SPLIT_FRACTIONS, AGPaths,
                     Profile, recipe, recipe_hash)
from . import sources as S
from . import synth

SPLITS = ("train", "val", "test")
BIN = 8
NEAR = 1                        # length bins (x 8 tokens) within which a short bin is filled before looking further


def _u(key: str, seed: int = SEED) -> float:
    return int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)


def split_of(key: str, fractions: Dict[str, float] = SPLIT_FRACTIONS, seed: int = SEED) -> str:
    x, acc = _u(key, seed), 0.0
    for name in SPLITS:
        acc += fractions.get(name, 0.0)
        if x < acc:
            return name
    return SPLITS[-1]


def assign_splits(by_cls: Dict[str, List[Dict]], split_mode: str, seed: int = SEED) -> None:
    """Split per GROUP key, consistently across classes (the LG pool's near-duplicate groups can hold a harmful and a
    benign prompt).  split_mode=source: a group with any holdout row goes to test as a whole, and the other groups of a
    class that has holdout rows fill train/val only.  test_only sources (LLMail-Inject) go to test in every mode.
    Mined paragraphs keep their source document's split."""
    items_all = [it for items in by_cls.values() for it in items]
    preset = {it["group"] for it in items_all if it.get("split") in SPLITS}  # mined paragraphs: document split
    src_hold = {it["group"] for it in items_all if it["holdout"]} if split_mode == "source" else set()
    test_only = {it["group"] for it in items_all if it.get("test_only")}    # e.g. LLMail-Inject (plan 5.2)
    hold = src_hold | test_only
    tv = {"train": SPLIT_FRACTIONS["train"], "val": SPLIT_FRACTIONS["val"]}
    tv = {k: v / sum(tv.values()) for k, v in tv.items()}
    for cls, items in by_cls.items():
        # only a SOURCE-mode holdout turns the rest of the class into train/val; a test_only source (LLMail) is an
        # extra held-out test set and leaves the other sources' 0.6/0.1/0.3 split alone
        cls_hold = any(it["group"] in src_hold for it in items)
        for it in items:
            if it.get("split") in SPLITS:
                continue
            if it["group"] in hold:
                it["split"] = "test"
            else:
                it["split"] = split_of(it["group"], tv if cls_hold else SPLIT_FRACTIONS, seed)
    owners: Dict[str, set] = defaultdict(set)
    for cls, items in by_cls.items():
        for it in items:
            owners[it["group"]].add(cls)
    frozen = {g for g, cs in owners.items() if len(cs) > 1} | hold | preset
    for items in by_cls.values():
        _fill_empty_splits(items, seed, frozen)


def _fill_empty_splits(items: List[Dict], seed: int, frozen: Optional[set] = None) -> None:
    """A class with >= 3 groups gets at least one group in every split: an empty split takes the group of the largest
    split whose hash lies closest to the empty split's interval (deterministic).  Groups in `frozen` (shared with
    another class, holdout, or mined from a document whose background pieces are already split) never move."""
    frozen = frozen or set()
    groups: Dict[str, str] = {}
    for it in items:
        groups.setdefault(it["group"], it["split"])
    if len(groups) < 3:
        return
    lo = {"train": 0.0, "val": SPLIT_FRACTIONS["train"], "test": SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]}
    for s in SPLITS:
        if s in groups.values():
            continue
        donors = Counter(sp for g, sp in groups.items() if g not in frozen)
        if not donors:
            return
        donor = donors.most_common(1)[0][0]
        cands = [g for g, sp in groups.items() if sp == donor and g not in frozen]
        if len(cands) < 2:
            continue
        g = min(cands, key=lambda g: abs(_u(g, seed) - lo[s]))
        groups[g] = s
        for it in items:
            if it["group"] == g:
                it["split"] = s


# ----------------------------------------------------------------------------------------------
# lengths
# ----------------------------------------------------------------------------------------------
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def trim_to_tokens(text: str, tok, hi: int) -> str:
    """Longest sentence prefix with <= hi tokens ('' when even the first sentence is too long)."""
    sents = _SENT_END.split(text)
    out = ""
    for s in sents:
        cand = (out + " " + s).strip() if out else s
        if tok.count([cand])[0] > hi:
            break
        out = cand
    return out


VARIANT_SOURCES = ("gen_instr", "mined_instr", "bg_plain")      # benign: a sentence prefix keeps the label


def _keeps_label(it: Dict, text: str) -> bool:
    """A shortened benign snippet keeps its class only if it still meets that class's rule: mined paragraphs are
    re-checked with the mining rule, template text must keep at least one imperative sentence."""
    if it["source"] == "mined_instr":
        return S.is_instruction_heavy(text)
    if it["source"] == "bg_plain":
        return S.is_plain(text)
    if it["source"] == "gen_instr":
        return S.imperative_count(text)[0] >= 1
    return True


def half_variants(items: List[Dict]) -> List[Dict]:
    """First half of the sentences of benign multi-sentence snippets (same group, so splits stay clean), kept only when
    the half still meets the class rule.  Gives the instruction-heavy and plain classes coverage of the short length bins
    the positive classes occupy."""
    out = []
    for it in items:
        if it["source"] not in VARIANT_SOURCES:
            continue
        sents = _SENT_END.split(it["text"])
        if len(sents) < 3:
            continue
        half = " ".join(sents[:math.ceil(len(sents) / 2)])
        if _keeps_label(it, half):
            out.append({**it, "sid": it["sid"] + "#half", "text": half, "meta": {**it.get("meta", {}), "variant": "half"}})
    return out


def length_filter(by_cls: Dict[str, List[Dict]], tok, band: Tuple[int, int]) -> Dict[str, Dict]:
    """Trim long mined paragraphs (re-checking their class rule), add half variants, drop exact duplicates created by
    either step (the full snippet wins), then keep snippets inside the band."""
    lo, hi = band
    stats = {}
    for cls in list(by_cls):
        items = by_cls[cls]
        n0 = len(items)
        trimmed = []
        for it in items:
            if it["source"] in ("mined_instr", "bg_plain") and tok.count([it["text"]])[0] > hi:
                t = trim_to_tokens(it["text"], tok, hi)
                if not t or not _keeps_label(it, t):
                    continue
                it = {**it, "text": t}
            trimmed.append(it)
        seen, uniq = set(), []
        for it in trimmed + half_variants(trimmed):
            k = S.exact_key(it["text"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(it)
        counts = tok.count([it["text"] for it in uniq]) if uniq else []
        keep = []
        for it, n in zip(uniq, counts):
            if lo <= n <= hi:
                it["n_tok"] = int(n)
                keep.append(it)
        by_cls[cls] = keep
        stats[cls] = {"n_in": n0, "n_after_trim_variants_dedup": len(uniq), "n_kept": len(keep)}
    return stats


def target_hist(by_cls: Dict[str, List[Dict]], band: Tuple[int, int], pos=("harm", "inj")) -> Dict[int, float]:
    """Length-bin distribution every class is sampled to: the intersection of the positive classes' histograms."""
    bins = range(band[0] // BIN, band[1] // BIN + 1)
    hs = []
    for c in pos:
        cnt = Counter(it["n_tok"] // BIN for it in by_cls.get(c, []))
        tot = sum(cnt.values())
        if tot:
            hs.append({b: cnt[b] / tot for b in bins})
    if not hs:
        return {b: 1.0 / len(bins) for b in bins}
    h = {b: min(x[b] for x in hs) for b in bins}
    if sum(h.values()) < 0.2:                       # nearly disjoint: fall back to the mean histogram
        h = {b: sum(x[b] for x in hs) / len(hs) for b in bins}
    s = sum(h.values())
    return {b: v / s for b, v in h.items() if v > 0}


def _int_targets(hist: Dict[int, float], n: int) -> Dict[int, int]:
    """Integer bin targets summing to n (largest remainder)."""
    raw = {b: n * p for b, p in hist.items()}
    out = {b: int(math.floor(v)) for b, v in raw.items()}
    rest = n - sum(out.values())
    for b in sorted(raw, key=lambda b: (-(raw[b] - out[b]), b))[:max(0, rest)]:
        out[b] += 1
    return out


def stratified_draw(items: List[Dict], hist: Dict[int, float], n: int, max_reuse: int, rng: random.Random) -> List[Dict]:
    """n draws following `hist` over length bins, sources weighted equally within a bin.

    A bin short of snippets hands its deficit to nearby bins, ring by ring: within +-NEAR bins it takes unused distinct
    snippets, then reuse (<= max_reuse each); only then it walks further out, distinct before reuse at each distance.
    Length matching therefore beats diversity beyond one bin (8 tokens).  Returns fewer than n only when even reuse runs
    out.  Source balancing uses Efraimidis-Spirakis keys in log space (log(u) * n_source), which cannot underflow."""
    if not items or n <= 0:
        return []
    src_n = Counter(it["source"] for it in items)
    by_bin: Dict[int, List[Dict]] = defaultdict(list)
    for it in items:
        by_bin[it["n_tok"] // BIN].append(it)
    for b, lst in by_bin.items():
        keyed = [(math.log(1.0 - rng.random()) * src_n[it["source"]], it) for it in lst]
        keyed.sort(key=lambda x: -x[0])
        by_bin[b] = [it for _, it in keyed]
    uniq = {b: len(v) for b, v in by_bin.items()}
    alloc = {b: 0 for b in by_bin}
    deficit: Dict[int, int] = {}
    for b, t in _int_targets(hist, n).items():
        take = min(t, uniq.get(b, 0))
        if b in alloc:
            alloc[b] += take
        if t > take:
            deficit[b] = t - take

    def spend(b0: int, need: int, cap, d_lo: int, d_hi: int) -> int:
        for b in sorted(by_bin, key=lambda x: (abs(x - b0), x)):
            if need <= 0 or abs(b - b0) > d_hi:
                break
            if abs(b - b0) < d_lo:
                continue
            k = min(cap(b) - alloc[b], need)
            if k > 0:
                alloc[b] += k
                need -= k
        return need
    distinct, reuse = (lambda x: uniq[x]), (lambda x: uniq[x] * max_reuse)
    lo, hi = min([*by_bin, *deficit]), max([*by_bin, *deficit])
    far = hi - lo                                               # every bin is within reach of every deficit bin
    for b in sorted(deficit):                                   # the tolerance ring first: distinct, then reuse
        deficit[b] = spend(b, deficit[b], distinct, 0, NEAR)
        deficit[b] = spend(b, deficit[b], reuse, 0, NEAR)
    for d in range(NEAR + 1, far + 1):                          # then outward, distinct before reuse at each distance
        for b in sorted(deficit):
            deficit[b] = spend(b, deficit[b], distinct, d, d)
            deficit[b] = spend(b, deficit[b], reuse, d, d)
    out: List[Dict] = []
    for b, k in alloc.items():
        lst = by_bin[b]
        out += [lst[j % len(lst)] for j in range(k)]
    rng.shuffle(out)
    return out[:n]


# ----------------------------------------------------------------------------------------------
# insertion
# ----------------------------------------------------------------------------------------------
def _snap(text: str, k: int, radius: int = 120) -> int:
    """Cut index near k: prefer a newline, then a sentence end, then whitespace."""
    lo, hi = max(0, k - radius), min(len(text), k + radius)
    best = None
    for pat in ("\n", ". ", " "):
        idx = [m.start() + len(pat) for m in re.finditer(re.escape(pat), text[lo:hi])]
        if idx:
            best = min((lo + i for i in idx), key=lambda x: abs(x - k))
            break
    return k if best is None else best


class Backgrounds:
    def __init__(self, docs: Sequence[Dict], tok, rng: random.Random):
        self.docs, self.tok, self.rng = list(docs), tok, rng

    def span(self, n_tokens: int, tries: int = 20) -> Optional[Tuple[str, Dict]]:
        """A contiguous background text of ~n_tokens tokens (cut at token boundaries) and the document it came from."""
        for _ in range(tries):
            d = self.rng.choice(self.docs)
            text = d["text"]
            n_chars = (n_tokens + 32) * 7
            if len(text) < n_chars // 2:
                continue
            a = self.rng.randrange(0, max(1, len(text) - n_chars))
            m = re.search(r"\s", text[a:a + 200])
            a = a + (m.end() if m else 0)
            chunk = text[a:a + n_chars].lstrip()
            if n_tokens <= 0:
                return "", d
            ids, offs = self.tok.encode_with_offsets(chunk)
            if len(ids) < n_tokens:
                continue
            return chunk[:offs[n_tokens - 1][1]], d
        return None


def make_window(snip: Dict, bg: Backgrounds, window: int, rng: random.Random, sep: str = "\n\n") -> Optional[Dict]:
    r = max(0, window - snip["n_tok"] - 2)
    got = bg.span(r)
    if got is None:
        return None
    btxt, doc = got
    depth = rng.uniform(*DEPTH_RANGE)
    k = _snap(btxt, int(round(depth * len(btxt)))) if btxt else 0
    if btxt and not btxt[:k].strip():           # never let the snippet open the window (position 0 = attention sink)
        k = _snap(btxt, int(round(DEPTH_RANGE[0] * len(btxt))) + 40)
    pre, suf = btxt[:k].rstrip(), btxt[k:].lstrip()
    head = pre + sep if pre else ""
    text = head + snip["text"] + (sep + suf if suf else "")
    a = len(head)
    return {"text": text, "snip_span": [a, a + len(snip["text"])], "bg_doc": doc["id"],
            "bg_src": doc.get("doc", doc["id"]), "bg_domain": doc["domain"],
            "depth": round(k / max(1, len(btxt)), 4) if btxt else 0.0}


# ----------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------
def _write_jsonl_gz(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl_gz(path: Path) -> List[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def audit(rows: Sequence[Dict]) -> Dict:
    """Leakage checks across splits: snippet ids, groups, exact snippet texts, background pieces and the source
    documents they were cut from (pieces of one book never straddle splits)."""
    sets = {k: defaultdict(set) for k in ("sid", "group", "bg_doc", "bg_src", "snip_key")}
    for r in rows:
        sets["sid"][r["sid"]].add(r["split"])
        sets["group"][r["group"]].add(r["split"])
        sets["bg_doc"][r["bg_doc"]].add(r["split"])
        sets["bg_src"][r.get("bg_src", r["bg_doc"])].add(r["split"])
        a, b = r["snip_span"]
        sets["snip_key"][S.exact_key(r["text"][a:b])].add(r["split"])
    out = {f"n_{k}_in_several_splits": sum(1 for v in d.values() if len(v) > 1) for k, d in sets.items()}
    out["pass"] = all(v == 0 for v in out.values())
    return out


HASH_FIELDS = ("id", "split", "cls", "source", "group", "text", "snip_span")


def content_sha(rows: Sequence[Dict]) -> str:
    """Content hash of the windows (not of the gzip bytes, whose header embeds a timestamp)."""
    from lgs1.lineage import canonical_hash
    return canonical_hash([{k: r.get(k) for k in HASH_FIELDS} for r in rows])[:16]


def data_manifest_sha(card: Dict) -> str:
    """The dataset identity every derived file records: windows + big pool (+ recipe)."""
    return card["data_manifest_sha"]


def _len_stats(xs: Sequence[int]) -> Dict:
    if not xs:
        return {}
    a = np.asarray(xs)
    return {"n": int(a.size), "mean": round(float(a.mean()), 1), "p10": int(np.percentile(a, 10)),
            "p50": int(np.percentile(a, 50)), "p90": int(np.percentile(a, 90))}


def build(run: str, profile: Profile, split_mode: str = "group", tok_name: str = "qwen3",
          bg_domains: Sequence[str] = BG_DOMAINS, smoke: bool = False, seed: int = SEED,
          sources_only: Optional[Sequence[str]] = None, P: Optional[AGPaths] = None) -> Dict:
    from lgs2.tok import get_tok
    from ag.harness.writer import atomic_write_json
    from . import lineage as LN
    P = P or AGPaths.load()
    R = P.run(run)
    t0 = time.time()
    # an interrupted rebuild must not leave the old data_manifest_sha next to new windows: drop the card first
    R.datacard.unlink(missing_ok=True)
    (R.data / "lineage.json").unlink(missing_ok=True)
    tok = get_tok("whitespace" if smoke else tok_name)
    rng = random.Random(seed)
    if smoke:
        docs, where = synth.smoke_background(seed=seed), {"smoke": "agdepth.synth.smoke_background"}
    else:
        docs, where = S.load_backgrounds(P, bg_domains, seed=seed)
    if not docs:
        raise SystemExit(f"[build] no background documents for {list(bg_domains)}: build the LG v2 haystack "
                         f"(python s2ctl.py ...) or download the streams (bash slurm/s1_download_login.sh --only pg19,arxiv,...)")
    doc_split = {d.get("doc", d["id"]): split_of("doc:" + d.get("doc", d["id"]), seed=seed) for d in docs}
    by_cls, src_report = S.collect_snippets(P, docs, doc_split, smoke=smoke, seed=seed, only=sources_only)
    missing = [c for c in CLASSES if not by_cls.get(c)]
    for c in missing:
        print(f"[build] WARNING class {c!r} has no snippets ({CLASSES[c]})", file=sys.stderr)
    if any(c in missing for c in ("harm", "inj")):
        raise SystemExit("[build] both positive classes (harm, inj) are required; see the source report above")
    len_stats = length_filter(by_cls, tok, profile.snip_tok)
    empty = [c for c in ("harm", "inj") if not by_cls.get(c)]
    if empty:
        raise SystemExit(f"[build] no {empty} snippet survives the length band {profile.snip_tok} (tokens of {tok_name}): "
                         f"{ {c: len_stats.get(c) for c in empty} } -- widen it with --snip-tok LO,HI")
    assign_splits(by_cls, split_mode, seed)
    hist = target_hist(by_cls, profile.snip_tok)
    bgs = {s: Backgrounds([d for d in docs if doc_split[d.get("doc", d["id"])] == s], tok, random.Random(f"{seed}-{s}"))
           for s in SPLITS}
    for s, b in bgs.items():
        if not b.docs:
            raise SystemExit(f"[build] no background documents in split {s} (too few documents: {len(docs)})")
    rows, shortfall = [], {}
    n_by_split = {"train": profile.n_train, "val": profile.n_val, "test": profile.n_test}
    for cls in sorted(by_cls):
        for split in SPLITS:
            pool_all = [it for it in by_cls[cls] if it["split"] == split]
            # group mode: a test_only source (LLMail) is drawn on its own, to the same length histogram, so the
            # in-distribution windows the gate uses keep the full n_test and their length matching (the held-out set
            # is extra rows).  Source mode: the held-out sources ARE the gate's test positives, one pool as before.
            parts = [("", pool_all)]
            if split == "test" and split_mode == "group" and any(it.get("test_only") for it in pool_all):
                parts = [("", [it for it in pool_all if not it.get("test_only")]),
                         ("ho-", [it for it in pool_all if it.get("test_only")])]
            for tag, pool in parts:                             # an empty pool draws nothing and is a shortfall
                key = f"{cls}/{split}" + ("/heldout" if tag else "")
                draws = stratified_draw(pool, hist, n_by_split[split], profile.max_reuse,
                                        random.Random(f"{seed}-{cls}-{split}{tag}"))
                if len(draws) < n_by_split[split]:
                    shortfall[key] = {"want": n_by_split[split], "got": len(draws), "snippets": len(pool)}
                for j, sn in enumerate(draws):
                    w = make_window(sn, bgs[split], profile.window, rng)
                    if w is None:
                        shortfall.setdefault(key, {"want": n_by_split[split], "got": len(draws),
                                                   "snippets": len(pool)})["no_background"] = True
                        continue
                    rows.append({"id": f"{split}-{cls}-{tag}{j:06d}", "split": split, "cls": cls, "source": sn["source"],
                                 "sid": sn["sid"], "group": sn["group"], "holdout": sn["holdout"],
                                 "test_only": bool(sn.get("test_only")), "n_tok_snip": sn["n_tok"], **w})
    counts = tok.count([r["text"] for r in rows])
    for r, n in zip(rows, counts):
        r["n_tok"] = int(n)
    _write_jsonl_gz(R.windows, rows)
    big = []
    if profile.n_bigneg:
        for cls in BIGNEG_CLASSES:
            pool = [it for it in by_cls.get(cls, []) if it["split"] == "test"]
            if not pool:
                continue
            brng = random.Random(f"{seed}-big-{cls}")
            draws = stratified_draw(pool, hist, profile.n_bigneg, max(1, math.ceil(profile.n_bigneg / len(pool)) + 1), brng)
            for j, sn in enumerate(draws):
                w = make_window(sn, bgs["test"], profile.window, brng)
                if w is not None:
                    big.append({"id": f"bigneg-{cls}-{j:07d}", "split": "bigneg", "cls": cls, "source": sn["source"],
                                "sid": sn["sid"], "group": sn["group"], "holdout": sn["holdout"],
                                "n_tok_snip": sn["n_tok"], **w})
        _write_jsonl_gz(R.bigneg, big)
    else:
        R.bigneg.unlink(missing_ok=True)          # never leave a big pool from an earlier recipe behind
    found = sorted({d["domain"] for d in docs})
    used_sources = sorted(k for k, v in src_report.items() if v.get("status") in ("ok", "smoke"))
    rec = recipe(profile, split_mode, found, tok_name if not smoke else "whitespace", seed, used_sources)
    w_sha, b_sha = content_sha(rows), content_sha(big)
    from lgs1.lineage import canonical_hash
    dsha = canonical_hash({"windows": w_sha, "bigneg": b_sha, "recipe": recipe_hash(rec)})[:16]
    ho_apart = split_mode == "group"                   # test_only rows are drawn and reported apart (see above)
    card = {
        "run": run, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
        "recipe": rec, "recipe_hash": recipe_hash(rec), "smoke": smoke,
        "windows_sha": w_sha, "bigneg_sha": b_sha, "data_manifest_sha": dsha,
        "backgrounds": where, "n_background_docs": {s: len(b.docs) for s, b in bgs.items()},
        "sources": src_report, "snippets": S.class_summary(by_cls), "length_filter": len_stats,
        "length_hist_target": {str(b * BIN): round(p, 4) for b, p in sorted(hist.items())},
        "windows": {s: dict(Counter(r["cls"] for r in rows if r["split"] == s)) for s in SPLITS},
        "windows_by_source": {c: dict(Counter(r["source"] for r in rows if r["cls"] == c)) for c in sorted(by_cls)},
        # the gate side only: in group mode the test_only rows are a separate held-out contrast
        "snippet_tokens": {c: {s: _len_stats([r["n_tok_snip"] for r in rows
                                              if r["cls"] == c and r["split"] == s and not (ho_apart and r.get("test_only"))])
                               for s in SPLITS} for c in sorted(by_cls)},
        "snippet_tokens_heldout": {c: _len_stats([r["n_tok_snip"] for r in rows if r["cls"] == c and r.get("test_only")])
                                   for c in sorted(by_cls)
                                   if ho_apart and any(r["cls"] == c and r.get("test_only") for r in rows)},
        "window_tokens": _len_stats([r["n_tok"] for r in rows]),
        "bigneg": dict(Counter(r["cls"] for r in big)), "shortfall": shortfall, "audit": audit(rows),
    }
    means = {c: v["test"].get("mean") for c, v in card["snippet_tokens"].items() if v.get("test")}
    ref = [means[c] for c in ("harm", "inj") if means.get(c)]
    if ref:
        m0 = sum(ref) / len(ref)
        off = {c: m for c, m in means.items() if m and abs(m - m0) > 0.2 * m0}
        if off:
            card["length_warning"] = (f"snippet length differs by > 20 % from the positive classes' mean {m0:.1f} for "
                                      f"{off}; compare the `span` pooling (no dilution) before reading a depth gap")
            print(f"[build] WARNING {card['length_warning']}", file=sys.stderr)
    R.data.mkdir(parents=True, exist_ok=True)
    exp_id = exp_of(profile)
    chash = LN.config_hash(rec)
    rid = LN.make_run_id(exp_id, "build", "none", f"depthwin-{profile.window}", chash, seed)
    lin = LN.start(R.data, rid, exp_id, chash, seed=seed, producer="mock" if smoke else "reference", model_sha={},
                   data_manifest_sha=dsha, fresh=True)
    card["run_id"] = rid
    atomic_write_json(R.datacard, card)
    LN.finish(R.data, lin)
    print(f"[build] {len(rows):,} windows (+{len(big):,} bigneg) -> {R.windows}")
    for s in SPLITS:
        print(f"[build]   {s:5s} " + "  ".join(f"{c}={n}" for c, n in sorted(card['windows'][s].items())))
    print(f"[build] audit: {card['audit']}")
    if shortfall:
        print(f"[build] shortfall (not enough distinct snippets): {shortfall}", file=sys.stderr)
    return card


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build the window dataset")
    ap.add_argument("--run", required=True)
    ap.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    ap.add_argument("--split-mode", default="group", choices=["group", "source"])
    ap.add_argument("--tok", default="qwen3", help="reference tokenizer for lengths (qwen3 | llama3 | repo id)")
    ap.add_argument("--bg-domains", default=",".join(BG_DOMAINS))
    ap.add_argument("--sources", default="", help="comma list of snippet sources (default: all available)")
    ap.add_argument("--smoke", action="store_true", help="placeholder data, whitespace tokenizer (offline plumbing test)")
    ap.add_argument("--seed", type=int, default=SEED)
    add_profile_overrides(ap)
    a = ap.parse_args(argv)
    build(a.run, profile_from_args(a), a.split_mode, a.tok, [d for d in a.bg_domains.split(",") if d], a.smoke, a.seed,
          [s for s in a.sources.split(",") if s] or None)
    return 0


def exp_of(profile: Profile) -> str:
    """E0.3 for smoke/quick, E1.1 for full (an overridden profile keeps its base name before '+')."""
    from .config import EXP_OF_PROFILE
    return EXP_OF_PROFILE.get(profile.name.split("+")[0], "E0.3")


def load_card(R) -> Dict:
    if not R.datacard.exists():
        raise SystemExit(f"{R.datacard} missing: run build first")
    return json.loads(R.datacard.read_text())


def add_profile_overrides(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--window", type=int, default=None, help="override the profile's window length (tokens)")
    ap.add_argument("--snip-tok", default=None, help="override the snippet length band, e.g. 12,112")
    for k in ("n_train", "n_val", "n_test", "n_bigneg", "max_len"):
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=None, help=f"override profile.{k}")


def profile_from_args(a) -> Profile:
    """The named profile with any command-line overrides (recorded in the recipe, hence in recipe_hash)."""
    from dataclasses import replace
    prof = PROFILES[a.profile]
    kw = {k: getattr(a, k) for k in ("window", "n_train", "n_val", "n_test", "n_bigneg", "max_len")
          if getattr(a, k, None) is not None}
    if getattr(a, "snip_tok", None):
        lo, hi = (int(x) for x in a.snip_tok.split(","))
        kw["snip_tok"] = (lo, hi)
    if kw:
        prof = replace(prof, name=prof.name + "+custom", **kw)
        if prof.window < prof.snip_tok[1] + 32 or prof.max_len < prof.window + 16:
            raise SystemExit(f"[build] inconsistent profile {prof}: need window >= snip_hi + 32 and max_len >= window + 16")
    return prof


if __name__ == "__main__":
    sys.exit(main())
