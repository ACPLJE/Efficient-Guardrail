"""D6 — synthesis of the long-context sets (report §5.4, §5.7; plan S2 D6 dry-run, S3 D6 full run).

    python -m lgs2.synth --dry-run                       # ~200 rows -> synth_dryrun/, schema checks, exit 0/1
    python -m lgs2.synth --split test --set test          # S3: full test main set + benign-long
    python -m lgs2.synth --split train --set train,train16k --native

Rules (all hashed into recipe_hash via config.RECIPE_V2)
  * length targets are Qwen3 tokens; a row is |n_tok - L_target| <= 10 %
  * the background is ONE domain's screened pieces concatenated with blank lines; every piece id is recorded;
    a piece belongs to exactly one split (splits/haystack.json)
  * the needle is inserted as its own paragraph at the paragraph boundary nearest to depth * len(background)
    (code: blank lines only); no paragraph is ever cut for the insertion
  * every harmful row has a benign TWIN: same background pieces, same depth, same L, benign needle within
    ±20 % tokens; rows are linked by twin_id / pair_id, so the label depends only on the needle
  * the same background is reused for every depth of a pair (controlled L x depth grid)
  * needle_char_span, needle_share = |needle| / n_tok, recipe_hash on every row
Outputs synth[_dryrun]/<split>/<subset>/rows.jsonl.gz and manifests/<split>.<subset>.manifest.json.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from lgs1.lineage import canonical_hash, code_commit, sha256_file
from lgs1.metrics import assert_unique_ids
from .config import NEEDLE_MAX_TOK, OOD_DOMAINS, RECIPE_V2, SYNTH_ROW_FIELDS, TRAIN_DOMAINS, S2Paths

POOL_FACTOR = 2.0        # a domain is used at length L only if its split pool has >= 2 x L tokens (see Pools.eligible)
from .haystack import load_docs
from .needles import load_pool
from .screen import load_keep
from .splits import load_splits
from .textutil import insert_at_depth
from .tok import get_tok

_SENT_SPLIT = __import__("re").compile(r"(?<=[.!?。！？])\s+|\n")


def recipe_hash(recipe: Optional[Dict] = None) -> str:
    return canonical_hash(recipe if recipe is not None else RECIPE_V2)


# ----------------------------------------------------------------------------------------------
# pools: needles by split/label, haystack pieces by split/domain
# ----------------------------------------------------------------------------------------------
class Pools:
    def __init__(self, P: S2Paths, splits: Sequence[str], allow_unscreened: bool = False, tok=None):
        self.P = P
        self.tok = tok
        pool = load_pool(P)
        nsplit, hsplit = load_splits(P)
        self.needle_split = nsplit["assign"]
        self.hay_split = hsplit["assign"]
        self.hay_provisional = bool(hsplit.get("provisional"))
        if self.hay_provisional and not allow_unscreened:
            raise SystemExit("haystack split is provisional (screening not summarised); pass --allow-unscreened for a dry-run only")
        self.needles: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        self.by_id: Dict[str, Dict] = {}
        for r in pool:
            sp = self.needle_split.get(r["id"])
            if sp in splits and r["usable"]:
                self.needles[sp][r["label"]].append(r)
                self.by_id[r["id"]] = r
        for sp in self.needles:
            for lab in self.needles[sp]:
                self.needles[sp][lab].sort(key=lambda r: r["id"])
        self._hay: Dict[Tuple[str, str], List[Dict]] = {}

    def haystack(self, split: str, domain: str) -> List[Dict]:
        key = (split, domain)
        if key not in self._hay:
            keep = load_keep(self.P, domain)
            docs = [d for d in load_docs(self.P, domain, keep=keep) if self.hay_split.get(d["id"]) == split]
            docs.sort(key=lambda d: d["id"])
            self._hay[key] = docs
        return self._hay[key]

    def main_needles(self, split: str, label: str) -> List[Dict]:
        return [r for r in self.needles[split][label] if r["n_tok_qwen3"] <= NEEDLE_MAX_TOK]

    def pool_tokens(self, split: str, domain: str) -> int:
        return sum(int(d.get("n_tok", 0)) for d in self.haystack(split, domain))

    def eligible(self, split: str, domains: Sequence[str], L: int, factor: float = POOL_FACTOR) -> List[str]:
        """Domains whose split pool holds >= factor x L tokens.  A background is built from consecutive pieces
        without repeating a doc, so a pool smaller than L cannot reach L at all, and a pool barely larger than L
        yields one near-identical background per row; 2x keeps some variety (agentdojo screened down to 29k tokens
        on 2026-09-18 -> usable up to L=8k only)."""
        return [d for d in domains if self.pool_tokens(split, d) >= factor * L]


def eligible_or_exit(pools: Pools, split: str, domains: Sequence[str], L: int, what: str, seen: set) -> List[str]:
    ok = pools.eligible(split, domains, L)
    dropped = [d for d in domains if d not in ok]
    for d in dropped:
        key = (split, what, d, L)
        if key not in seen:
            seen.add(key)
            print(f"[synth] {split}/{what} L={L}: {d} skipped (pool {pools.pool_tokens(split, d):,} tok < {POOL_FACTOR:g} x L)", flush=True)
    if not ok:
        raise SystemExit(f"{split}/{what} L={L}: no domain has >= {POOL_FACTOR:g} x L tokens ({', '.join(domains)})")
    return ok


# ----------------------------------------------------------------------------------------------
# sampling helpers
# ----------------------------------------------------------------------------------------------
def stratified_pairs(harmful: List[Dict], n_pairs: int, rng: random.Random) -> List[Dict]:
    """Round-robin over (bucket, source) groups after a seeded shuffle inside each group."""
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in harmful:
        groups[(r["bucket"], r["source"])].append(r)
    keys = sorted(groups)
    for k in keys:
        rng.shuffle(groups[k])
    out: List[Dict] = []
    while len(out) < n_pairs and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(out) < n_pairs:
                out.append(groups[k].pop())
    if len(out) < n_pairs:
        raise SystemExit(f"only {len(out)} usable harmful needles for {n_pairs} pairs")
    return out


def pick_twin(benign: List[Dict], n_tok: int, rng: random.Random, tol: float, cand: int = 64) -> Tuple[Dict, float]:
    if not benign:
        raise SystemExit("no benign needles in this split for twins")
    sample = [benign[rng.randrange(len(benign))] for _ in range(min(cand, len(benign)))]
    sample.sort(key=lambda r: abs(r["n_tok_qwen3"] - n_tok))
    best = sample[0]
    return best, best["n_tok_qwen3"] / max(1, n_tok)


def positions_for(spec, rng: random.Random) -> List[float]:
    """Depths for one (pair, L).  Random depths are rounded to 4 decimals and re-drawn when they collide with each
    other or with the ends (2026-09-19: 3 of 8,000 train pairs drew the same depth twice -> duplicate row ids;
    a re-draw only consumes extra randomness in that rare case, so every other pair's depths are unchanged)."""
    if isinstance(spec, list):
        return [float(x) for x in spec]
    if spec in ("random2+ends", "random2"):
        taken = {0.0, 1.0}
        out: List[float] = []
        while len(out) < 2:
            d = round(rng.random(), 4)
            if d not in taken:
                taken.add(d)
                out.append(d)
        return out + ([0.0, 1.0] if spec == "random2+ends" else [])
    raise ValueError(spec)


def _cut_to_fit(p: str, c: int, need_lo: int, need_hi: int, tok=None) -> Tuple[Optional[str], int]:
    """A prefix of paragraph `p` (exact token count `c`) holding need_lo..need_hi tokens, or (None, 0).

    Cut at sentence/line units first, then at word boundaries (a single long line of chat or minified code has
    no sentence units).  With a tokenizer the prefix is measured exactly and re-cut up to three times; a prefix is
    accepted only INSIDE the window (2026-09-18: accepting `est >= need_lo` let a 3,000-token first line through and
    produced 4x rows).  Without a tokenizer the count is proportional to `c` (write_rows recounts anyway)."""
    if need_hi <= 0 or need_lo > need_hi:
        return None, 0
    sep = "\n" if "\n" in p else " "
    levels = [([u for u in _SENT_SPLIT.split(p) if u.strip()], sep), (p.split(" "), " ")]
    if len(p) > 64:                                     # no spaces at all (minified code, base64): character-proportional cut
        step = max(1, len(p) // max(2, c))
        levels.append(([p[i:i + step] for i in range(0, len(p), step)], ""))
    for units, joiner in levels:
        if len(units) < 2:
            continue
        per = c / len(units)
        take = int(need_hi // per)
        if take < 1:
            continue
        take = min(take, len(units) - 1)
        piece = joiner.join(units[:take])
        est = int(per * take)
        if tok is not None:
            est = int(tok.count([piece])[0])
            for _ in range(3):
                if need_lo <= est <= need_hi:
                    break
                take = max(1, min(len(units) - 1, int(take * ((need_lo + need_hi) / 2) / max(1, est))))
                piece = joiner.join(units[:take])
                est = int(tok.count([piece])[0])
        if need_lo <= est <= need_hi:
            return piece, est
    return None, 0


def bg_bounds(L: int, extra: int, tol: float, n_items: int = 1, extra_max: Optional[int] = None) -> Tuple[int, int, int]:
    """(target, lo, hi) for the JOINED background when `extra` tokens of inserted text (needles, frames) will be added
    as `n_items` own paragraphs, so that the FINAL row lands inside L +- tol whatever the separators tokenise to
    (0..2 tokens per item).  Applying +-tol to (L - extra) instead compounds (the framed subset came out at 0.8999 L,
    2026-09-18); a margin of max(2, 0.2 % L) absorbs boundary-token merges.  `extra_max` (default = extra) is the
    longest of the alternative insertions sharing this background (a benign twin up to 20 % longer than the harmful
    needle): the window must hold the shortest AND the longest variant."""
    m = max(2, int(0.002 * L))
    extra_max = extra if extra_max is None else max(extra, extra_max)
    lo = int(math.ceil(L * (1 - tol))) - extra + m
    hi = int(math.floor(L * (1 + tol))) - extra_max - 2 * n_items - m
    target = L - (extra + extra_max) // 2 - n_items
    return target, lo, min(hi, max(lo, hi))


def length_window(L: int, tol: float) -> Tuple[int, int]:
    """Row-level acceptance window of gate 7 / D8: |n_tok - L| <= tol * L."""
    return int(math.ceil(L * (1 - tol))), int(math.floor(L * (1 + tol)))


_SEP_COST: Dict[Tuple[int, str], int] = {}


def sep_cost(tok, joiner: str = "\n\n") -> int:
    """Tokens one paragraph separator adds to a joined text (Qwen3: 1 for a blank line; whitespace tokenizer: 0).
    Measured once per tokenizer so the collection estimate counts separators (2026-09-18: 42/64,000 train rows and
    3/4,000 calib benign-long rows were still outside +-10 % — backgrounds of very short chat lines add one token per
    line, more than the exact trim could remove in its 6 rounds)."""
    if tok is None:
        return 0
    key = (id(tok), joiner)
    if key not in _SEP_COST:
        # upper bound over paragraph endings: a bare word ('from' + '\n\n' + 'you' = 3 tokens in Qwen3) and a full stop
        # ('.\n\n' merges into one token).  Over-estimating only makes the exact trim shorter; under-estimating would
        # make the bulk pop in _exact_fit remove too much.
        cost = 0
        for a, b in (("from", "you"), ("Alpha.", "Beta."), ("done!", "Next")):
            na, nb, nab = tok.count([a, b, a + joiner + b])
            cost = max(cost, int(nab) - int(na) - int(nb))
        _SEP_COST[key] = max(0, cost)
    return _SEP_COST[key]


def _exact_fit(pars: List[str], cnt: List[int], target: int, lo: int, hi: int, tok, joiner: str = "\n\n", rounds: int = 40) -> int:
    """Trim the END of the background until the JOINED text has <= hi tokens (exact count), aiming at `target`.
    Whole trailing paragraphs are dropped by estimate (paragraph count + separator) and the joined text re-measured,
    so any overshoot converges in a few rounds (the old 6 x one-paragraph rounds could not remove hundreds of
    separator tokens); the last kept paragraph is then cut to land near the target.  Returns the exact count."""
    if not pars:
        return 0
    sep = sep_cost(tok, joiner)
    n = int(tok.count([joiner.join(pars)])[0])
    for _ in range(rounds):
        if n <= hi or not pars:
            break
        over = n - target
        while pars and cnt[-1] + sep <= over + 2:
            over -= cnt[-1] + sep
            pars.pop()
            cnt.pop()
        if pars and over > 0:
            c_last = cnt[-1]
            slack = max(4, (hi - target) // 2)
            piece, est = _cut_to_fit(pars[-1], c_last, max(1, c_last - over - slack), c_last - over, tok)
            if piece is None:
                pars.pop()
                cnt.pop()
            else:
                pars[-1], cnt[-1] = piece, est
        n = int(tok.count([joiner.join(pars)])[0]) if pars else 0
    return n


class _Cursor:
    """Paragraphs of `docs` in order from a random start (wrapping), each doc at most once; `ids` = docs touched."""

    def __init__(self, docs: List[Dict], start: int):
        self.docs, self.i, self.j, self.n_docs = docs, start, 0, 0
        self.ids: List[str] = []

    def next(self) -> Optional[Tuple[str, int]]:
        while self.n_docs < len(self.docs):
            d = self.docs[self.i % len(self.docs)]
            if self.j == 0:
                self.ids.append(d["id"])
            if self.j < len(d["pars"]):
                p, c = d["pars"][self.j], d["par_tok"][self.j]
                self.j += 1
                return p, int(c)
            self.i, self.j, self.n_docs = self.i + 1, 0, self.n_docs + 1
        return None


def build_background(docs: List[Dict], target_tokens: int, rng: random.Random, tol: float, tok=None,
                     bounds: Optional[Tuple[int, int]] = None, info: Optional[Dict] = None,
                     starts: int = 4, rounds: int = 8) -> Tuple[List[str], List[int], List[str]]:
    """Paragraphs (+token counts, piece ids) of one domain reaching target_tokens within [lo, hi].

    lo/hi default to target +- tol; callers that insert material pass `bounds` from bg_bounds().  Whole pieces are
    taken consecutively from a random start until the ESTIMATED joined count (paragraph counts + one token per
    separator, an upper bound: '.\n\n' merges into one Qwen3 token, 'word\n\n' does not) reaches the target; with a
    tokenizer the JOINED text is then measured exactly and (a) trimmed into the window at its END (_exact_fit: whole
    paragraphs, then a sentence/line/word/character cut of the last one) or (b) extended by the exact shortfall with
    more whole pieces, re-measuring each round.  A background that still misses the window (pool exhausted) is
    rebuilt from another random start, up to `starts`; the best attempt is returned.  Only the exact count decides:
    2026-09-18 the estimate alone (stop at the first doc boundary past lo) left prose rows a paragraph count short.
    `info["n_exact"]` receives the exact joined count when a tokenizer is given."""
    if not docs:
        raise SystemExit("empty haystack for this split/domain")
    lo, hi = bounds if bounds else (target_tokens * (1 - tol), target_tokens * (1 + tol))
    lo, hi = int(lo), int(hi)
    sep = sep_cost(tok)
    best = None
    for _attempt in range(starts if tok is not None else 1):
        cur = _Cursor(docs, rng.randrange(len(docs)))
        pars: List[str] = []
        cnt: List[int] = []
        est = 0
        goal = target_tokens
        n = 0
        for _round in range(rounds):
            exhausted = False
            while est < goal:
                nxt = cur.next()
                if nxt is None:
                    exhausted = True
                    break
                p, c = nxt
                s_ = sep if pars else 0
                if tok is None and est + s_ + c > hi:                 # no tokenizer (tests): proportional cut at the end
                    piece, e = _cut_to_fit(p, c, int(max(1, lo - est - s_)), int(hi - est - s_), None)
                    if piece is not None and e > 0:
                        pars.append(piece)
                        cnt.append(e)
                        est += s_ + e
                        break
                    continue
                pars.append(p)
                cnt.append(c)
                est += s_ + c
            if tok is None:
                return pars, cnt, cur.ids
            if not pars:
                break
            n = int(tok.count(["\n\n".join(pars)])[0])
            if n > hi:
                n = _exact_fit(pars, cnt, target_tokens, lo, hi, tok)
                est = n
            if lo <= n <= hi:
                if info is not None:
                    info["n_exact"] = n
                return pars, cnt, cur.ids
            if exhausted or n > hi:
                break
            goal = est + (lo - n) + max(8, (target_tokens - lo) // 2)      # short: pull more, aim between lo and target
        if tok is None:
            return pars, cnt, cur.ids
        if best is None or abs(n - target_tokens) < abs(best[0] - target_tokens):
            best = (n, pars, cnt, cur.ids)
    n, pars, cnt, ids = best
    if info is not None:
        info["n_exact"] = n
    return pars, cnt, ids


def _row_id(split: str, subset: str, L: int, depth: Optional[float], pair: int, tag: str) -> str:
    d = "na" if depth is None else f"{depth:.4f}"
    return f"{split}-{subset}-L{L}-d{d}-p{pair}-{tag}"


# ----------------------------------------------------------------------------------------------
# generators
# ----------------------------------------------------------------------------------------------
def gen_main(pools: Pools, split: str, set_cfg: Dict, recipe: Dict, seed: int, domains: List[str],
             pairs_override: Optional[int] = None, L_override: Optional[List[int]] = None,
             pos_override=None) -> Iterable[Dict]:
    rng = random.Random(f"{seed}:{split}:{set_cfg['subset']}")
    rh = recipe_hash(recipe)
    subset = set_cfg["subset"]
    n_pairs = pairs_override or set_cfg["pairs"]
    Ls = L_override or set_cfg["L"]
    pos_spec = pos_override if pos_override is not None else set_cfg["positions"]
    harmful = stratified_pairs(pools.main_needles(split, "harmful"), n_pairs, rng)
    benign = pools.main_needles(split, "benign")
    tol = recipe["length_tol"]
    seen: set = set()
    elig = {L: eligible_or_exit(pools, split, domains, L, subset, seen) for L in Ls}
    for pi, h in enumerate(harmful):
        twin, ratio = pick_twin(benign, h["n_tok_qwen3"], rng, recipe["twin_len_tol"])
        n_h, n_t = h["n_tok_qwen3"], twin["n_tok_qwen3"]
        for L in Ls:
            dom = elig[L][pi % len(elig[L])]
            docs = pools.haystack(split, dom)
            bg_target, lo, hi = bg_bounds(L, min(n_h, n_t), tol, n_items=1, extra_max=max(n_h, n_t))
            depths = positions_for(pos_spec, rng)
            pars, cnt, ids, made, exact = build_rows_checked(docs, bg_target, lo, hi, rng, tol, pools.tok, L,
                                                             [(depth, tag, nd) for depth in depths for tag, nd in (("h", h), ("b", twin))])
            bg_tok = sum(cnt)
            pair_id = f"{split}-{subset}-L{L}-p{pi}"
            for (depth, tag, nd, text, span), n_ex in zip(made, exact):
                n_est = n_ex if n_ex is not None else bg_tok + nd["n_tok_qwen3"]
                rid = _row_id(split, subset, L, depth, pi, tag)
                yield {"id": rid, "split": split, "subset": subset, "L_target": L, "depth": depth,
                       "needle_id": nd["id"], "twin_id": _row_id(split, subset, L, depth, pi, "b" if tag == "h" else "h"),
                       "pair_id": pair_id, "haystack_domain": dom, "haystack_doc_ids": ids,
                       "needle_char_span": list(span), "label": nd["label"],
                       "needle_share": round(nd["n_tok_qwen3"] / max(1, n_est), 6), "n_tok_qwen3": n_est,
                       "n_tok_needle": nd["n_tok_qwen3"], "n_tok_estimated": n_ex is None, "twin_len_ratio": round(ratio, 3),
                       "needle_source": nd["source"], "needle_bucket": nd["bucket"], "recipe_hash": rh, "text": text}


def build_rows_checked(docs: List[Dict], bg_target: int, lo: int, hi: int, rng: random.Random, tol: float, tok, L: int,
                       items: List[Tuple[float, str, Dict]], attempts: int = 4):
    """One background + every (depth, tag, needle) insertion on it, with the FINAL rows measured exactly; a background
    whose rows do not all land inside L +- tol is rebuilt (new random start) up to `attempts` times, keeping the best.
    Returns (pars, cnt, ids, [(depth, tag, needle, text, span)], [exact count or None])."""
    lo_row, hi_row = length_window(L, tol)
    best = None
    for _ in range(attempts if tok is not None else 1):
        pars, cnt, ids = build_background(docs, bg_target, rng, tol, tok, bounds=(lo, hi))
        made = []
        for depth, tag, nd in items:
            text, span, _ = insert_at_depth(pars, nd["text"], depth)
            made.append((depth, tag, nd, text, span))
        if tok is None:
            return pars, cnt, ids, made, [None] * len(made)
        exact = [int(x) for x in tok.count([m[3] for m in made])]
        n_bad = sum(1 for n in exact if not lo_row <= n <= hi_row)
        if best is None or n_bad < best[0]:
            best = (n_bad, pars, cnt, ids, made, exact)
        if n_bad == 0:
            break
        shift = L - sorted(exact)[len(exact) // 2]          # systematic miss: recentre the window on the measured rows
        bg_target, lo, hi = bg_target + shift, lo + shift, hi + shift
    return best[1:]


def gen_benign_long(pools: Pools, split: str, per_L: int, Ls: List[int], recipe: Dict, seed: int, domains: List[str],
                    subset: str = "benign_long") -> Iterable[Dict]:
    rng = random.Random(f"{seed}:{split}:{subset}")
    rh = recipe_hash(recipe)
    tol = recipe["length_tol"]
    seen: set = set()
    for L in Ls:
        doms = eligible_or_exit(pools, split, domains, L, subset, seen)
        for j in range(per_L):
            dom = doms[j % len(doms)]
            docs = pools.haystack(split, dom)
            info: Dict = {}
            pars, cnt, ids = build_background(docs, L, rng, tol, pools.tok, bounds=bg_bounds(L, 0, tol, n_items=0)[1:], info=info)
            text = "\n\n".join(pars)
            n_ex = info.get("n_exact")
            yield {"id": f"{split}-{subset}-L{L}-{dom}-{j}", "split": split, "subset": subset, "L_target": L, "depth": None,
                   "needle_id": None, "twin_id": None, "pair_id": None, "haystack_domain": dom, "haystack_doc_ids": ids,
                   "needle_char_span": None, "label": "benign", "needle_share": 0.0,
                   "n_tok_qwen3": n_ex if n_ex is not None else sum(cnt),
                   "n_tok_needle": 0, "n_tok_estimated": n_ex is None, "recipe_hash": rh, "text": text}


def gen_native(pools: Pools, split: str, recipe: Dict, limit: Optional[int] = None, subset: str = "native") -> Iterable[Dict]:
    rh = recipe_hash(recipe)
    n = 0
    for lab in ("harmful", "benign"):
        for r in pools.needles[split][lab]:
            yield {"id": f"{split}-{subset}-{r['id']}", "split": split, "subset": subset, "L_target": None, "depth": None,
                   "needle_id": r["id"], "twin_id": None, "pair_id": None, "haystack_domain": None, "haystack_doc_ids": [],
                   "needle_char_span": [0, len(r["text"])], "label": r["label"], "needle_share": 1.0,
                   "n_tok_qwen3": r["n_tok_qwen3"], "n_tok_needle": r["n_tok_qwen3"], "n_tok_estimated": False,
                   "needle_source": r["source"], "recipe_hash": rh, "text": r["text"]}
            n += 1
            if limit and n >= limit:
                return


# ----------------------------------------------------------------------------------------------
# writing with exact token counts + manifest
# ----------------------------------------------------------------------------------------------
def write_rows(rows: Iterable[Dict], out: Path, tok=None, batch: int = 32, progress_every: int = 2000,
               length_tol: float = RECIPE_V2["length_tol"]) -> Dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    counts: Counter = Counter()
    n_len_bad = 0
    len_bad_examples: List[Dict] = []
    t0 = time.time()
    with gzip.open(out, "wt", encoding="utf-8") as f:
        buf: List[Dict] = []

        def flush():
            nonlocal n, n_len_bad
            if tok is not None:
                todo = [r for r in buf if r.get("n_tok_estimated", True)]     # generators that measured the final text skip this
                exact = tok.count([r["text"] for r in todo]) if todo else []
                for r, e in zip(todo, exact):
                    r["n_tok_qwen3"] = int(e)
                    r["n_tok_estimated"] = False
                    if r.get("n_tok_needle"):
                        r["needle_share"] = round(r["n_tok_needle"] / max(1, int(e)), 6)
            for r in buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                counts[(r["subset"], r["L_target"], r["label"])] += 1
                n += 1
                if tok is not None and r.get("L_target") and abs(r["n_tok_qwen3"] - r["L_target"]) > length_tol * r["L_target"]:
                    n_len_bad += 1
                    if len(len_bad_examples) < 10:
                        len_bad_examples.append({"id": r["id"], "L_target": r["L_target"], "n_tok": r["n_tok_qwen3"],
                                                 "domain": r.get("haystack_domain")})
                if n % progress_every == 0:
                    print(f"   {out.parent.parent.name}/{out.parent.name}: {n:,} rows  {time.time() - t0:.0f}s  "
                          f"(last L={r['L_target']})", flush=True)
            buf.clear()

        for r in rows:
            buf.append(r)
            if len(buf) >= batch:
                flush()
        if buf:
            flush()
    if n_len_bad:
        print(f"[synth] LENGTH CHECK: {n_len_bad}/{n} rows of {out.parent.parent.name}/{out.parent.name} outside ±{length_tol:.0%} of L_target, "
              f"e.g. {len_bad_examples[:3]}", flush=True)
    return {"path": str(out), "rows": n, "seconds": round(time.time() - t0, 1), "n_len_bad": n_len_bad, "len_bad_examples": len_bad_examples,
            "counts": {f"{s}/L{L}/{lab}": c for (s, L, lab), c in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1] or 0, x[0][2]))}}


def write_manifest(P: S2Paths, split: str, subset: str, rows_path: Path, info: Dict, recipe: Dict, dry_run: bool) -> Path:
    m = {"split": split, "subset": subset, "path": str(rows_path), "sha256": sha256_file(rows_path), "rows": info["rows"],
         "recipe_hash": recipe_hash(recipe),
         "recipe_version": recipe.get("version") or "+".join(str(v.get("version", "?")) for v in recipe.values() if isinstance(v, dict)),
         "code_commit": code_commit(Path(__file__).resolve().parents[1]), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "counts": info["counts"], "dry_run": dry_run}
    mp = (P.manifests if not dry_run else P.synth_dryrun / "manifests") / f"{split}.{subset}.manifest.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return mp


# ----------------------------------------------------------------------------------------------
# validation (schema + rules) — used by the dry-run and by the E0-4 gate in S3
# ----------------------------------------------------------------------------------------------
def validate_rows(rows: List[Dict], hay_split: Optional[Dict[str, str]] = None, by_id: Optional[Dict[str, Dict]] = None,
                  length_tol: float = RECIPE_V2["length_tol"], twin_tol: float = RECIPE_V2["twin_len_tol"]) -> Dict:
    problems: List[str] = []
    ids = {r["id"]: r for r in rows}
    missing = Counter()
    for r in rows:
        for k in SYNTH_ROW_FIELDS:
            if k not in r:
                missing[k] += 1
    if missing:
        problems.append(f"missing fields: {dict(missing)}")
    try:
        assert_unique_ids(rows, keys=("split", "subset", "id"))
    except AssertionError as e:
        problems.append(f"duplicate (split, subset, id): {e}")
    n_len_bad = n_span_bad = n_twin_bad = n_needle_long = n_hay_bad = n_twin_len = 0
    len_bad_examples: List[Dict] = []
    for r in rows:
        if r["needle_char_span"] and by_id and r["needle_id"] in by_id:
            a, b = r["needle_char_span"]
            if r["text"][a:b] != by_id[r["needle_id"]]["text"]:
                n_span_bad += 1
        if r["L_target"] and not r.get("n_tok_estimated") and abs(r["n_tok_qwen3"] - r["L_target"]) > length_tol * r["L_target"]:
            n_len_bad += 1
            if len(len_bad_examples) < 12:
                len_bad_examples.append({"id": r["id"], "domain": r.get("haystack_domain"), "L_target": r["L_target"],
                                         "n_tok": r["n_tok_qwen3"]})
        if r["subset"] in ("main", "main16k") and r["n_tok_needle"] > NEEDLE_MAX_TOK:
            n_needle_long += 1
        if r["label"] == "harmful" and r["subset"] in ("main", "main16k"):
            t = ids.get(r["twin_id"])
            if not t or t["label"] != "benign" or t["haystack_doc_ids"] != r["haystack_doc_ids"] or t["depth"] != r["depth"] \
                    or t["L_target"] != r["L_target"] or t["twin_id"] != r["id"]:
                n_twin_bad += 1
            elif abs(t["n_tok_needle"] - r["n_tok_needle"]) > twin_tol * max(1, r["n_tok_needle"]):
                n_twin_len += 1
        if hay_split:
            for d in r["haystack_doc_ids"]:
                if hay_split.get(d) != r["split"]:
                    n_hay_bad += 1
                    break
    if n_span_bad:
        problems.append(f"{n_span_bad} rows: needle_char_span does not reproduce the needle text")
    if n_len_bad:
        problems.append(f"{n_len_bad} rows: |n_tok - L_target| > {length_tol:.0%}")
    if n_needle_long:
        problems.append(f"{n_needle_long} main rows: needle > {NEEDLE_MAX_TOK} tokens")
    if n_twin_bad:
        problems.append(f"{n_twin_bad} harmful rows: twin missing / wrong background / depth / L / back-link")
    if n_hay_bad:
        problems.append(f"{n_hay_bad} rows: a background piece belongs to another split")
    # label balance per (split, subset main, L)
    bal = Counter((r["split"], r["subset"], r["L_target"], r["label"]) for r in rows if r["subset"] in ("main", "main16k"))
    for (sp, su, L, lab), n in list(bal.items()):
        other = bal.get((sp, su, L, "benign" if lab == "harmful" else "harmful"), 0)
        if other != n:
            problems.append(f"label imbalance {sp}/{su}/L{L}: harmful vs benign {n} vs {other}")
            break
    return {"n_rows": len(rows), "pass": not problems, "problems": problems, "n_twin_len_outside_tol": n_twin_len,
            "n_span_bad": n_span_bad, "n_len_bad": n_len_bad, "len_bad_examples": len_bad_examples}


def read_rows(path: Path) -> List[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ----------------------------------------------------------------------------------------------
# dry-run (~200 rows) and full runs
# ----------------------------------------------------------------------------------------------
DRYRUN_PLAN = {
    "test": {"pairs": 10, "L": [1024, 4096, 28672], "positions": [0.0, 0.5, 1.0], "benign_long_per_L": 2},
    "train": {"pairs": 8, "L": [1024, 4096], "positions": "random2", "benign_long_per_L": 0},
    "native": 10,
}


def run_dry(P: S2Paths, tok, seed: int, allow_unscreened: bool, recipe: Optional[Dict] = None) -> Dict:
    recipe = recipe if recipe is not None else RECIPE_V2          # looked up at call time (tests patch the module global)
    pools = Pools(P, ("train", "val", "calib", "test"), allow_unscreened=allow_unscreened, tok=tok)
    out_root = P.synth_dryrun
    results: Dict[str, Dict] = {}
    all_rows: List[Dict] = []
    for split, plan in (("test", DRYRUN_PLAN["test"]), ("train", DRYRUN_PLAN["train"])):
        set_cfg = recipe["sets"][split]
        doms = recipe["domains_per_split"][split]
        rows = list(gen_main(pools, split, set_cfg, recipe, seed, doms, pairs_override=plan["pairs"],
                             L_override=plan["L"], pos_override=plan["positions"]))
        info = write_rows(rows, out_root / split / "main" / "rows.jsonl.gz", tok)
        write_manifest(P, split, "main", out_root / split / "main" / "rows.jsonl.gz", info, recipe, dry_run=True)
        results[f"{split}/main"] = info
        all_rows += read_rows(out_root / split / "main" / "rows.jsonl.gz")
        if plan["benign_long_per_L"]:
            bl = list(gen_benign_long(pools, split, plan["benign_long_per_L"], plan["L"], recipe, seed,
                                      recipe["benign_long_domains"][split] if split in recipe["benign_long_domains"] else doms))
            info = write_rows(bl, out_root / split / "benign_long" / "rows.jsonl.gz", tok)
            write_manifest(P, split, "benign_long", out_root / split / "benign_long" / "rows.jsonl.gz", info, recipe, dry_run=True)
            results[f"{split}/benign_long"] = info
            all_rows += read_rows(out_root / split / "benign_long" / "rows.jsonl.gz")
    nat = list(gen_native(pools, "train", recipe, limit=DRYRUN_PLAN["native"]))
    info = write_rows(nat, out_root / "train" / "native" / "rows.jsonl.gz", tok)
    write_manifest(P, "train", "native", out_root / "train" / "native" / "rows.jsonl.gz", info, recipe, dry_run=True)
    results["train/native"] = info
    all_rows += read_rows(out_root / "train" / "native" / "rows.jsonl.gz")
    v = validate_rows(all_rows, pools.hay_split, pools.by_id, recipe["length_tol"], recipe["twin_len_tol"])
    report = {"dry_run": True, "seed": seed, "tokenizer": getattr(tok, "name", None), "sets": results, "validation": v,
              "haystack_provisional": pools.hay_provisional, "recipe_hash": recipe_hash(recipe),
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_root / "dryrun_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_full(P: S2Paths, tok, seed: int, split: str, sets: List[str], native: bool, recipe: Optional[Dict] = None,
             pairs_override: Optional[int] = None) -> Dict:
    recipe = recipe if recipe is not None else RECIPE_V2
    pools = Pools(P, (split,), allow_unscreened=False, tok=tok)
    out: Dict[str, Dict] = {}
    for s in sets:
        cfg = recipe["sets"][s]
        doms = recipe["domains_per_split"][split]
        rows = gen_main(pools, split, cfg, recipe, seed, doms, pairs_override=pairs_override)
        path = P.synth / split / cfg["subset"] / "rows.jsonl.gz"
        info = write_rows(rows, path, tok)
        manifest_if_clean(P, split, cfg["subset"], path, info, recipe)
        out[f"{split}/{cfg['subset']}"] = info
        if cfg.get("benign_long"):
            bl = gen_benign_long(pools, split, cfg["benign_long"], cfg["L"], recipe, seed, recipe["benign_long_domains"][split])
            path = P.synth / split / "benign_long" / "rows.jsonl.gz"
            info = write_rows(bl, path, tok)
            manifest_if_clean(P, split, "benign_long", path, info, recipe)
            out[f"{split}/benign_long"] = info
    if native:
        path = P.synth / split / "native" / "rows.jsonl.gz"
        info = write_rows(gen_native(pools, split, recipe), path, tok)
        manifest_if_clean(P, split, "native", path, info, recipe)
        out[f"{split}/native"] = info
    return out


def manifest_if_clean(P: S2Paths, split: str, subset: str, path: Path, info: Dict, recipe: Dict, dry_run: bool = False) -> Optional[Path]:
    """Write the manifest only when every row is inside the length window; otherwise remove a stale one so that
    `s3ctl.py status` shows the set as PENDING (2026-09-18: a failed set with a manifest looked DONE)."""
    if info.get("n_len_bad") and not dry_run:
        stale = P.manifests / f"{split}.{subset}.manifest.json"
        if stale.exists():
            stale.unlink()
        print(f"[synth] {split}/{subset}: manifest withheld — {info['n_len_bad']} rows outside the length window "
              f"(rows file kept for inspection: {path})", flush=True)
        return None
    return write_manifest(P, split, subset, path, info, recipe, dry_run=dry_run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D6 synthesis")
    ap.add_argument("--dry-run", action="store_true", help="~200 rows into synth_dryrun/ + schema checks")
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", default="", help="comma list of recipe sets (train,train16k,val,calib,test)")
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--pairs", type=int, default=None, help="override the number of pairs (smoke)")
    ap.add_argument("--tok", default="qwen3", help="qwen3 | whitespace (tests only)")
    ap.add_argument("--seed", type=int, default=RECIPE_V2.get("seed", 20260917))
    ap.add_argument("--allow-unscreened", action="store_true", help="dry-run before screening finished (provisional haystack)")
    a = ap.parse_args(argv)
    P = S2Paths.load().ensure()
    tok = get_tok(a.tok)
    if a.dry_run:
        rep = run_dry(P, tok, a.seed, a.allow_unscreened)
        for k, v in rep["sets"].items():
            print(f"[synth] {k:18s} rows={v['rows']:4d}  {v['seconds']}s")
        v = rep["validation"]
        print(f"[synth] validation: {'PASS' if v['pass'] else 'FAIL'}  rows={v['n_rows']}  twin_len_outside_tol={v['n_twin_len_outside_tol']}")
        for p in v["problems"]:
            print(f"   - {p}")
        for ex in v.get("len_bad_examples", []):
            print(f"       {ex['id']}  {ex['domain']}  L={ex['L_target']}  n_tok={ex['n_tok']}")
        if rep["haystack_provisional"]:
            print("[synth] NOTE: haystack split is provisional (unscreened) — dry-run only; rerun after `screen --summarize` + `splits`")
        print(f"[written] {P.synth_dryrun}/dryrun_report.json")
        return 0 if v["pass"] else 1
    sets = [s for s in a.set.split(",") if s] or [a.split]
    out = run_full(P, tok, a.seed, a.split, sets, a.native, pairs_override=a.pairs)
    bad = 0
    for k, v in out.items():
        print(f"[synth] {k:18s} rows={v['rows']:6d}  {v['seconds']}s  len_bad={v.get('n_len_bad', 0)}  -> {v['path']}")
        bad += v.get("n_len_bad", 0)
    if bad:
        print(f"[synth] FAIL: {bad} rows outside ±{RECIPE_V2['length_tol']:.0%} of L_target (gate 7 would reject the split) -> rc=2",
              file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
