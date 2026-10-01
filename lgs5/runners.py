"""S5 row producers: trained LG (lg / sliding / oracle / needle-only), dense-FT and zero-shot base guards, all writing the
same eval-row schema with the E0-2 lineage.

Eval row keys (superset of lgs3.lgeval.ROW_KEYS):
    id split subset L_target depth label(0/1) method model score verdict n_tok
    hit selected needle_windows deep_probs sel_scores theta n_windows n_deep      (LG family)
    raw_label verdict_strict verdict_loose                                       (base guards)
    truncated ms_route ms_head ms_total needle_id pair_id haystack_domain
`deep_probs` = head probability of every routed window in router order (score desc), so recall/hit at any k' <= k is a
max over the first k' entries (lgs5.tables reads k=8/16 off the k=32 rows).
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from lgs1.lgcore.model import LGConfig, LandmarkGuard, head_forward, sliding_scan
from lgs1.lgcore.windows import windows_containing
from lgs1.lineage import Lineage, RowsWriter, canonical_hash, code_commit, dir_fingerprint, done_ids, read_first_lineage, sha256_file
from lgs2.synth import recipe_hash
from lgs2.tok import HFTok, token_span_for_chars
from lgs4 import backbone as BB
from lgs4.config import BACKBONES

from .config import ModelSpec, S5Paths
from .data import VARIANT_KEYS

HARNESS = "s5-eval-v1"
STREAM_MIN_TOK = 32768          # LG rows longer than this stream window chunks (S6 64k/128k): same verdict, flat memory
# S8 tail-pad arm: a neutral, content-free suffix appended (after a blank line) to every document before windowing, so the
# last window's probe reads the needle with content behind it even when the needle ends the document (depth 1.0).  The
# same suffix goes on harmful, twin and benign-long rows, so the calibration thresholds see it too.
TAIL_PAD_TEXT = ("This is the end of the document. No further content follows. The remaining lines are intentionally left "
                 "blank and carry no additional information. End of document.")


def _label01(r: Dict) -> int:
    lab = r.get("label")
    if isinstance(lab, (int, bool)):
        return int(lab)
    return 1 if str(lab) == "harmful" else 0


def _windows_overlapping(span: Tuple[int, int], windows: Sequence[Tuple[int, int]]) -> List[int]:
    a, b = span
    return [i for i, (s, e) in enumerate(windows) if s < b and e > a]


def lg_model_sha(run_dir: Path) -> str:
    """Fingerprint of the files an LG run actually loads (head.pt + router.pt) -- NOT the whole run directory: dense-FT
    writes dense.resume.pt / dense_ft.pt into the same directory, which changed the whole-dir fingerprint under a running
    e2 job and made the resume refuse the file (LineageMismatch on model_sha, 2026-09-21).  For a run dir that held only
    head.pt / router.pt when the rows were first written, this equals the old value."""
    return dir_fingerprint(run_dir, patterns=(BB.HEAD_FILE, BB.ROUTER_FILE))


def dense_model_sha(run_dir: Path) -> str:
    return dir_fingerprint(run_dir, patterns=(BB.DENSE_FILE,))


def lg_cfg_dict(cfg: LGConfig, method: str, k: int, tail_pad: int = 0) -> Dict:
    """The dict behind an LG rows file's cfg_hash.  The S8 options (LGConfig.router_ells, tail_pad) appear only when set,
    so the default configuration hashes exactly as it did when the S5/S6 rows were written; the stride override is inside
    cfg (and, like the tail pad, in the method name)."""
    d = dict(cfg.to_dict(), method=method, harness=HARNESS, k=int(k))
    if tail_pad:
        d["tail_pad"] = int(tail_pad)
    return d


def lg_cfg_dict_20260927(cfg: LGConfig, method: str, k: int, tail_pad: int = 0) -> Dict:
    """The same dict as the S8 code of 2026-09-27 14:46 hashed it: `router_ells` (None) and `tail_pad` (0) always present.
    Rows files written by that code (the S8 inference arms) carry this hash; `write_rows` keeps it when it resumes them."""
    return dict(lg_cfg_dict(cfg, method, k, tail_pad), router_ells=list(cfg.router_ells) if cfg.router_ells else None,
                tail_pad=int(tail_pad or 0))


class LGEval:
    """lg / sliding / oracle / needle modes on one trained run (runs/v2/<name>)."""

    def __init__(self, P: S5Paths, spec: ModelSpec, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 stream: str = "auto", chunk_windows: int = 64):
        self.spec = spec
        run_dir = P.run_dir(spec.name)
        if not (run_dir / BB.HEAD_FILE).exists():
            raise SystemExit(f"{run_dir / BB.HEAD_FILE} missing (S4 head)")
        hc = BB.load_ckpt(run_dir / BB.HEAD_FILE)
        bb = BACKBONES.get(hc["backbone"])
        repo = bb.repo if bb is not None else hc["repo"]
        model, _, head, router, cfg, info = BB.load_lg(run_dir, device=device, k=spec.k, dtype=dtype)
        if router is None and spec.kind in ("lg", "sliding", "needle"):
            raise SystemExit(f"{run_dir / BB.ROUTER_FILE} missing (S4 router)")
        if router is None:                                  # oracle needs no router; LandmarkGuard wants one
            from lgs1.lgcore.model import Router
            router = Router(BB.hidden_size(model)).to(device)
        if spec.stride:                                     # E6-2 inference arm: window stride override (cfg.b unchanged)
            cfg = dataclasses.replace(cfg, stride=int(spec.stride))
        self.lg = LandmarkGuard(model, cfg, router, head).to(device).eval()
        self.tok = HFTok(repo, bb.tok if bb is not None else "qwen3")
        self.device = device
        self.cfg = cfg
        self.method = spec.method
        self.run_dir = run_dir
        if stream not in ("auto", "on", "off"):
            raise ValueError("stream must be auto | on | off")
        self.stream = stream
        self.chunk_windows = int(chunk_windows)
        self.tail_pad = int(spec.tail_pad or 0)
        self._tail_ids: List[int] = []
        if self.tail_pad:
            base_ids, _ = self.tok.encode_with_offsets("\n\n" + TAIL_PAD_TEXT)
            reps = -(-self.tail_pad // max(1, len(base_ids)))
            self._tail_ids = (list(base_ids) * reps)[: self.tail_pad]
        # the streaming path returns the same verdict (tests/test_lgcore_stream.py), so it is not part of cfg_hash;
        # stride / tail pad change the rows and are (also through the method name)
        self.lineage = Lineage(model_sha=lg_model_sha(run_dir), router_sha=sha256_file(run_dir / BB.ROUTER_FILE) if (run_dir / BB.ROUTER_FILE).exists() else "none",
                               cfg_hash=canonical_hash(lg_cfg_dict(cfg, self.method, spec.k, self.tail_pad)), recipe_hash=recipe_hash(),
                               data_manifest_sha="none", code_commit=code_commit(Path(__file__).resolve().parents[1]))
        # rows files written by the S8 code of 2026-09-27 (before this hash form was restored) resume under their own hash
        self.lineage_aliases = [dataclasses.replace(self.lineage, cfg_hash=canonical_hash(lg_cfg_dict_20260927(cfg, self.method, spec.k, self.tail_pad)))]

    def _ids(self, text: str, spans: Sequence[Sequence[int]]):
        """Token ids + the token span of every character span (needle pieces / several needles: lgs3.variants).
        With tail padding the neutral suffix ids are appended (the spans are unaffected: they lie in the prefix)."""
        ids, offs = self.tok.encode_with_offsets(text)
        tspans = [token_span_for_chars(offs, (int(a), int(b))) for a, b in spans] if spans else []
        if self._tail_ids:
            ids = list(ids) + self._tail_ids
        return torch.tensor(ids, dtype=torch.long, device=self.device), tspans

    def _use_stream(self, n_tok: int) -> bool:
        return self.stream == "on" or (self.stream == "auto" and n_tok > STREAM_MIN_TOK)

    @staticmethod
    def _needle_windows(tspans, windows) -> List[int]:
        """Windows that wholly contain any needle span; when a span is longer than a window (long_needle, > 2b tokens),
        the windows overlapping it."""
        nw: List[int] = []
        for ts in tspans:
            ws = windows_containing(ts, windows) or _windows_overlapping(ts, windows)
            nw += [w for w in ws if w not in nw]
        return nw

    @torch.no_grad()
    def score_row(self, r: Dict) -> Dict:
        kind = self.spec.kind
        text = r["text"]
        spans = r.get("needle_char_spans") or ([r["needle_char_span"]] if r.get("needle_char_span") else [])
        if kind == "needle":                                  # the needle alone (E2-7 i): the first / only needle
            if spans:
                a, b = spans[0]
                text, spans = text[int(a):int(b)], []
        ids, tspans = self._ids(text, spans)
        tspan = tspans[0] if tspans else None
        t0 = time.perf_counter()
        out = {"n_tok": int(ids.numel()), "needle_tok_span": list(tspan) if tspan else None,
               "needle_tok_spans": [list(t) for t in tspans] if len(tspans) > 1 else None}
        if self.tail_pad:
            out["tail_pad"] = self.tail_pad
        if self.cfg.stride:
            out["stride"] = int(self.cfg.stride)
        if kind in ("lg", "needle"):
            o = self.lg(ids, stream=self._use_stream(int(ids.numel())), chunk_windows=self.chunk_windows)
            t1 = time.perf_counter()
            sel = [int(i) for i in o.selected.tolist()]
            nw = self._needle_windows(tspans, o.windows)
            out.update(score=o.score(), verdict=bool(o.verdict_unsafe), theta=float(o.theta), n_windows=len(o.windows),
                       n_deep=int(o.n_deep), selected=sel[:64], needle_windows=nw, hit=(any(w in sel for w in nw) if nw else None),
                       deep_probs=[round(float(p), 5) for p in o.head_probs.tolist()[:64]],
                       sel_scores=[round(float(s), 4) for s in o.window_scores[o.selected].tolist()[:64]])
        elif kind == "sliding":
            o = sliding_scan(self.lg, ids, chunk_windows=self.chunk_windows)
            t1 = time.perf_counter()
            sel = [int(i) for i in o.selected.tolist()]
            nw = self._needle_windows(tspans, o.windows)
            out.update(score=o.score(), verdict=bool(o.verdict_unsafe), theta=float(o.theta), n_windows=len(o.windows),
                       n_deep=int(o.n_deep), selected=sel[:64], needle_windows=nw, hit=(True if nw else None),
                       deep_probs=[round(float(p), 5) for p in o.head_probs.tolist()[:64]], sel_scores=None)
        elif kind == "oracle":                                # the head on the needle's own windows (E2-7 ii)
            from lgs1.lgcore.model import pool_scores
            from lgs1.lgcore.windows import make_windows
            windows = make_windows(int(ids.numel()), self.cfg.b, self.cfg.stride)
            nw = self._needle_windows(tspans, windows)
            if not nw:
                out.update(score=None, verdict=None, n_windows=len(windows), n_deep=0, selected=[], needle_windows=[],
                           hit=None, deep_probs=[], sel_scores=None, theta=None)
                t1 = time.perf_counter()
            else:
                # block-local trunk: the needle windows alone give exactly the states they would have in the full pass
                # (E0-1 c), so a 128k oracle row costs the same as a 1k one
                hid, hr, val = self.lg._trunk(ids, [windows[w] for w in nw])
                wscore = pool_scores(self.lg.router(hr), val, self.cfg.pool)
                logits = head_forward(self.lg.model, hid, val, self.cfg.ell, self.lg.head)
                probs = torch.sigmoid(logits)
                t1 = time.perf_counter()
                out.update(score=float(probs.max()), verdict=bool(probs.max() > self.cfg.tau), theta=None, n_windows=len(windows),
                           n_deep=len(nw), selected=nw, needle_windows=nw, hit=True, deep_probs=[round(float(p), 5) for p in probs.tolist()],
                           sel_scores=[round(float(x), 4) for x in wscore.tolist()])
        else:
            raise ValueError(kind)
        out.update(ms_route=None, ms_head=None, ms_total=round((t1 - t0) * 1000, 2))
        return out


class DenseEval:
    """dense-FT baseline: full-context forward -> last-token probe.  Beyond the window (`max_tokens`, default the model's
    32k) the spec policy applies: head = keep the first max_tokens (default), tail = keep the last, yarn = YaRN x4 rope
    (128k window, nothing truncated up to 128k).  The <=28k rows are identical under every policy.  Rows longer than
    `prefill_min` tokens are prefilled `prefill_chunk` tokens at a time through a KV cache (lgs4.backbone.dense_logits_chunked;
    2026-09-23: single-pass 128k YaRN rows OOMed on a 28 GB [L, L] mask) -- same last-token state, O(chunk x L) memory."""

    def __init__(self, P: S5Paths, spec: ModelSpec, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 max_tokens: Optional[int] = None, micro_tokens: int = 16384, prefill_min: int = BB.DENSE_PREFILL_MIN,
                 prefill_chunk: int = BB.DENSE_PREFILL_CHUNK):
        self.spec = spec
        run_dir = P.run_dir(spec.name)
        if not (run_dir / BB.DENSE_FILE).exists():
            raise SystemExit(f"{run_dir / BB.DENSE_FILE} missing (S4 dense)")
        self.policy = spec.policy or "head"
        self.model, self.tok, self.head, dc = BB.load_dense(run_dir, device=device, dtype=dtype, rope="yarn" if self.policy == "yarn" else None)
        self.device = device
        self.method = spec.method
        mpe = int(getattr(BB._inner(self.model).config, "max_position_embeddings", 32768) or 32768)
        self.max_tokens = int(max_tokens or min(mpe, 32768 * (4 if self.policy == "yarn" else 1)))
        self.micro_tokens = micro_tokens
        self.prefill_min, self.prefill_chunk = int(prefill_min), int(prefill_chunk)
        self.pad_id = BB.pad_id_of(self.model, self.tok)
        # the chunked prefill is a memory device that reproduces the single-pass state: not part of cfg_hash
        self.lineage = Lineage(model_sha=dense_model_sha(run_dir), router_sha="none",
                               cfg_hash=canonical_hash({"method": self.method, "max_tokens": self.max_tokens, "harness": HARNESS,
                                                        "train_max_tokens": dc.get("max_tokens"),
                                                        **({"policy": self.policy} if self.policy != "head" else {})}),
                               recipe_hash=recipe_hash(), data_manifest_sha="none", code_commit=code_commit(Path(__file__).resolve().parents[1]))

    def encode(self, text: str) -> Tuple[List[int], bool]:
        """Token ids after the context policy, and whether the document was truncated."""
        e = self.tok(text, add_special_tokens=False)["input_ids"]
        trunc = len(e) > self.max_tokens
        return (e[-self.max_tokens:] if self.policy == "tail" else e[:self.max_tokens]), trunc

    @torch.no_grad()
    def score_encoded(self, encs: List[Tuple[List[int], bool]]) -> List[Dict]:
        enc = [e for e, _ in encs]
        trunc = [tr for _, tr in encs]
        L = max(1, max(len(e) for e in enc))
        ids = torch.full((len(enc), L), self.pad_id, dtype=torch.long)
        valid = torch.zeros((len(enc), L), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            valid[i, :len(e)] = 1
        t0 = time.perf_counter()
        logits = BB.dense_logits_auto(self.model, self.head, ids.to(self.device), valid.to(self.device), self.prefill_min, self.prefill_chunk)
        ms = (time.perf_counter() - t0) * 1000 / len(enc)
        probs = torch.sigmoid(logits).tolist()
        chunked = len(enc) == 1 and len(enc[0]) > self.prefill_min
        return [{"score": float(p), "verdict": bool(p > 0.5), "n_tok": len(e), "truncated": tr, "ms_total": round(ms, 2),
                 **({"prefill_chunk": self.prefill_chunk} if chunked else {})}
                for p, e, tr in zip(probs, enc, trunc)]

    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> List[Dict]:
        return self.score_encoded([self.encode(t) for t in texts])

    def score_rows(self, rows: Iterable[Dict]) -> Iterable[Tuple[Dict, Dict]]:
        """Token-budget micro-batches on the REAL token counts (short prompts many at a time, long documents alone; the
        old chars/3 estimate under-counted CJK backgrounds 3-4x and could pad a long row into a batch)."""
        buf: List[Tuple[Dict, Tuple[List[int], bool]]] = []
        buf_tok = 0

        def flush():
            res = self.score_encoded([e for _, e in buf])
            for (r, _), s in zip(buf, res):
                yield r, s
            buf.clear()

        for r in rows:
            e = self.encode(r["text"])
            n = max(1, len(e[0]))
            if buf and (n > self.prefill_min or buf_tok > self.prefill_min or max(buf_tok, n) * (len(buf) + 1) > self.micro_tokens):
                yield from flush()
                buf_tok = 0
            buf.append((r, e))
            buf_tok = max(buf_tok, n)
            if n > self.prefill_min:                      # a chunked-prefill row is always its own batch
                yield from flush()
                buf_tok = 0
        if buf:
            yield from flush()


class BaseEval:
    """Zero-shot guard (lgs1.zeroshot.guards) on short prompts: raw verdict kept for strict/loose aggregation."""

    def __init__(self, spec: ModelSpec, dtype: str = "bfloat16", batch: int = 8):
        from lgs3.longbench import load_guard, parse_policy
        self.spec = spec
        self.gspec, self.tok, self.model, self.dev, self.snap = load_guard(spec.name, dtype, parse_policy("native"))
        self.method = spec.method
        self.batch = batch
        self.lineage = Lineage(model_sha=dir_fingerprint(self.snap) if self.snap else "none", router_sha="none",
                               cfg_hash=canonical_hash({"method": self.method, "harness": HARNESS, "guard": spec.name}),
                               recipe_hash=recipe_hash(), data_manifest_sha="none", code_commit=code_commit(Path(__file__).resolve().parents[1]))

    def score_rows(self, rows: Iterable[Dict]) -> Iterable[Tuple[Dict, Dict]]:
        from lgs3.longbench import score_texts
        buf: List[Dict] = []
        for r in rows:
            buf.append(r)
            if len(buf) >= self.batch:
                yield from self._flush(buf, score_texts)
                buf = []
        if buf:
            yield from self._flush(buf, score_texts)

    def _flush(self, buf: List[Dict], score_texts):
        t0 = time.perf_counter()
        res = score_texts(self.gspec, self.spec.name, self.tok, self.model, self.dev, [r["text"] for r in buf])
        ms = (time.perf_counter() - t0) * 1000 / len(buf)
        for r, x in zip(buf, res):
            lab = x.get("label")
            yield r, {"score": x.get("score"), "raw_label": lab, "verdict": (lab in ("unsafe", "controversial")) if lab else None,
                      "verdict_strict": (lab in ("unsafe", "controversial")) if lab else None,
                      "verdict_loose": (lab == "unsafe") if lab else None, "n_tok": None, "n_decode": x.get("n_decode", 0),
                      "raw": (x.get("raw") or "")[:120], "ms_total": round(ms, 2)}


def make_runner(P: S5Paths, spec: ModelSpec, device: str = "cuda", **kw):
    if spec.kind in ("lg", "sliding", "oracle", "needle"):
        return LGEval(P, spec, device=device, **{k: v for k, v in kw.items() if k in ("stream", "chunk_windows", "dtype")})
    if spec.kind == "dense":
        return DenseEval(P, spec, device=device, **{k: v for k, v in kw.items() if k in ("max_tokens", "micro_tokens", "dtype", "prefill_min", "prefill_chunk")})
    if spec.kind == "base":
        return BaseEval(spec)
    raise ValueError(spec.kind)


def adopt_lineage_alias(runner, out_path: Path, lin: Lineage) -> Lineage:
    """A rows file whose cfg_hash is one of the runner's `lineage_aliases` (an earlier encoding of the SAME configuration --
    the S8 inference arms of 2026-09-27) is resumed under that hash, so the file stays one lineage and the resume is not
    refused.  Only cfg_hash may differ; every other resume key still has to match (assert_resume_compatible)."""
    aliases = getattr(runner, "lineage_aliases", None) or []
    if not aliases:
        return lin
    first = read_first_lineage(out_path)
    if first is None or first.cfg_hash == lin.cfg_hash:
        return lin
    for alt in aliases:
        if alt.cfg_hash == first.cfg_hash:
            print(f"[s5] {out_path.name}: rows carry the 2026-09-27 cfg_hash of this configuration ({first.cfg_hash[:12]}…) -> resuming under it",
                  flush=True)
            return dataclasses.replace(lin, cfg_hash=alt.cfg_hash)
    return lin


def write_rows(runner, spec: ModelSpec, rows: Iterable[Dict], out_path: Path, manifest_sha: str = "none",
               resume: bool = True, log_every: int = 200) -> int:
    """Score `rows` with `runner` -> lineage-stamped eval rows, resumable by id."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lin = Lineage(**{**runner.lineage.__dict__, "data_manifest_sha": manifest_sha})
    done = done_ids(out_path) if resume and out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"[s5] {spec} -> {out_path.name}: done={len(done):,} todo={len(todo):,}", flush=True)
    if not todo:
        return 0
    if done:
        lin = adopt_lineage_alias(runner, out_path, lin)
    n = 0
    t0 = time.time()

    def base(r: Dict) -> Dict:
        d = {"id": r["id"], "split": r.get("split"), "subset": r.get("subset"), "L_target": r.get("L_target"), "depth": r.get("depth"),
             "label": _label01(r), "method": runner.method, "model": spec.name, "needle_id": r.get("needle_id"),
             "pair_id": r.get("pair_id"), "haystack_domain": r.get("haystack_domain")}
        if r.get("meta"):
            d["meta"] = {k: v for k, v in r["meta"].items() if k in ("adversarial", "category", "source_split")}
        for k in VARIANT_KEYS:                                   # dilution / robustness axes (S6 tables); absent on old rows
            if r.get(k) is not None and k != "needle_char_spans":
                d[k] = r[k]
        return d

    from lgs1.interrupt import Stop
    Stop.install()                                       # time-limit SIGTERM -> flush + rc=3 -> the sbatch script requeues itself
    with RowsWriter(out_path, lin, resume=out_path.exists()) as w:
        if hasattr(runner, "score_rows"):
            it = runner.score_rows(todo)
        else:
            it = ((r, runner.score_row(r)) for r in todo)
        for r, s in it:
            w.write({**base(r), **s})
            n += 1
            if n % log_every == 0:
                el = time.time() - t0
                print(f"   {n:,}/{len(todo):,}  {el:.0f}s  ({el / n * 1000:.0f} ms/row)", flush=True)
            Stop.check(f"{spec} {n:,}/{len(todo):,}")
    print(f"[s5] {spec}: {n:,} rows in {time.time() - t0:.0f}s -> {out_path}", flush=True)
    return n
