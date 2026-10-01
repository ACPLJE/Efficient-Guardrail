"""E4 cost measurement (plan S7; report E4-1..4, E4-9): one process = one method on one GPU.

    python -m lgs7.cost --exp latency    --method lg:qwen3guard_0.6b_s0 --L all                 # p50/p99 ms vs L, batch 1 (E4-1)
    python -m lgs7.cost --exp latency    --method lg:qwen3guard_0.6b_s0:k32:noskip --L all      # cost-model switches: k<K>, noskip, pad, nostream
    python -m lgs7.cost --exp throughput --method lg:qwen3guard_0.6b_s0 --L 8192,32768,131072 --batches 1,2,4,8,16,32   # docs/s, tokens/s (E4-2)
    python -m lgs7.cost --exp memory     --method dense:qwen3guard_0.6b_s0:yarn --L all         # peak allocated / reserved vs L (E4-3)
    python -m lgs7.cost --exp breakdown  --method lg:qwen3guard_0.6b_s0 --L all                 # tokenize / trunk / router / head / aggregate ms (E4-4)
    python -m lgs7.cost --exp ksweep     --method lg:qwen3guard_0.6b_s0 --L 28672,131072 --ks 4,8,16,32,64,all   # (E4-9)
    python -m lgs7.cost --exp latency    --method base:qwen3guard_gen_8b:yarn --L all           # zero-shot dense guard (native | yarn | trunc_keep_head)
    python -m lgs7.cost --exp latency    --method cd:qwen3guard_gen_8b:512:256 --L all          # chunk-and-decode, chunk batch 8
    python -m lgs7.cost --exp latency    --method sliding:qwen3guard_0.6b_s0 --L all            # B5: the LG head on every window

Every row = one (exp, method, L, batch, k) measurement with p50 / p90 / p99 / mean ms over `--iters` timed runs after
`--warmup` runs (CUDA events + synchronize; distinct documents of the requested L from test/benign_long, random tokens
when the data is absent), peak memory (torch.cuda.max_memory_allocated / _reserved, reset before every measurement,
baseline = resting allocation after the model load), the GPU / driver / CUDA / torch / transformers / dtype / attention
implementation / batch / repeat metadata (plan S7: every measured row records them), and the lineage of the run.
Rows -> $LG_OUT/cost/<gpu slug>/<exp>__<method tag>.rows.jsonl (append; a key already present is skipped unless --force).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _one_gpu() -> None:
    """E4 rows are single-GPU measurements.  `--exclusive` nodes expose every GPU of the node, and transformers'
    device_map='auto' then balances a guard across all of them (9/24: base 0.6B yarn failed with cuda:0 vs cuda:7, the
    8B / 20B guards ran pipelined over 8 GPUs) -> keep only the first visible device, before CUDA is initialised.
    LG_COST_ALL_GPUS=1 opts out."""
    if os.environ.get("LG_COST_ALL_GPUS") == "1":
        return
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is None:
        try:
            n = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout.count("GPU ")
        except Exception:                                    # noqa: BLE001  (no nvidia-smi: CPU box)
            n = 0
        if n > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif "," in v:
        os.environ["CUDA_VISIBLE_DEVICES"] = v.split(",")[0]


_one_gpu()

import torch  # noqa: E402

from lgs1.lineage import canonical_hash, code_commit, dir_fingerprint, sha256_file
from lgs1.paths import env
from lgs4.common import device_auto
from lgs5.config import E2_L_ALL, S5Paths, parse_Ls

HARNESS = "s7-cost-v1"
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32)
DEFAULT_KS = ("4", "8", "16", "32", "64", "all")


# ------------------------------------------------------------------------------------------------
# method specs
# ------------------------------------------------------------------------------------------------
@dataclass
class CostSpec:
    kind: str                     # lg | sliding | dense | base | cd
    name: str                     # run dir or guard
    k: int = 32
    skip: bool = True             # lg: block-local skipping (False = full depth on every window, E6-13)
    pad: bool = False             # lg: pad the head batch to k windows when the document has fewer (E5-5)
    stream: bool = True           # lg: window streaming (False = keep every window's trunk state)
    policy: str = ""              # dense/base: head | tail | yarn | native | trunc_keep_head | trunc_keep_tail ; cd: W:overlap
    W: int = 512
    overlap: int = 256
    chunk: Optional[int] = None   # lg/sliding: streaming chunk in windows (c<N>; default --chunk-windows 64).  G2 (c) 9/25: the
                                  # 1k -> 128k peak growth (+40 %) is the transient of one 64-window (16k-token) chunk -> sweep c8/16/32
    head_batch: Optional[int] = None   # lg: deep head on h<N> buffered windows at a time (9/26: +17 % floor at c8 = the 32-window head batch)
    stride: Optional[int] = None  # lg/sliding: window stride override (E6-2 s256 = disjoint windows), as lgs5 `lg:<run>:s256`
    tail_pad: int = 0             # lg/sliding: neutral suffix tokens appended before windowing (S8 tail pad `tp64`; the main-config
                                  # candidate of 9/28 is k32 + tp64 + rg3, so its cost is measured as lg:<run>:k32:tp64:c16:h8)

    @property
    def opts(self) -> str:
        """The eval-side options in the lgs5 method spelling ('_s256', '_tp64'): they change the rows, so they are part of the
        method name, unlike the harness knobs (+cN / +hN / +noskip) that only change how the same rows are computed."""
        return (f"_s{self.stride}" if self.stride else "") + (f"_tp{self.tail_pad}" if self.tail_pad else "")

    @property
    def tag(self) -> str:
        t = f"{self.kind}__{self.name}"
        if self.kind == "lg":
            t += f"_k{self.k}{self.opts}" + ("" if self.skip else "_noskip") + ("_pad" if self.pad else "") + ("" if self.stream else "_nostream")
        elif self.kind == "sliding" and self.opts:
            t += self.opts
        elif self.kind == "cd":
            t += f"_cd{self.W}o{self.overlap}"
        elif self.policy:
            t += f"_{self.policy}"
        if self.chunk and self.kind in ("lg", "sliding"):
            t += f"_c{self.chunk}"
        if self.head_batch and self.kind == "lg":
            t += f"_h{self.head_batch}"
        return t

    @property
    def method(self) -> str:
        if self.kind == "lg":
            return f"lg_k{self.k}{self.opts}" + ("" if self.skip else "+noskip") + ("+pad" if self.pad else "") + ("" if self.stream else "+nostream") + \
                (f"+c{self.chunk}" if self.chunk else "") + (f"+h{self.head_batch}" if self.head_batch else "")
        if self.kind == "sliding":
            return f"sliding{self.opts}" + (f"+c{self.chunk}" if self.chunk else "")
        if self.kind == "cd":
            return f"{self.name}/cd{self.W}o{self.overlap}"
        if self.kind == "dense":
            return "dense_ft" if self.policy in ("", "head") else f"dense_ft_{self.policy}"
        if self.kind == "base":
            return f"{self.name}/{self.policy or 'native'}"
        return self.kind


def parse_cost_spec(s: str) -> CostSpec:
    parts = s.split(":")
    kind, name = parts[0], parts[1] if len(parts) > 1 else ""
    if kind not in ("lg", "sliding", "dense", "base", "cd") or not name:
        raise SystemExit(f"bad cost method {s!r}: lg:<run>[:kK][:sS][:tpN][:cN][:hN][:noskip][:pad][:nostream] | sliding:<run>[:sS][:tpN][:cN] | "
                         f"dense:<run>[:head|tail|yarn] | base:<guard>[:native|yarn|trunc_keep_head|trunc_keep_tail] | cd:<guard>[:W[:overlap]]")
    sp = CostSpec(kind, name)
    if kind == "cd":
        sp.W = int(parts[2]) if len(parts) > 2 else 512
        sp.overlap = int(parts[3]) if len(parts) > 3 else sp.W // 2
        return sp
    for x in parts[2:]:
        m = re.fullmatch(r"k(\d+)", x)
        mc = re.fullmatch(r"c(\d+)", x)
        mh = re.fullmatch(r"h(\d+)", x)
        ms = re.fullmatch(r"s(\d+)", x)
        mt = re.fullmatch(r"tp(\d+)", x)
        if m and kind == "lg":
            sp.k = int(m.group(1))
        elif mc and kind in ("lg", "sliding"):
            sp.chunk = int(mc.group(1))
        elif mh and kind == "lg":
            sp.head_batch = int(mh.group(1))
        elif ms and kind in ("lg", "sliding"):
            sp.stride = int(ms.group(1))
        elif mt and kind in ("lg", "sliding"):
            sp.tail_pad = int(mt.group(1))
        elif x == "noskip" and kind == "lg":
            sp.skip = False
        elif x == "pad" and kind == "lg":
            sp.pad = True
        elif x == "nostream" and kind == "lg":
            sp.stream = False
        elif kind in ("dense", "base"):
            sp.policy = x
        else:
            raise SystemExit(f"bad cost method {s!r}: unknown option {x!r}")
    return sp


# ------------------------------------------------------------------------------------------------
# environment metadata
# ------------------------------------------------------------------------------------------------
def gpu_slug(override: Optional[str] = None) -> str:
    if override:
        return override
    if torch.cuda.is_available():
        n = torch.cuda.get_device_name(0).lower()
        n = n.replace("nvidia", "").replace("geforce", "").replace("tesla", "").replace("pcie", "").replace("-", "").replace("gb", "")
        return re.sub(r"[^a-z0-9]+", "", n) or "gpu"
    return "cpu"


def env_meta(model=None, dtype: Optional[torch.dtype] = None) -> Dict:
    import transformers
    meta = {"host": socket.gethostname(), "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "dtype": str(dtype).replace("torch.", "") if dtype is not None else None,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None}
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        p = torch.cuda.get_device_properties(0)
        meta["gpu_mem_gb"] = round(p.total_memory / 2 ** 30, 1)
        meta["capability"] = f"{p.major}.{p.minor}"
        meta["n_gpus_visible"] = torch.cuda.device_count()               # must be 1 for an E4 row (see _one_gpu)
        meta["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            meta["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                                            text=True, timeout=10).stdout.strip().splitlines()[0]
        except Exception:
            meta["driver"] = None
        try:
            meta["sdpa_flash"] = bool(torch.backends.cuda.flash_sdp_enabled())
            meta["sdpa_mem_efficient"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
        except Exception:
            pass
    else:
        meta["gpu"] = "cpu"
    if model is not None:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            meta["attn_implementation"] = getattr(cfg, "_attn_implementation", None)
            meta["model_type"] = getattr(cfg, "model_type", None)
    return meta


def _sync(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, device: str) -> float:
    """Wall ms of one call with device synchronisation (CUDA events on the GPU)."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        return float(s.elapsed_time(e))
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def _pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return float(ys[i])


def stats(ms: Sequence[float]) -> Dict:
    return {"p50_ms": _pct(ms, 0.5), "p90_ms": _pct(ms, 0.9), "p99_ms": _pct(ms, 0.99), "mean_ms": (sum(ms) / len(ms)) if ms else float("nan"),
            "min_ms": min(ms) if ms else float("nan"), "max_ms": max(ms) if ms else float("nan"), "n": len(ms)}


# ------------------------------------------------------------------------------------------------
# documents
# ------------------------------------------------------------------------------------------------
def docs_for_L(P: S5Paths, L: int, n: int, tok=None, seed: int = 0) -> Tuple[List[str], str]:
    """Up to n distinct test/benign_long texts of length L (file order); when the data is absent, n random-token strings of
    ~L tokens (source='synthetic')."""
    f = P.data.synth / "test" / "benign_long" / "rows.jsonl.gz"
    texts: List[str] = []
    if f.exists():
        import gzip
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("L_target") == L:
                    texts.append(r["text"])
                    if len(texts) >= n:
                        break
    if texts:
        return texts, "test/benign_long"
    g = torch.Generator().manual_seed(seed + L)
    words = ["the", "of", "and", "to", "in", "system", "data", "report", "model", "value", "process", "result", "state", "policy",
             "window", "token", "note", "section", "figure", "table"]
    out = []
    for _ in range(n):
        idx = torch.randint(0, len(words), (int(L),), generator=g).tolist()
        out.append(" ".join(words[i] for i in idx))
    return out, "synthetic"


# ------------------------------------------------------------------------------------------------
# method drivers: prepare(text) -> input ; run(input) ; run_batch(inputs)
# ------------------------------------------------------------------------------------------------
class LGCost:
    def __init__(self, P: S5Paths, spec: CostSpec, device: str, dtype: torch.dtype, chunk_windows: int = 64):
        from lgs1.lgcore.model import LandmarkGuard
        from lgs2.tok import HFTok
        from lgs4 import backbone as BB
        from lgs4.config import BACKBONES
        from lgs5.runners import lg_model_sha
        run_dir = P.run_dir(spec.name)
        hc = BB.load_ckpt(run_dir / BB.HEAD_FILE)
        bb = BACKBONES.get(hc["backbone"])
        model, _, head, router, cfg, info = BB.load_lg(run_dir, device=device, k=spec.k if spec.kind == "lg" else 32, dtype=dtype)
        if router is None:
            raise SystemExit(f"{run_dir / BB.ROUTER_FILE} missing")
        if spec.stride:                                       # E6-2 stride override, as lgs5.runners.LGEval
            import dataclasses
            cfg = dataclasses.replace(cfg, stride=int(spec.stride))
        self.lg = LandmarkGuard(model, cfg, router, head).to(device).eval()
        self.tok = HFTok(bb.repo if bb is not None else hc["repo"], bb.tok if bb is not None else "qwen3")
        self.spec, self.device, self.cfg, self.chunk = spec, device, cfg, (spec.chunk or chunk_windows)
        self.model = model
        self.n_layers = BB.n_layers(model)
        # S8 tail pad: the same neutral suffix LGEval appends (its cost = the extra tokens; at most one more window)
        self._tail_ids: List[int] = []
        if spec.tail_pad:
            from lgs5.runners import TAIL_PAD_TEXT
            base_ids, _ = self.tok.encode_with_offsets("\n\n" + TAIL_PAD_TEXT)
            reps = -(-int(spec.tail_pad) // max(1, len(base_ids)))
            self._tail_ids = (list(base_ids) * reps)[: int(spec.tail_pad)]
        self.lineage = {"model_sha": lg_model_sha(run_dir), "router_sha": sha256_file(run_dir / BB.ROUTER_FILE),
                        "cfg_hash": canonical_hash(dict(cfg.to_dict(), harness=HARNESS, method=spec.method)), "run": spec.name,
                        "ell": cfg.ell, "b": cfg.b, "k": cfg.k, "n_layers": self.n_layers, "d": BB.hidden_size(model),
                        "stride": cfg.stride, "tail_pad": int(spec.tail_pad or 0)}

    def prepare(self, text: str):
        ids, _ = self.tok.encode_with_offsets(text)
        if self._tail_ids:
            ids = list(ids) + self._tail_ids
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @staticmethod
    def n_tok(x) -> int:
        return int(x.numel())

    def set_k(self, k: int):
        self.lg.cfg.k = int(k)

    @torch.no_grad()
    def run(self, ids, timer=None):
        from lgs1.lgcore.model import sliding_scan
        if self.spec.kind == "sliding":
            return sliding_scan(self.lg, ids, chunk_windows=self.chunk, timer=timer)
        if self.spec.stream or not self.spec.skip or self.spec.pad:
            return self.lg.forward_stream(ids, chunk_windows=self.chunk, timer=timer, skipping=self.spec.skip, pad_to_k=self.spec.pad,
                                          head_batch=self.spec.head_batch)
        return self.lg(ids, stream=False, timer=timer)

    @torch.no_grad()
    def run_batch(self, docs):
        if self.spec.kind == "sliding":
            return [self.run(d) for d in docs]
        return self.lg.forward_batch(docs, chunk_windows=self.chunk)

    def meta(self) -> Dict:
        return {**env_meta(self.model, next(self.model.parameters()).dtype), "chunk_windows": self.chunk}


class DenseCost:
    def __init__(self, P: S5Paths, spec: CostSpec, device: str, dtype: torch.dtype):
        from lgs4 import backbone as BB
        from lgs5.config import ModelSpec
        from lgs5.runners import DenseEval
        ms = ModelSpec("dense", spec.name, policy=spec.policy)
        self.ev = DenseEval(P, ms, device=device, dtype=dtype)
        self.spec, self.device = spec, device
        self.BB = BB
        self.lineage = {"model_sha": self.ev.lineage.model_sha, "cfg_hash": self.ev.lineage.cfg_hash, "run": spec.name,
                        "max_tokens": self.ev.max_tokens, "policy": self.ev.policy}

    def prepare(self, text: str):
        e = self.ev.tok(text, add_special_tokens=False)["input_ids"]
        mt = self.ev.max_tokens
        e = e[-mt:] if self.ev.policy == "tail" else e[:mt]
        return torch.tensor(e, dtype=torch.long)

    @staticmethod
    def n_tok(x) -> int:
        return int(x.numel())

    @torch.no_grad()
    def run(self, ids):
        v = torch.ones(1, ids.numel(), dtype=torch.long)
        return self.BB.dense_logits_auto(self.ev.model, self.ev.head, ids[None].to(self.device), v.to(self.device), self.ev.prefill_min, self.ev.prefill_chunk)

    @torch.no_grad()
    def run_batch(self, docs):
        L = max(d.numel() for d in docs)
        ids = torch.full((len(docs), L), self.ev.pad_id, dtype=torch.long)
        valid = torch.zeros((len(docs), L), dtype=torch.long)
        for i, d in enumerate(docs):
            ids[i, :d.numel()] = d
            valid[i, :d.numel()] = 1
        return self.BB.dense_logits(self.ev.model, self.ev.head, ids.to(self.device), valid.to(self.device))

    def meta(self) -> Dict:
        return env_meta(self.ev.model, next(self.ev.model.parameters()).dtype)


class BaseCost:
    """Zero-shot guard through lgs3.longbench (native / yarn / truncation) -- the full guard call incl. decode."""

    def __init__(self, spec: CostSpec, dtype: str = "bfloat16", batch: int = 8, prefill_chunk: Optional[int] = None):
        from lgs3.longbench import context_limit, load_guard, parse_policy, template_overhead
        self.spec = spec
        self.policy = parse_policy(spec.policy or "native")
        self.gspec, self.tok, self.model, self.dev, self.snap = load_guard(spec.name, dtype, self.policy)
        # classifier pipelines (Prompt Guard, B6 chunk-OR) carry their tokenizer on the model, tok is None (9/24: every L errored)
        self.ctok = getattr(self.tok, "tokenizer", None) or self.tok or getattr(self.model, "tokenizer", None)
        if self.ctok is None:
            raise SystemExit(f"{spec.name}: no tokenizer on the guard object (need one for fit_doc / chunking)")
        self.ctx = context_limit(spec.name, self.model)
        self.overhead = template_overhead(self.gspec, self.ctok) if self.gspec.kind in ("causal", "yesno") else 16
        self.batch = batch
        self.prefill_chunk = prefill_chunk
        self.device = self.dev
        self.lineage = {"model_sha": dir_fingerprint(self.snap) if self.snap else "none", "guard": spec.name, "policy": self.policy,
                        "cfg_hash": canonical_hash({"harness": HARNESS, "guard": spec.name, "policy": self.policy, "ctx": self.ctx})}

    def prepare(self, text: str):
        from lgs3.longbench import fit_doc
        t, n, tr = fit_doc(self.ctok, text, self.ctx, self.overhead, self.policy)
        return {"text": t, "n_tok": n, "truncated": tr, "n_in": len(self.ctok(t, add_special_tokens=False)["input_ids"]) + self.overhead}

    @staticmethod
    def n_tok(x) -> int:
        return int(x["n_in"])

    def run(self, x):
        from lgs3.longbench import score_texts, score_texts_prefill
        if self.prefill_chunk and self.gspec.kind == "causal" and x["n_in"] > self.prefill_chunk:
            return score_texts_prefill(self.gspec, self.spec.name, self.tok, self.model, self.dev, [x["text"]], self.prefill_chunk)
        return score_texts(self.gspec, self.spec.name, self.tok, self.model, self.dev, [x["text"]])

    def run_batch(self, xs):
        from lgs3.longbench import score_texts
        out = []
        texts = [x["text"] for x in xs]
        for i in range(0, len(texts), self.batch):
            out += score_texts(self.gspec, self.spec.name, self.tok, self.model, self.dev, texts[i:i + self.batch])
        return out

    def meta(self) -> Dict:
        m = env_meta(getattr(self.model, "model", self.model))
        m.update(batch_chunks=self.batch, max_new_tokens=self.gspec.max_new_tokens, guard_kind=self.gspec.kind, ctx=self.ctx)
        return m


class CDCost(BaseCost):
    """Chunk-and-decode: the guard on W-token chunks (overlap o), chunk batches of `batch`, OR verdict."""

    def __init__(self, spec: CostSpec, dtype: str = "bfloat16", batch: int = 8):
        super().__init__(CostSpec("base", spec.name, policy="native"), dtype, batch)
        self.spec = spec
        self.lineage["policy"] = {"name": "chunk_or", "W": spec.W, "overlap": spec.overlap}

    def prepare(self, text: str):
        from lgs3.longbench import chunk_ids
        ids = self.ctok(text, add_special_tokens=False)["input_ids"]
        spans = chunk_ids(ids, self.spec.W, self.spec.overlap)
        return {"chunks": [self.ctok.decode(ids[s:e], skip_special_tokens=True) for s, e in spans], "n_in": len(ids),
                "n_proc": sum(e - s for s, e in spans) + self.overhead * len(spans)}

    def run(self, x):
        from lgs3.longbench import score_texts
        out = []
        for i in range(0, len(x["chunks"]), self.batch):
            out += score_texts(self.gspec, self.spec.name, self.tok, self.model, self.dev, x["chunks"][i:i + self.batch])
        return out

    def run_batch(self, xs):
        allc = [c for x in xs for c in x["chunks"]]
        return self.run({"chunks": allc})


def make_driver(P: S5Paths, spec: CostSpec, device: str, dtype: torch.dtype, chunk_windows: int, cd_batch: int, prefill_chunk: Optional[int]):
    if spec.kind in ("lg", "sliding"):
        return LGCost(P, spec, device, dtype, chunk_windows)
    if spec.kind == "dense":
        return DenseCost(P, spec, device, dtype)
    if spec.kind == "base":
        return BaseCost(spec, str(dtype).replace("torch.", ""), cd_batch, prefill_chunk)
    if spec.kind == "cd":
        return CDCost(spec, str(dtype).replace("torch.", ""), cd_batch)
    raise ValueError(spec.kind)


# ------------------------------------------------------------------------------------------------
# experiments
# ------------------------------------------------------------------------------------------------
def _mem_reset():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _mem_now() -> Dict:
    if not torch.cuda.is_available():
        return {"peak_alloc_mb": None, "peak_reserved_mb": None, "alloc_mb": None}
    return {"peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
            "alloc_mb": round(torch.cuda.memory_allocated() / 2 ** 20, 1)}


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or "CUDA error" in str(e)


def measure_latency(drv, texts: List[str], warmup: int, iters: int, device: str) -> Dict:
    prep = [drv.prepare(t) for t in texts]
    t_tok = []
    for t in texts[:max(1, min(5, len(texts)))]:
        t0 = time.perf_counter()
        drv.prepare(t)
        t_tok.append((time.perf_counter() - t0) * 1000.0)
    for i in range(warmup):
        drv.run(prep[i % len(prep)])
    _sync(device)
    _mem_reset()
    ms = [_timed(lambda i=i: drv.run(prep[i % len(prep)]), device) for i in range(iters)]
    out = stats(ms)
    out.update(ms_tokenize_mean=sum(t_tok) / len(t_tok), n_tok_mean=sum(drv.n_tok(p) for p in prep) / len(prep), n_docs=len(prep), **_mem_now())
    out["ms_total_p50"] = out["p50_ms"] + out["ms_tokenize_mean"]
    return out


def measure_throughput(drv, texts: List[str], batch: int, warmup: int, iters: int, device: str) -> Dict:
    prep = [drv.prepare(t) for t in texts]
    docs = [prep[i % len(prep)] for i in range(batch)]
    for _ in range(warmup):
        drv.run_batch(docs)
    _sync(device)
    _mem_reset()
    ms = [_timed(lambda: drv.run_batch(docs), device) for _ in range(iters)]
    out = stats(ms)
    n_tok = sum(drv.n_tok(d) for d in docs)
    out.update(docs_per_s=batch / (out["mean_ms"] / 1000.0), tokens_per_s=n_tok / (out["mean_ms"] / 1000.0), n_tok_batch=n_tok, **_mem_now())
    return out


def measure_memory(drv, texts: List[str], device: str) -> Dict:
    prep = [drv.prepare(t) for t in texts[:2]]
    drv.run(prep[0])
    _sync(device)
    _mem_reset()
    base = _mem_now()["alloc_mb"]
    ms = _timed(lambda: drv.run(prep[-1]), device)
    m = _mem_now()
    m.update(baseline_alloc_mb=base, peak_delta_mb=(m["peak_alloc_mb"] - base) if (m["peak_alloc_mb"] is not None and base is not None) else None,
             ms=ms, n_tok=drv.n_tok(prep[-1]))
    return m


def measure_breakdown(drv, texts: List[str], repeats: int, device: str) -> Dict:
    from lgs1.lgcore.model import PhaseTimer
    tm = PhaseTimer(enabled=True, device=device)
    drv.run(drv.prepare(texts[0]))                      # warm
    tot = {}
    for i in range(repeats):
        tm.reset()
        with tm.phase("tokenize"):
            ids = drv.prepare(texts[i % len(texts)])
        drv.run(ids, timer=tm)
        for k, v in tm.ms.items():
            tot[k] = tot.get(k, 0.0) + v / repeats
    total = sum(tot.values())
    return {"phases_ms": tot, "total_ms": total, "phases_pct": {k: (v / total * 100.0 if total else None) for k, v in tot.items()}, "repeats": repeats}


# ------------------------------------------------------------------------------------------------
def row_key(exp: str, spec: CostSpec, L: int, batch: int, k) -> str:
    return f"{exp}|{spec.tag}|{L}|{batch}|{k}"


def existing_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass
    return keys


def run(P: S5Paths, exp: str, spec: CostSpec, Ls: Sequence[int], batches: Sequence[int] = DEFAULT_BATCHES, ks: Sequence[str] = DEFAULT_KS,
        warmup: int = 5, iters: int = 20, n_docs: int = 20, device: str = "auto", dtype: torch.dtype = torch.bfloat16, chunk_windows: int = 64,
        cd_batch: int = 8, prefill_chunk: Optional[int] = None, gpu: Optional[str] = None, force: bool = False, out_dir: Optional[Path] = None) -> Dict:
    device = device_auto(device)
    if device == "cpu" and dtype == torch.bfloat16:
        dtype = torch.float32
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and os.environ.get("LG_COST_ALL_GPUS") != "1":
        raise SystemExit(f"E4 rows are single-GPU measurements but {torch.cuda.device_count()} GPUs are visible "
                         f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); device_map='auto' would split the guard")
    out_dir = out_dir or (P.out / "cost" / gpu_slug(gpu))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp}__{spec.tag}.rows.jsonl"
    have = existing_keys(path) if not force else set()
    drv = make_driver(P, spec, device, dtype, chunk_windows, cd_batch, prefill_chunk)
    meta = drv.meta()
    meta.update(warmup=warmup, iters=iters, harness=HARNESS, code_commit=code_commit(Path(__file__).resolve().parents[1]),
                gpu_slug=gpu_slug(gpu), measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    n_new = 0
    with open(path, "a", encoding="utf-8") as f:
        def emit(row: Dict):
            nonlocal n_new
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            keep = {k: v for k, v in row.items() if k in ("L_target", "batch", "k", "p50_ms", "p99_ms", "docs_per_s", "peak_alloc_mb", "phases_ms", "error")}
            print(f"[cost] {exp} {spec.method} {keep}", flush=True)

        for L in Ls:
            texts, src = docs_for_L(P, L, n_docs)
            base = {"exp": exp, "method": spec.method, "model": spec.name, "kind": spec.kind, "L_target": int(L), "docs": src, "meta": meta,
                    "lineage": drv.lineage, "spec": spec.__dict__}
            if exp == "latency":
                key = row_key(exp, spec, L, 1, spec.k)
                if key in have:
                    continue
                try:
                    r = measure_latency(drv, texts, warmup, iters, device)
                    emit({**base, "key": key, "batch": 1, "k": spec.k, **r, "error": None})
                except Exception as e:                                # noqa: BLE001
                    if _is_oom(e):
                        _mem_reset()
                    emit({**base, "key": key, "batch": 1, "k": spec.k, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            elif exp == "throughput":
                for B in batches:
                    key = row_key(exp, spec, L, B, spec.k)
                    if key in have:
                        continue
                    try:
                        r = measure_throughput(drv, texts, int(B), max(1, warmup // 2), max(3, iters // 4), device)
                        emit({**base, "key": key, "batch": int(B), "k": spec.k, **r, "error": None})
                    except Exception as e:                            # noqa: BLE001
                        if _is_oom(e):
                            _mem_reset()
                        emit({**base, "key": key, "batch": int(B), "k": spec.k, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                        if _is_oom(e):
                            break                                     # larger batches will not fit either
            elif exp == "memory":
                key = row_key(exp, spec, L, 1, spec.k)
                if key in have:
                    continue
                try:
                    r = measure_memory(drv, texts, device)
                    emit({**base, "key": key, "batch": 1, "k": spec.k, **r, "error": None})
                except Exception as e:                                # noqa: BLE001
                    if _is_oom(e):
                        _mem_reset()
                    emit({**base, "key": key, "batch": 1, "k": spec.k, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            elif exp == "breakdown":
                if spec.kind not in ("lg", "sliding"):
                    raise SystemExit("breakdown is defined for lg / sliding methods")
                key = row_key(exp, spec, L, 1, spec.k)
                if key in have:
                    continue
                r = measure_breakdown(drv, texts, 3, device)
                emit({**base, "key": key, "batch": 1, "k": spec.k, **r, "error": None})
            elif exp == "ksweep":
                if spec.kind != "lg":
                    raise SystemExit("ksweep is defined for lg methods")
                for kk in ks:
                    key = row_key(exp, spec, L, 1, kk)
                    if key in have:
                        continue
                    if kk == "all":
                        from lgs1.lgcore.model import sliding_scan
                        prep = [drv.prepare(t) for t in texts]
                        for i in range(warmup):
                            sliding_scan(drv.lg, prep[i % len(prep)], chunk_windows=chunk_windows)
                        _sync(device)
                        _mem_reset()
                        ms = [_timed(lambda i=i: sliding_scan(drv.lg, prep[i % len(prep)], chunk_windows=chunk_windows), device) for i in range(iters)]
                        r = {**stats(ms), **_mem_now(), "n_tok_mean": sum(p.numel() for p in prep) / len(prep)}
                    else:
                        drv.set_k(int(kk))
                        r = measure_latency(drv, texts, warmup, iters, device)
                    emit({**base, "key": key, "batch": 1, "k": kk, **r, "error": None})
                drv.set_k(spec.k)
            else:
                raise SystemExit(f"unknown exp {exp}")
    print(f"[cost] {n_new} new rows -> {path}", flush=True)
    return {"path": str(path), "new": n_new}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E4 cost measurement")
    ap.add_argument("--exp", required=True, choices=["latency", "throughput", "memory", "breakdown", "ksweep"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--L", default="all")
    ap.add_argument("--batches", default=",".join(str(b) for b in DEFAULT_BATCHES))
    ap.add_argument("--ks", default=",".join(DEFAULT_KS))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--n-docs", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--chunk-windows", type=int, default=64)
    ap.add_argument("--cd-batch", type=int, default=8)
    ap.add_argument("--prefill-chunk", type=int, default=None)
    ap.add_argument("--gpu", default=env("LG_COST_GPU", "") or None, help="output dir slug (default: from the device name)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    P = S5Paths.load().ensure()
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]
    r = run(P, a.exp, parse_cost_spec(a.method), parse_Ls(a.L), [int(x) for x in a.batches.split(",") if x], [x for x in a.ks.split(",") if x],
            a.warmup, a.iters, a.n_docs, a.device, dt, a.chunk_windows, a.cd_batch, a.prefill_chunk, a.gpu, a.force)
    print(f"[cost] {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
