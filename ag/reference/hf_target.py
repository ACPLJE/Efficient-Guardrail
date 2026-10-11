"""HF transformers reference target (SPEC 4.3; plan 6.4, 8.2; build plan S2 step 0).

The real-model B0 path without attention recording: the HF chat template renders the segments,
the fast tokenizer's offset mapping turns char spans into token spans (ag.data.render), and a
manual greedy loop (SDPA prefill with logits_to_keep=1, then one token per step with the KV cache)
generates the output. Ledger recording on this path (SDPA prefill + eager decode) is S2 step 3.

Works with transformers 4.51+ and 5.x: `dtype=` falls back to `torch_dtype=`, `logits_to_keep`
falls back to no kwarg.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ag.data.render import CharSpan, ChatTemplate, render
from ag.data.types import RenderedPrompt, Request, Segment
from ag.targets.base import RecordSpec, Target, TargetOutput

log = logging.getLogger("ag.reference.hf_target")

# SPEC 4.12: documents travel as user messages
ROLE_MAP = {"system": "system", "user": "user", "document": "user", "tool": "tool", "assistant": "assistant"}
DOWNLOAD_HINT = "download it on the login node first (bash slurm/ag_download_login.sh --stage pilot), never inside sbatch"


# --------------------------------------------------------------------------------------------
# tokenizer and chat template
# --------------------------------------------------------------------------------------------
class HFTokenizerAdapter:
    """ag.data.render.Tokenizer over a HF fast tokenizer. The rendered chat template already holds
    BOS and the role headers, so encoding never adds special tokens."""

    def __init__(self, tok: Any):
        if not getattr(tok, "is_fast", False):
            raise TypeError(f"{type(tok).__name__} is not a fast tokenizer; offset mapping is required (SPEC A-03)")
        self.tok = tok
        self.vocab_size = len(tok)

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[CharSpan]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [int(i) for i in enc["input_ids"]], [(int(a), int(b)) for a, b in enc["offset_mapping"]]

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: Sequence[int]) -> str:
        """Window text: special tokens kept (SPEC 2.2)."""
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def decode_output(self, ids: Sequence[int]) -> str:
        """Output text as a client sees it: special tokens (eos, role markers) dropped."""
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def token_texts(self, ids: Sequence[int]) -> list[str]:
        """Incremental detokenisation, one piece per token, with "".join(pieces) == decode_output(ids).

        piece_i = decode_output(ids[:i+1])[len(emitted):]. A prefix whose text ends in U+FFFD (a
        multi-byte character split over tokens) or does not extend the emitted text is held back and
        released with a later token, as streaming detokenisers do; held tokens get "".
        """
        ids = [int(i) for i in ids]
        pieces: list[str] = []
        emitted = ""
        for i in range(len(ids)):
            cur = self.decode_output(ids[:i + 1])
            last = i == len(ids) - 1
            if not last and (cur.endswith("�") or not cur.startswith(emitted)):
                pieces.append("")
                continue
            if not cur.startswith(emitted):     # last token rewrote earlier text: cannot happen for HF decoders
                raise ValueError(f"detokenisation is not prefix-stable: {emitted!r} vs {cur!r}")
            pieces.append(cur[len(emitted):])
            emitted = cur
        return pieces


def hf_chat_template(tok: Any) -> ChatTemplate:
    """Segments -> (prompt text, content char spans) through tok.apply_chat_template.

    One message per segment (role map above), add_generation_prompt=True, no extra template kwargs
    (T2 is an instruct, non-thinking model). Each segment text is located in order with
    str.find(text, cursor); a template that changes the content (e.g. `| trim`) raises ValueError
    naming the segment, so generators must strip their texts."""
    if not getattr(tok, "chat_template", None):
        raise ValueError(f"{type(tok).__name__} has no chat_template")

    def template(segments: Sequence[Segment]) -> tuple[str, list[CharSpan]]:
        msgs = []
        for i, s in enumerate(segments):
            if s.role not in ROLE_MAP:
                raise ValueError(f"segment {i}: unknown role {s.role!r}")
            msgs.append({"role": ROLE_MAP[s.role], "content": s.text})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if not isinstance(text, str):
            raise TypeError(f"apply_chat_template(tokenize=False) returned {type(text).__name__}, expected str")
        spans: list[CharSpan] = []
        cur = 0
        for i, s in enumerate(segments):
            a = text.find(s.text, cur)
            if a < 0:
                raise ValueError(f"segment {i} (role {s.role}) does not appear verbatim in the rendered chat template "
                                 "(a template that trims or rewrites content needs stripped segment texts)")
            spans.append((a, a + len(s.text)))
            cur = a + len(s.text)
        return text, spans

    return template


# --------------------------------------------------------------------------------------------
# model files
# --------------------------------------------------------------------------------------------
def resolve_model_dir(model_path: str | os.PathLike, *, revision: str | None = None,
                      local_files_only: bool = True) -> Path:
    """A local directory as is; otherwise a hub repo id resolved to its HF-cache snapshot
    (.../snapshots/<sha>; revision None = refs/main). Never downloads when local_files_only."""
    p = Path(os.path.expanduser(os.fspath(model_path)))
    if p.is_dir():
        return p.resolve()
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo_id=os.fspath(model_path), revision=revision,
                                      local_files_only=local_files_only))
    except Exception as e:
        raise FileNotFoundError(f"model {os.fspath(model_path)!r} (revision {revision or 'main'}) is not in the HF cache "
                                f"(HF_HOME={os.environ.get('HF_HOME')}): {type(e).__name__}: {e}; {DOWNLOAD_HINT}") from e


# files that decide what a local (non-HF-cache) model directory computes and how it renders prompts
# (exact names, no prefix globs: `hf download --local-dir` keeps timestamped vocab.json.metadata files under .cache/)
LOCAL_ID_PATTERNS = ("*.safetensors", "*.bin", "config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json", "added_tokens.json",
                     "chat_template.jinja", "chat_template.json", "vocab.json", "vocab.txt", "merges.txt",
                     "*.tiktoken")


def model_sha_of(model_dir: str | os.PathLike) -> str:
    """The snapshot commit for an HF-cache path .../snapshots/<sha>; otherwise "local:" + a fingerprint of the
    weights, config, tokenizer and chat template (lgs1 dir_fingerprint, file hashes cached by size + mtime), so a
    swapped checkpoint or an edited template is a different model_sha and resume refuses it (CLAUDE.md 3.3)."""
    p = Path(model_dir)
    if p.name and p.parent.name == "snapshots" and p.parent.parent.name.startswith(("models--", "datasets--")):
        return p.name
    from ag.lineage_util import dir_fingerprint
    return "local:" + dir_fingerprint(p, LOCAL_ID_PATTERNS)[:16]


def load_tokenizer(model_dir: str | os.PathLike) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.fspath(model_dir), local_files_only=True)


def load_causal_lm(model_dir: str | os.PathLike, torch_dtype: Any, attn_implementation: str, device: Any) -> Any:
    from transformers import AutoModelForCausalLM
    kw: dict[str, Any] = {"attn_implementation": attn_implementation, "local_files_only": True}

    def load(**extra: Any) -> Any:
        try:
            return AutoModelForCausalLM.from_pretrained(os.fspath(model_dir), dtype=torch_dtype, **kw, **extra)
        except TypeError:           # transformers < 4.56 knows only torch_dtype
            return AutoModelForCausalLM.from_pretrained(os.fspath(model_dir), torch_dtype=torch_dtype, **kw, **extra)

    model = None
    if getattr(device, "type", "cpu") == "cuda":
        try:                        # load straight onto the GPU (needs accelerate)
            model = load(device_map={"": str(device)})
        except (ImportError, ValueError) as e:
            log.warning("device_map load failed (%s); loading on CPU and moving", e)
    if model is None:
        model = load()
    if model.dtype != torch_dtype:  # an ignored dtype kwarg must not silently change the numerics
        model = model.to(torch_dtype)
    if model.device.type != getattr(device, "type", str(device)):
        model = model.to(device)
    return model.eval()


def _attn_kinds(text_cfg: Any, n_layers: int) -> tuple[str, ...]:
    lt = getattr(text_cfg, "layer_types", None)
    if not lt:
        return tuple(["full"] * n_layers)
    return tuple("full" if k == "full_attention" else "sliding" for k in lt)


# --------------------------------------------------------------------------------------------
# target
# --------------------------------------------------------------------------------------------
class HFReferenceTarget(Target):
    """Target over a HF causal LM (single device). See the module docstring."""

    def __init__(self, model_path: str | os.PathLike, *, target_id: str, revision: str | None = None,
                 dtype: str = "bfloat16", device: str | None = None, attn_implementation: str = "sdpa",
                 split_tokens: int = 128, now_ns: Callable[[], int] | None = None, local_files_only: bool = True,
                 max_context: int | None = None):
        import torch
        self.target_id = target_id
        self.model_path = os.fspath(model_path)
        self.revision = revision
        self.model_dir = resolve_model_dir(model_path, revision=revision, local_files_only=local_files_only)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = "float32" if self.device.type == "cpu" else dtype       # bf16 matmuls on CPU are slow and inexact
        tdt = getattr(torch, self.dtype, None)
        if not isinstance(tdt, torch.dtype):
            raise ValueError(f"unknown dtype {dtype!r}")
        self.split_tokens = int(split_tokens)
        self.now_ns = now_ns or time.monotonic_ns
        self.tokenizer = load_tokenizer(self.model_dir)
        self.adapter = HFTokenizerAdapter(self.tokenizer)
        self.template = hf_chat_template(self.tokenizer)
        self.model = load_causal_lm(self.model_dir, tdt, attn_implementation, self.device)
        cfg = self.model.config
        text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        self.n_layers = int(text.num_hidden_layers)
        self.n_q_heads = int(text.num_attention_heads)
        self.n_kv_heads = int(getattr(text, "num_key_value_heads", None) or self.n_q_heads)
        self.layer_attn_kind = _attn_kinds(text, self.n_layers)
        self.max_context = int(max_context or getattr(text, "max_position_embeddings", 0) or 1 << 20)
        self.attn_implementation = str(getattr(cfg, "_attn_implementation", None) or attn_implementation)
        eos = self.model.generation_config.eos_token_id
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        if self.tokenizer.eos_token_id is not None:
            eos.append(self.tokenizer.eos_token_id)
        self.eos_ids = frozenset(int(e) for e in eos)
        import inspect
        params = inspect.signature(self.model.forward).parameters
        self._ltk = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in params), None)
        self.last_timing: dict[str, int] = {}
        self._model_sha = model_sha_of(self.model_dir)

    # ---------------------------------------------------------------- identity
    @property
    def model_sha(self) -> str:
        return self._model_sha

    def info(self) -> Mapping[str, Any]:
        import torch
        import transformers
        return {"target_id": self.target_id, "hf_repo": None if Path(self.model_path).is_dir() else self.model_path,
                "model_path": str(self.model_dir), "model_sha": self.model_sha, "dtype": self.dtype,
                "attn_implementation": self.attn_implementation, "device": str(self.device),
                "n_layers": self.n_layers, "n_q_heads": self.n_q_heads, "n_kv_heads": self.n_kv_heads,
                "max_context": self.max_context, "transformers": transformers.__version__, "torch": torch.__version__}

    # ---------------------------------------------------------------- rendering
    def render(self, req: Request) -> RenderedPrompt:
        rp = render(req, self.adapter, self.template, self.target_id, self.split_tokens)
        if rp.n_tokens > self.max_context:
            raise ValueError(f"{req.request_id}: {rp.n_tokens} tokens > max_context {self.max_context}")
        return rp

    def token_texts(self, output_ids: Sequence[int]) -> list[str]:
        return self.adapter.token_texts(output_ids)

    # ---------------------------------------------------------------- generation
    def _forward(self, ids: Any, past: Any, *, prefill: bool) -> Any:
        kw: dict[str, Any] = {"input_ids": ids, "use_cache": True}
        if past is not None:
            kw["past_key_values"] = past
        if prefill and self._ltk:
            kw[self._ltk] = 1           # no [n, vocab] logits at 128k (SPEC 4.3)
        try:
            return self.model(**kw)
        except TypeError:
            if not (prefill and self._ltk):
                raise
            kw.pop(self._ltk)
            self._ltk = None
            return self.model(**kw)

    def generate(self, rp: RenderedPrompt, max_new_tokens: int, *, record: RecordSpec | None = None,
                 forced_output: Sequence[int] | None = None, drop_tokens: Sequence[tuple[int, int]] = (),
                 ignore_eos: bool = False) -> TargetOutput:
        """Greedy decode. Stops after an eos token (included in output_ids) unless ignore_eos or
        forced_output. t_token_ns[i] = when token i is known; t_token_ns[0] == t_prefill_done_ns."""
        if record is not None:
            raise NotImplementedError("ledger recording on the real-model path is S2 step 3 (E0.1)")
        if drop_tokens:
            raise NotImplementedError("drop_tokens (leave-one-out scorer) is S2 work")
        import torch
        T = len(forced_output) if forced_output is not None else int(max_new_tokens)
        if rp.n_tokens + T > self.max_context:
            raise ValueError(f"{rp.request_id}: prompt {rp.n_tokens} + {T} new tokens > max_context {self.max_context}")
        stop_on_eos = not ignore_eos and forced_output is None
        cuda = self.device.type == "cuda"

        def sync() -> None:
            if cuda:
                torch.cuda.synchronize(self.device)

        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        sync()
        t0 = self.now_ns()
        out_ids: list[int] = []
        times: list[int] = []
        with torch.inference_mode():
            ids = torch.tensor([list(rp.prompt_ids)], dtype=torch.long, device=self.device)
            res = self._forward(ids, None, prefill=True)
            past = res.past_key_values
            for t in range(T):
                nxt = int(forced_output[t]) if forced_output is not None else int(res.logits[0, -1].argmax())
                sync()
                times.append(self.now_ns())
                out_ids.append(nxt)
                if t + 1 == T or (stop_on_eos and nxt in self.eos_ids):
                    break
                res = self._forward(torch.tensor([[nxt]], dtype=torch.long, device=self.device), past, prefill=False)
                past = res.past_key_values
            if not times:
                sync()
            t_f = times[0] if times else self.now_ns()
            del past, res
        peak = int(torch.cuda.max_memory_allocated(self.device)) if cuda else 0
        self.last_timing = {"t_start_ns": t0, "prefill_ns": t_f - t0, "decode_ns": (times[-1] - t_f) if times else 0}
        return TargetOutput(tuple(out_ids), self.adapter.decode_output(out_ids), (), t_f, tuple(times), peak, None)
