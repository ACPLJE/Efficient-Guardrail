"""B0 on the HF reference path (SPEC 4.2, 3.3; build plan S2 step 0).

`impl: hf_reference`, `gate: none` only: render with the target's chat template, greedy-generate
min(req.max_new_tokens, fairness.max_new_tokens) tokens, then yield PREFILL_START (generation
start), PREFILL_DONE, one TOKENS per token (payload text = the token's detokenised piece) or one
TOKENS at the end for a tool sink, and FINISHED. The harness turns each TOKENS into one chunk
(info.flags.b0_chunks). Times are the target's measured timestamps, taken with ctx.clock.now_ns
(the harness timebase under MonotonicClock; t_source "engine", "virtual" under VirtualClock).
A failure is one fatal ERROR event, logged with its traceback (as MockSystem).

Record writing on this path is S2 step 3 (E0.1): a job with `record` needs `write_records: false`.
"""
from __future__ import annotations

import threading
import traceback
from collections.abc import Iterator

from ag.data.types import ChunkInfo, ChunkingConfig, Generation, Request
from ag.harness.context import RunContext
from ag.harness.events import Event, EventType as E
from ag.runtime.chunking import serving_chunks
from ag.systems.base import System, SystemConfig, SystemInfo
from ag.systems.mock_system import prompt_summary
from ag.systems.registry import register_system

OFF_FLAGS = ("harm_scan", "inj_path", "output_guard")


@register_system("hf_reference")
class HFReferenceSystem(System):
    system_id = "hf_reference"

    def setup(self, cfg: SystemConfig, ctx: RunContext) -> None:
        self.cfg, self.ctx = cfg, ctx
        f, exp = cfg.flags, cfg.exp
        if f.get("gate", "none") != "none":
            raise ValueError(f"hf_reference is B0 only (gate: none), got gate={f.get('gate')!r}")
        on = [k for k in OFF_FLAGS if f.get(k, "off") != "off"]
        if on:
            raise ValueError(f"hf_reference has no guard paths; set {on} to off")
        rec = exp.get("record")
        if rec and f.get("write_records", True) is not False:
            raise NotImplementedError("Record writing on the real-model path is S2 step 3; set write_records: false")
        fair = dict(exp.get("fairness") or {})
        if float(fair.get("temperature", 0.0)) != 0.0:
            raise ValueError(f"hf_reference decodes greedily; fairness.temperature={fair['temperature']} is not supported")
        self.max_new = int(fair.get("max_new_tokens", 256))
        self.ignore_eos = bool(fair.get("ignore_eos", False))
        self.chunking = ChunkingConfig(**dict(f.get("chunking", exp.get("chunking") or {})))
        self.t_source = "virtual" if ctx.clock.kind == "virtual" else "engine"
        from ag.targets.registry import create_hf_target
        self.target = create_hf_target(exp["target"], model_path=f.get("model_path"), device=f.get("device"),
                                       dtype=f.get("dtype") or fair.get("dtype"), targets_dir=f.get("targets_dir"),
                                       split_tokens=int((rec or {}).get("split_tokens", 128)), now_ns=ctx.clock.now_ns,
                                       attn_implementation=f.get("attn_implementation", "sdpa"))
        self._lock = threading.Lock()          # one model on one device: one generation at a time
        import transformers
        self.engine_version = f"transformers-{transformers.__version__}"
        # kept apart from the target so info() still works after teardown()
        self._ident = {"model_sha": self.target.model_sha, "device": str(self.target.device),
                       "attn": self.target.attn_implementation}
        ctx.log.info("hf_reference: %s", dict(self.target.info()))

    def info(self) -> SystemInfo:
        f = dict(self.cfg.flags)
        f.update(uses_judge=False, b0_chunks=True, engine_version=self.engine_version)
        return SystemInfo(system_id=self.cfg.system_id, mode="live", engine=str(f.get("engine", "hf")),
                          attn_backend=self._ident["attn"], patch_applied=False, patch_sha=None,
                          shim_active=False, model_sha={"target": self._ident["model_sha"]},
                          placement={"target": self._ident["device"]}, flags=f)

    def teardown(self) -> None:
        tgt = getattr(self, "target", None)
        if tgt is None:
            return
        self.target = None
        del tgt
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------------------------------
    def run(self, req: Request) -> Iterator[Event]:
        try:
            events = self._generate(req)
        except Exception as exc:  # fatal error ends the request (SPEC 3.1)
            self.ctx.log.error("request %s failed:\n%s", req.request_id, traceback.format_exc())
            events = [Event(self.ctx.run_id, req.request_id, E.ERROR, self.ctx.clock.now_ns(), self.t_source,
                            attempt=self.ctx.attempt, fatal=True, reason=f"{type(exc).__name__}: {exc}")]
        yield from events

    def _generate(self, req: Request) -> list[Event]:
        tgt = self.target
        rp = tgt.render(req)
        n_new = min(req.max_new_tokens, self.max_new)
        with self._lock:
            out = tgt.generate(rp, n_new, record=None, ignore_eos=req.ignore_eos or self.ignore_eos)
            timing = dict(tgt.last_timing)
        pieces = tgt.token_texts(out.output_ids)
        T = len(out.output_ids)
        att = {"attempt": self.ctx.attempt}

        def ev(type_: E, t: int, **kw) -> Event:
            return Event(self.ctx.run_id, req.request_id, type_, int(t), self.t_source, **att, **kw)

        f = out.t_prefill_done_ns
        evs = [ev(E.PREFILL_START, timing["t_start_ns"]), ev(E.PREFILL_DONE, f)]
        if req.sink == "tool":
            if T:
                evs.append(ev(E.TOKENS, out.t_token_ns[-1], tok_range=(0, T), payload={"text": "".join(pieces)}))
            chunks = serving_chunks(T, self.chunking, tool_spans=[(0, T)])
        else:
            evs += [ev(E.TOKENS, out.t_token_ns[i], tok_range=(i, i + 1), payload={"text": pieces[i]})
                    for i in range(T)]
            chunks = tuple(ChunkInfo(i, i, i + 1) for i in range(T))
        gen = Generation(req.request_id, tuple(out.output_ids), out.text, out.text, (), chunks, False, None)
        perf = {"peak_mem_bytes": out.peak_mem_bytes, "prefill_ns": timing["prefill_ns"],
                "decode_ns": timing["decode_ns"], "n_out": T, "n_prompt": rp.n_tokens}
        evs.append(ev(E.FINISHED, out.t_token_ns[-1] if T else f, generation=gen,
                      payload={"prompt": prompt_summary(rp), "perf": perf}))
        return evs
