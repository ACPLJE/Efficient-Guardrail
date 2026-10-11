"""Deterministic mock system (SPEC 4.2, build plan 4.1).

gate = "none"  B0: PREFILL_START, PREFILL_DONE, one TOKENS per token (or one at the end for a tool
               sink), FINISHED. The harness turns each TOKENS into one chunk (SPEC 3.3).
gate = "ag"    hold-then-release: per serving chunk, select windows from the live ledger (mock
               target attention), judge them with the mock judge (one pool worker, verdict cache),
               release when every gating verdict is safe, block (fail-closed) otherwise.
               Flags: block_before_generation, release_before_verdict, overlap, verdict_cache.

All times are computed from the request start (prefill cost per token, TPOT, judge latency) and
events are yielded in time order after clock.sleep_until(t), so the run is a discrete-event
simulation under VirtualClock and a paced replay under MonotonicClock.
"""
from __future__ import annotations

import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ag.data.render import window_text
from ag.data.types import ChunkInfo, ChunkingConfig, Generation, RenderedPrompt, Request, Verdict, is_unsafe
from ag.harness.context import RunContext
from ag.harness.events import DeadlineKind, Event, EventType as E, JobKind
from ag.judges.base import JudgeInput, VerdictCache, cache_key
from ag.judges.mock_judge import MockJudge
from ag.runtime.chunking import serving_chunks
from ag.selector.select import SelectorConfig, chunk_block_mass, select, window_scores
from ag.systems.base import System, SystemConfig, SystemInfo
from ag.systems.registry import register_system
from ag.targets.base import RecordSpec
from ag.targets.mock import MockTarget


@dataclass
class _Timeline:
    """Events with planned times; yielded sorted by (t, insertion order)."""
    run_id: str
    request_id: str
    t_source: str

    def __post_init__(self) -> None:
        self.items: list[tuple[int, int, Event]] = []

    def add(self, type_: E, t_ns: int, **kw) -> int:
        self.items.append((t_ns, len(self.items),
                           Event(self.run_id, self.request_id, type_, int(t_ns), self.t_source, **kw)))
        return t_ns

    def sorted(self) -> list[Event]:
        return [e for _, _, e in sorted(self.items, key=lambda x: (x[0], x[1]))]


@register_system("mock")
class MockSystem(System):
    system_id = "mock"

    def setup(self, cfg: SystemConfig, ctx: RunContext) -> None:
        self.cfg, self.ctx = cfg, ctx
        f = cfg.flags
        m = dict(f.get("mock", {}))
        self.gate = f.get("gate", "ag")
        if self.gate not in ("none", "ag"):
            raise ValueError(f"mock gate must be none|ag, got {self.gate!r}")
        if f.get("harm_scan", "off") != "off":
            raise ValueError("the harm path (harm_scan) is S3 work; mock systems need harm_scan: off")
        if f.get("release_before_verdict") and f.get("block_before_generation"):
            raise ValueError("release_before_verdict and block_before_generation are exclusive (SPEC 9.1)")
        tgt = dict(m.get("target", {}))
        self.target = MockTarget(tgt.pop("kind", "synthetic"), seed=ctx.seed, **tgt)
        self.prefill_ns_per_token = int(m.get("prefill_us_per_token", 20.0) * 1e3)
        self.tpot_ns = int(m.get("tpot_ms", 20.0) * 1e6)
        self.gate_ns = int(m.get("gate_us", 50.0) * 1e3)
        exp = cfg.exp
        fair = dict(exp.get("fairness", {}))
        self.max_new = int(fair.get("max_new_tokens", 64))
        self.chunking = ChunkingConfig(**dict(f.get("chunking", exp.get("chunking", {}))))
        rec = exp.get("record")
        self.record_spec = RecordSpec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in rec.items()}) \
            if (rec and self.gate == "none" and f.get("write_records", True)) else None
        self.t_source = "virtual" if ctx.clock.kind == "virtual" else ("client" if self.gate == "none" else "engine")
        if self.gate == "ag":
            sel = dict(f.get("selector", {}))
            if isinstance(sel.get("layers"), list):
                sel["layers"] = tuple(sel["layers"])
            self.selector = SelectorConfig(**sel)
            self.selector.validate()
            if self.selector.rule == "oracle" and not f.get("evaluation_only", False):
                raise ValueError("selector.rule=oracle needs evaluation_only: true in a live run (SPEC 9.1)")
            jcfg = dict(m.get("judge", {}))
            self.judge = MockJudge(f.get("judge", "MJ"), seed=ctx.seed, **jcfg)
            op = exp.get("operating_point", "F5")
            self.threshold_id = op
            self.tau = float(f["thresholds"][self.judge.judge_id][op])
            self.window_spec = (f.get("judge_windows") or ["128x2"])[0]
            self.cache = VerdictCache(f.get("verdict_cache", "per_request"))
            self.overlap = bool(f.get("overlap", True))
            self.rbv = bool(f.get("release_before_verdict", False))
            self.bbg = bool(f.get("block_before_generation", False))
            self.release_budget_ns = int(dict(f.get("deadline_budget_ms", {})).get("release", 1000) * 1e6)
            # live ledger: fp32, only what the selector reads
            self.live_spec = RecordSpec(layers="all", per_chunk_heads=False, gen_mass=False,
                                        first_token=self.selector.first_token == "include", precision="fp32")

    def reset(self) -> None:
        if self.gate == "ag":
            self.cache.clear()

    def info(self) -> SystemInfo:
        f = dict(self.cfg.flags)
        f.update(uses_judge=self.gate == "ag", b0_chunks=self.gate == "none")
        return SystemInfo(
            system_id=self.cfg.system_id, mode="live", engine="mock", attn_backend=None, patch_applied=False,
            patch_sha=None, shim_active=False,
            model_sha={"target": f"mock-{self.target.kind}-s{self.ctx.seed}",
                       **({"judge": self.judge.judge_id} if self.gate == "ag" else {})},
            placement={}, flags=f)

    # ------------------------------------------------------------------------------------------
    def run(self, req: Request) -> Iterator[Event]:
        clock = self.ctx.clock
        try:
            events = self._plan(req, clock.now_ns())
        except Exception as exc:  # fatal error ends the request (SPEC 3.1)
            self.ctx.log.error("request %s failed:\n%s", req.request_id, traceback.format_exc())
            events = [Event(self.ctx.run_id, req.request_id, E.ERROR, clock.now_ns(), self.t_source,
                            attempt=self.ctx.attempt, fatal=True, reason=f"{type(exc).__name__}: {exc}")]
        for e in events:
            clock.sleep_until(e.t_ns)
            yield e

    def _words(self, out, T: int) -> list[str]:
        w = out.text.split()
        return w if len(w) == T else [self.target.tokenizer.decode([i]) for i in out.output_ids]

    def _plan(self, req: Request, t0: int) -> list[Event]:
        rp = self.target.render(req)
        tl = _Timeline(self.ctx.run_id, req.request_id, self.t_source)
        att = {"attempt": self.ctx.attempt}
        n_new = min(req.max_new_tokens, self.max_new)
        if self.gate == "none":
            return self._plan_b0(req, rp, tl, t0, n_new, att)
        return self._plan_ag(req, rp, tl, t0, n_new, att)

    def _plan_b0(self, req, rp, tl, t0, n_new, att) -> list[Event]:
        out = self.target.generate(rp, n_new, record=self.record_spec)
        if out.record is not None and self.ctx.records is not None:
            self.ctx.records.put(out.record)
        T = len(out.output_ids)
        words = self._words(out, T)
        s = tl.add(E.PREFILL_START, t0, **att)
        f = tl.add(E.PREFILL_DONE, s + self.prefill_ns_per_token * rp.n_tokens + self.tpot_ns, **att)
        tok = [f + i * self.tpot_ns for i in range(T)]
        if req.sink == "tool":
            tl.add(E.TOKENS, tok[-1], tok_range=(0, T), payload={"text": " ".join(words)}, **att)
            chunks = serving_chunks(T, self.chunking, tool_spans=[(0, T)])
        else:
            for i in range(T):
                tl.add(E.TOKENS, tok[i], tok_range=(i, i + 1), payload={"text": words[i]}, **att)
            chunks = tuple(ChunkInfo(i, i, i + 1) for i in range(T))
        gen = Generation(req.request_id, tuple(out.output_ids), out.text, out.text, (), chunks, False, None)
        tl.add(E.FINISHED, tok[-1] if T else f, generation=gen, payload={"prompt": prompt_summary(rp)}, **att)
        return tl.sorted()

    def _judge_inputs(self, req: Request, rp: RenderedPrompt, chunk_id: int | None, wins: Sequence[int],
                      chunk_text: str | None, chunk_ids: tuple[int, ...] | None) -> list[JudgeInput]:
        task = next((s.text for s in req.segments if s.role == "user" and s.trusted), req.task)
        out = []
        for w in wins:
            text, ids = window_text(rp, w, self.target.tokenizer)
            out.append(JudgeInput(req.request_id, chunk_id, w, self.window_spec, task, text, ids,
                                  chunk_text, chunk_ids, self.threshold_id))
        return out

    def _judge(self, tl: _Timeline, req, rp, chunk_id, wins, t_submit, pool_free, att, *, deadline_ns,
               chunk_text=None, chunk_ids=None):
        """Cache lookup + one job for the misses.
        Returns (job_id | None, done_ns | None, cached_ns | None, verdicts, pool_free)."""
        inputs = self._judge_inputs(req, rp, chunk_id, wins, chunk_text, chunk_ids)
        cached: list[Verdict] = []
        miss: list[tuple[JudgeInput, tuple]] = []
        for x in inputs:
            key = cache_key(self.judge, x)
            v = self.cache.get(req.request_id, key)
            (cached.append(v) if v is not None else miss.append((x, key)))
        cached_ns = max((v.decided_ns for v in cached), default=None)
        verdicts = list(cached)
        job_id, done = None, None
        if miss:
            self._job_seq += 1
            job_id = f"{req.request_id}#{JobKind.INJ_JUDGE}#{self._job_seq}"
            batch = [x for x, _ in miss]
            start = max(t_submit, pool_free)
            done = start + self.judge.latency_ns(batch)
            scores = self.judge.score(batch)
            new = [Verdict(self.judge.judge_id, req.request_id, chunk_id, x.window_id, self.window_spec, s,
                           self.tau, self.threshold_id, is_unsafe(s, self.tau), done, done - start)
                   for x, s in zip(batch, scores)]
            for (x, key), v in zip(miss, new):
                self.cache.put(req.request_id, key, Verdict(**{**v.__dict__, "cached": True}))
            common = dict(job_id=job_id, job_kind=JobKind.INJ_JUDGE, chunk_id=chunk_id,
                          window_ids=tuple(x.window_id for x in batch), **att)
            tl.add(E.JOB_SUBMITTED, t_submit, deadline_kind=DeadlineKind.RELEASE if deadline_ns else DeadlineKind.NONE,
                   deadline_ns=deadline_ns, **common)
            tl.add(E.JOB_STARTED, start, payload={"batch_windows": len(batch)}, **common)
            tl.add(E.JOB_DONE, done, verdicts=tuple(new), payload={"batch_windows": len(batch)}, **common)
            verdicts += new
            pool_free = done
        return job_id, done, cached_ns, verdicts, pool_free

    def _plan_ag(self, req, rp, tl, t0, n_new, att) -> list[Event]:
        self._job_seq = 0
        out = self.target.generate(rp, n_new, record=self.live_spec)
        rec = out.record
        T = len(out.output_ids)
        words = self._words(out, T)
        chunks = serving_chunks(T, self.chunking, tool_spans=[(0, T)] if req.sink == "tool" else ())
        pool_free = t0
        s = t0
        pre_job, pre_done, pre_cached = None, None, None
        if self.bbg:   # judge before generation; chunk 0 carries the job, later chunks are exempt
            sc = window_scores(chunk_block_mass(rec, chunks[0], self.selector, n_target_layers=self.target.n_layers),
                               rp) if self.selector.rule not in ("all", "oracle") else _eligible_scores(rp)
            wins = select(sc, self.selector, oracle=rp.payload_windows, key=req.request_id)
            pre_job, pre_done, pre_cached, vs, pool_free = self._judge(tl, req, rp, None, wins, t0, pool_free, att,
                                                                       deadline_ns=None)
            t_ok = max(x for x in (t0, pre_done, pre_cached) if x is not None)
            sel_payload = {"selected": list(wins), "cached_windows": []}
            if any(v.unsafe for v in vs):
                tl.add(E.REQUEST_BLOCKED, t_ok + self.gate_ns, reason="inj_window", payload=sel_payload, **att)
                gen = Generation(req.request_id, (), "", "", (), (), True, "inj_window")
                tl.add(E.FINISHED, t_ok + self.gate_ns, generation=gen, payload={"prompt": prompt_summary(rp)},
                       **att)
                return tl.sorted()
            s = t_ok
        tl.add(E.PREFILL_START, s, **att)
        f = tl.add(E.PREFILL_DONE, s + self.prefill_ns_per_token * rp.n_tokens + self.tpot_ns, **att)
        shift = 0
        prev_rel = f
        delivered: list[str] = []
        blocked, reason = False, None
        unsafe_at: int | None = None
        last_t = f
        gen_chunks = []
        for c in chunks:
            g = f + (c.tok_end - 1) * self.tpot_ns + shift
            text = " ".join(words[c.tok_start:c.tok_end])
            gen_chunks.append(c)
            tl.add(E.TOKENS, g, tok_range=(c.tok_start, c.tok_end), **att)
            if self.bbg and c.chunk_id >= 1:
                job_id, done, cached_ns, vs = None, None, None, []
                wins: tuple[int, ...] = ()
            elif self.bbg:
                job_id, done, cached_ns, vs, wins = pre_job, pre_done, pre_cached, [], ()
            else:
                sc = window_scores(chunk_block_mass(rec, c, self.selector, n_target_layers=self.target.n_layers), rp)
                wins = select(sc, self.selector, oracle=rp.payload_windows, key=req.request_id)
                job_id, done, cached_ns, vs, pool_free = self._judge(
                    tl, req, rp, c.chunk_id, wins, g, pool_free, att, deadline_ns=g + self.release_budget_ns,
                    chunk_text=text if self.judge.needs_chunk else None,
                    chunk_ids=tuple(out.output_ids[c.tok_start:c.tok_end]) if self.judge.needs_chunk else None)
            cached_w = [v.window_id for v in vs if v.cached]
            tl.add(E.CHUNK_GENERATED, g, chunk_id=c.chunk_id, tok_range=(c.tok_start, c.tok_end),
                   required_jobs=(job_id,) if job_id else (), cached_verdict_ns=cached_ns,
                   payload={"selected": list(wins), "cached_windows": cached_w}, **att)
            V = max([x for x in (done, cached_ns) if x is not None], default=None)
            unsafe = any(v.unsafe for v in vs)
            deadline = g + self.release_budget_ns
            if self.rbv:
                if unsafe_at is not None and unsafe_at <= g:
                    blocked, reason = True, "inj_window"
                    last_t = tl.add(E.CHUNK_BLOCKED, g + self.gate_ns, chunk_id=c.chunk_id, reason=reason, **att)
                    break
                prev_rel = tl.add(E.CHUNK_RELEASED, max(prev_rel, g) + self.gate_ns, chunk_id=c.chunk_id,
                                  payload={"text": text}, **att)
                delivered.append(text)
                if unsafe and V is not None:
                    unsafe_at = V if unsafe_at is None else min(unsafe_at, V)
                last_t = max(last_t, prev_rel, V or 0)
                continue
            if V is not None and V > deadline and job_id is not None:
                # no verdict by the deadline: fail closed before the verdict arrives (SPEC 3.4 #6)
                blocked, reason = True, "deadline_miss"
                last_t = tl.add(E.CHUNK_BLOCKED, deadline, chunk_id=c.chunk_id, reason=reason, **att)
                break
            if unsafe:
                blocked, reason = True, "inj_window"
                last_t = tl.add(E.CHUNK_BLOCKED, max(g, V) + self.gate_ns, chunk_id=c.chunk_id, reason=reason, **att)
                break
            prev_rel = tl.add(E.CHUNK_RELEASED, max(prev_rel, g, V if V is not None else g) + self.gate_ns,
                              chunk_id=c.chunk_id, payload={"text": text}, **att)
            delivered.append(text)
            last_t = max(last_t, prev_rel)
            if not self.overlap and V is not None and V > g:
                shift += V - g
        last_t = max(last_t, pool_free)
        gen = Generation(req.request_id, tuple(out.output_ids), out.text, " ".join(delivered), (),
                         tuple(gen_chunks), blocked, reason)
        tl.add(E.FINISHED, last_t, generation=gen, payload={"prompt": prompt_summary(rp)}, **att)
        return tl.sorted()


def prompt_summary(rp: RenderedPrompt) -> dict:
    """What the harness copies into outcomes_run (payload windows, window counts)."""
    from ag.data import window_map as wm
    return {"n_tokens": rp.n_tokens, "n_blocks": rp.n_blocks, "n_windows": wm.n_windows(rp.n_tokens, rp.split_tokens),
            "n_eligible": rp.eligible_windows, "payload_windows": list(rp.payload_windows),
            "payload_windows_per_site": [list(x) for x in rp.payload_windows_per_site],
            "payload_rule": rp.payload_rule}


def _eligible_scores(rp: RenderedPrompt):
    """Score 0 on every eligible window (rule=all / oracle do not need the ledger)."""
    import numpy as np
    from ag.data import window_map as wm
    el = wm.eligible_windows(rp.block_untrusted_array())
    s = np.zeros(len(el))
    s[~el] = -np.inf
    return s
