"""OpenAI-compatible /v1/chat/completions front (SPEC 4.12). S1 backend: mock.

Request extensions per message: ag_trusted (bool), ag_role ("document" for document segments sent
as role "user"), ag_payload_spans ([[start, end], ...] char spans in content). Tool results
(role "tool") are untrusted. Header X-AG-Episode sets the episode; X-AG-Turn the turn.

Response: every SSE delta carries ag_chunk_id; a tool call is one delta. A blocked request ends
with finish_reason "content_filter", no tool call, and ag_block_reason. Whatever the backend, the
front calls System.run and sends every event to the collector.
"""
from __future__ import annotations

import itertools
import json
import logging
import threading
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ag.data.types import AttackMeta, Request, Segment, episode_id_of, make_request_id, pair_key_of
from ag.harness.context import Clock
from ag.harness.events import Event, EventType as E
from ag.harness.runner import drive_request
from ag.systems.base import System

ROLES = ("system", "user", "assistant", "tool", "document")


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):      # [{"type": "text", "text": ...}, ...]
        return "".join(p.get("text", "") for p in content if isinstance(p, Mapping))
    raise ValueError(f"unsupported content type {type(content).__name__}")


def message_to_segment(m: Mapping[str, Any]) -> Segment:
    role = m.get("ag_role") or m.get("role")
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    default_trusted = role in ("system", "user", "assistant")
    trusted = bool(m.get("ag_trusted", default_trusted))
    text = _content_text(m.get("content"))
    spans = tuple((int(a), int(b)) for a, b in m.get("ag_payload_spans") or ())
    for a, b in spans:
        if not 0 <= a < b <= len(text):
            raise ValueError(f"ag_payload_spans {a, b} outside content of length {len(text)}")
    return Segment(role, text, trusted, payload_char_spans=spans, tool_name=m.get("name"),
                   tool_call_id=m.get("tool_call_id"))


def messages_to_request(body: Mapping[str, Any], *, idx: int, episode: str | None = None, turn: int = 0,
                        source: str = "openai", split: str = "test", length_class: str = "na",
                        max_new_default: int = 256,
                        attack_lookup: Callable[[str], AttackMeta | None] | None = None) -> Request:
    segs = tuple(message_to_segment(m) for m in body.get("messages") or ())
    if not segs:
        raise ValueError("messages is empty")
    has_payload = any(s.payload_char_spans for s in segs)
    if episode:
        rid = f"{episode}/t{turn:02d}"
    else:
        rid = make_request_id(source, split, length_class, idx, has_payload, turn)
    attack = attack_lookup(episode_id_of(rid)) if attack_lookup else None
    if attack is None and has_payload:
        attack = AttackMeta(n_sites=sum(len(s.payload_char_spans) for s in segs))
    task = next((s.text for s in segs if s.role == "user" and s.trusted), "")
    return Request(
        request_id=rid, pair_key=pair_key_of(rid), episode_id=episode_id_of(rid), turn=turn, source=source,
        split=split, length_class=length_class, background_tier=None, task=task[:200], segments=segs,
        sink="tool" if body.get("tools") else "stream",
        max_new_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or max_new_default),
        ignore_eos=bool(body.get("ignore_eos", False)), attack=attack, tools=tuple(body.get("tools") or ()))


class OpenAIFront:
    """In-process front: builds a Request, drives System.run, turns delivered chunks into deltas."""

    def __init__(self, system: System, *, emit: Callable[[Event], None], clock: Clock, run_id: str,
                 b0_chunks: bool = False, model: str = "ag-mock", attempt: int = 0):
        self.system, self.emit, self.clock, self.run_id = system, emit, clock, run_id
        self.b0_chunks, self.model, self.attempt = b0_chunks, model, attempt
        self._idx = itertools.count()
        self._lock = threading.Lock()

    def _next_idx(self) -> int:
        with self._lock:
            return next(self._idx)

    def stream(self, body: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> Iterator[dict]:
        """Validate the request now (ValueError before anything is sent) and return a generator of
        OpenAI chat.completion.chunk dicts; the last one has finish_reason set."""
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        try:
            turn = int(headers.get("x-ag-turn", 0))
        except ValueError as exc:
            raise ValueError(f"bad X-AG-Turn header: {exc}") from exc
        req = messages_to_request(body, idx=self._next_idx(), episode=headers.get("x-ag-episode"), turn=turn)
        return self._chunks(req)

    def _chunks(self, req: Request) -> Iterator[dict]:
        cid = f"chatcmpl-{zlib.crc32(req.request_id.encode()):010d}"
        created = int(time.time())
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": self.model,
                "ag_request_id": req.request_id}
        block_reason = None
        src = "virtual" if getattr(self.clock, "kind", "") == "virtual" else "harness"
        for e in drive_request(self.system, req, emit=self.emit, clock=self.clock, run_id=self.run_id,
                               attempt=self.attempt, b0_chunks=self.b0_chunks, t_source=src):
            if e.type == E.CHUNK_DELIVERED:
                text = e.payload.get("text", "")
                if e.chunk_id:
                    text = " " + text
                yield {**base, "ag_chunk_id": e.chunk_id,
                       "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
            elif e.type in (E.CHUNK_BLOCKED, E.REQUEST_BLOCKED):
                block_reason = e.reason or "blocked"
            elif e.type == E.ERROR and e.fatal:
                block_reason = block_reason or f"error:{e.reason}"
        final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason":
                                      "content_filter" if block_reason else "stop"}]}
        if block_reason:
            final["ag_block_reason"] = block_reason
        yield final

    def complete(self, body: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> dict:
        parts, last = [], None
        for ch in self.stream(body, headers):
            last = ch
            parts.append(ch["choices"][0]["delta"].get("content", ""))
        out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": self.model,
               "ag_request_id": last["ag_request_id"],
               "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                            "finish_reason": last["choices"][0]["finish_reason"]}]}
        if "ag_block_reason" in last:
            out["ag_block_reason"] = last["ag_block_reason"]
        return out


def make_server(front: OpenAIFront, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet; the collector is the record
            pass

        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/health", "/v1/models"):
                self._json(200, {"status": "ok", "data": [{"id": front.model, "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                headers = dict(self.headers.items())
                if not body.get("stream"):
                    return self._json(200, front.complete(body, headers))
                chunks = front.stream(body, headers)       # validates; a ValueError still becomes a 400
            except ValueError as exc:
                return self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for ch in chunks:
                    self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except ValueError as exc:   # headers are out: report in-band, never a second HTTP response
                err = {"error": {"message": str(exc), "type": "invalid_request_error"}}
                self.wfile.write(f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n".encode())
                self.wfile.flush()
            except OSError as exc:      # client went away mid-stream: let System.run finish so the request
                logging.getLogger("ag.openai_front").warning(     # still ends with FINISHED (SPEC 3.4 #7)
                    "client disconnected mid-stream from %s: %r", self.client_address, exc)
                for _ in chunks:
                    pass

    return ThreadingHTTPServer((host, port), Handler)
