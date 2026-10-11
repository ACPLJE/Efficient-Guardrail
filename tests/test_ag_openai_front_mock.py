"""SPEC 4.12: OpenAI-compatible front over the mock backend (trust fields, SSE ag_chunk_id,
content_filter on block, events through the collector)."""
import json
import threading
import urllib.request

import pytest

from ag.config.schema import load_system
from ag.data import mock_lc_inject
from ag.harness.checks import run_checks
from ag.harness.collector import Collector
from ag.harness.context import RunContext, VirtualClock
from ag.harness.lineage import build_lineage
from ag.harness.openai_front import OpenAIFront, make_server, messages_to_request
from ag.systems.base import SystemConfig
from ag.systems.registry import create_system

EXP = {"fairness": {"max_new_tokens": 48}, "operating_point": "F5", "chunking": {"first": 8, "next": 32},
       "record": None}


def _front(tmp_path, system_name):
    sysd = load_system(system_name)
    clock = VirtualClock()
    col = Collector("run-front", tmp_path / "events.jsonl")
    lin = build_lineage("run-front", "mock", "c", clock="virtual", seed=0, producer="mock", model_sha={},
                        data_manifest_sha="")
    import logging
    ctx = RunContext("run-front", "mock", tmp_path, clock, 0, lin, None, col.emit, logging.getLogger("t"))
    system = create_system(sysd["impl"])
    system.setup(SystemConfig(sysd["system_id"], sysd["impl"], sysd["flags"], exp=EXP), ctx)
    front = OpenAIFront(system, emit=col.emit, clock=clock, run_id="run-front",
                        b0_chunks=bool(system.info().flags.get("b0_chunks")))
    return front, col, system


def _messages(attack: bool):
    reqs = mock_lc_inject.generate(n_pairs=1, length_classes=["1k"], seed=1)
    r = next(x for x in reqs if x.is_attack == attack)
    sys_seg, user_seg, doc = r.segments
    m_doc = {"role": "user", "ag_role": "document", "ag_trusted": False, "content": doc.text}
    if attack:
        m_doc["ag_payload_spans"] = [list(s) for s in doc.payload_char_spans]
    return [{"role": "system", "content": sys_seg.text}, {"role": "user", "content": user_seg.text}, m_doc]


def test_messages_to_request_mapping():
    body = {"messages": _messages(True) + [{"role": "tool", "content": "x", "tool_call_id": "c1"}],
            "max_tokens": 16, "tools": [{"type": "function"}]}
    req = messages_to_request(body, idx=3, episode="agentdojo_l/test/32k/000045/a", turn=2)
    assert req.request_id == "agentdojo_l/test/32k/000045/a/t02" and req.turn == 2
    assert [s.role for s in req.segments] == ["system", "user", "document", "tool"]
    assert [s.trusted for s in req.segments] == [True, True, False, False]
    assert req.segments[2].payload_char_spans and req.is_attack and req.sink == "tool" and req.max_new_tokens == 16
    with pytest.raises(ValueError):
        messages_to_request({"messages": [{"role": "user", "content": "ab", "ag_payload_spans": [[0, 9]]}]}, idx=0)


def test_complete_benign_and_attack(tmp_path):
    front, col, _ = _front(tmp_path, "mock_ag")
    ok = front.complete({"messages": _messages(False)})
    assert ok["choices"][0]["finish_reason"] == "stop" and "ANSWER" in ok["choices"][0]["message"]["content"]
    bad = front.complete({"messages": _messages(True)})
    assert bad["choices"][0]["finish_reason"] == "content_filter" and bad["ag_block_reason"] == "inj_window"
    assert "PWNED" not in bad["choices"][0]["message"]["content"]
    col.close()
    assert all(c.ok for c in run_checks(col.events, {"uses_judge": True}))


def test_client_disconnect_mid_stream_still_finishes(tmp_path):
    import socket
    import time
    from ag.harness.context import MonotonicClock
    sysd = load_system("mock_ag")
    flags = {**sysd["flags"], "mock": {**sysd["flags"]["mock"], "tpot_ms": 30.0}}
    col = Collector("run-front", tmp_path / "events.jsonl")
    lin = build_lineage("run-front", "mock", "c", clock="monotonic", seed=0, producer="mock", model_sha={},
                        data_manifest_sha="")
    import logging
    ctx = RunContext("run-front", "mock", tmp_path, MonotonicClock(), 0, lin, None, col.emit, logging.getLogger("t"))
    system = create_system(sysd["impl"])
    system.setup(SystemConfig(sysd["system_id"], sysd["impl"], flags, exp=EXP), ctx)
    front = OpenAIFront(system, emit=col.emit, clock=MonotonicClock(), run_id="run-front")
    srv = make_server(front)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        body = json.dumps({"messages": _messages(False), "stream": True, "max_tokens": 48}).encode()
        s = socket.create_connection(srv.server_address, timeout=10)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        buf = b""
        while b"ag_chunk_id" not in buf:                           # first delta arrived
            buf += s.recv(4096)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
        s.close()                                                  # RST: the client is gone
        deadline = time.time() + 20
        while time.time() < deadline and not any(e.type.name == "FINISHED" for e in col.events):
            time.sleep(0.05)
    finally:
        srv.shutdown()
        srv.server_close()
        col.close()
    assert sum(e.type.name == "FINISHED" for e in col.events) == 1
    assert all(c.ok for c in run_checks(col.events, {"uses_judge": True}))


def test_http_sse_stream(tmp_path):
    front, col, _ = _front(tmp_path, "mock_b0")
    srv = make_server(front)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        port = srv.server_address[1]
        body = json.dumps({"messages": _messages(False), "stream": True, "max_tokens": 12}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            lines = [l.decode().strip() for l in resp if l.strip()]
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(l[len("data: "):]) for l in lines[:-1]]
        ids = [c["ag_chunk_id"] for c in chunks if "ag_chunk_id" in c]
        assert ids == list(range(12))                       # B0: one chunk per token
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as resp:
            assert json.loads(resp.read())["status"] == "ok"
        bad = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=b'{"messages": []}',
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(bad, timeout=10)
        assert ei.value.code == 400
        bad_stream = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                            data=b'{"messages": [], "stream": true}',
                                            headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(bad_stream, timeout=10)
        assert ei.value.code == 400 and b"messages is empty" in ei.value.read()
    finally:
        srv.shutdown()
        col.close()
