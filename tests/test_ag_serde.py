"""SPEC 2.0 serialisation: enums by name, bytes base64, int map keys, +-inf strings."""
import math

import numpy as np

from ag.data.types import (AttackMeta, ChunkInfo, Generation, Outcome, RenderedPrompt, Request, Segment, Verdict,
                           WindowSpec, is_unsafe, make_request_id, pair_key_of, episode_id_of)
from ag.harness.events import DeadlineKind, Event, EventType, JobKind, event_from_row, event_row


def _req(attack=True):
    rid = make_request_id("bipia", "test", "32k", 123, attack)
    return Request(rid, pair_key_of(rid), episode_id_of(rid), 0, "bipia", "test", "32k", None, "qa",
                   (Segment("system", "sys", True), Segment("document", "doc IGNORE", False, ((4, 10),))),
                   "stream", 64, False, AttackMeta("I", "x", 0.5) if attack else None, {"a": 1})


def test_ids():
    rid = make_request_id("bipia", "test", "32k", 123, True)
    assert rid == "bipia/test/32k/000123/a/t00"
    assert pair_key_of(rid) == "bipia/test/32k/000123"
    assert episode_id_of(rid) == "bipia/test/32k/000123/a"


def test_request_roundtrip():
    r = _req()
    d = r.to_json()
    assert d["segments"][1]["payload_char_spans"] == [[4, 10]]
    assert Request.from_json(d) == r
    assert Request.from_json(_req(False).to_json()).attack is None


def test_rendered_prompt_bytes_roundtrip():
    mask = np.packbits(np.array([0, 1, 1, 0, 1], dtype=bool), bitorder="little").tobytes()
    rp = RenderedPrompt("r", "MT", (5, 6, 7, 8, 9), mask, ((1, 3),), (), 2, 3, bytes([0, 1, 1]), (0,), (), "half", 2)
    d = rp.to_json()
    assert isinstance(d["untrusted_mask"], str)
    back = RenderedPrompt.from_json(d)
    assert back == rp
    assert back.untrusted_bool().tolist() == [False, True, True, False, True]
    assert back.block_untrusted_array().tolist() == [False, True, True]


def test_outcome_int_keys_and_inf():
    o = Outcome("run", "request", "u", "live", {}, "F5", float("inf"), "AG", "MT", "MJ", {"inj": True}, True,
                False, True, "b0run", True, True, "inj_window", ("inj_window",), False, None, 3, 2, 4, "128x2",
                {0: (1, 2), 1: (3,)}, (2,), (), {"union": 1}, {"union": {8: True, 1: False}}, {8: True},
                {"probe": -math.inf}, None, {})
    d = o.to_json()
    assert d["tau"] == "inf" and d["doc_scores"]["probe"] == "-inf"
    assert set(d["selected"]) == {"0", "1"}
    back = Outcome.from_json(d)
    assert back.selected == {0: (1, 2), 1: (3,)}
    assert back.hit["union"][8] is True and back.hit_live == {8: True}
    assert math.isinf(back.tau)


def test_event_enums_by_name():
    v = Verdict("MJ", "r", 0, 3, "128x2", 0.7, 0.5, "F5", is_unsafe(0.7, 0.5), 10, 5)
    gen = Generation("r", (1, 2), "a b", "a", (), (ChunkInfo(0, 0, 1),), False, None)
    e = Event("run", "r", EventType.JOB_DONE, 10, "virtual", job_kind=JobKind.INJ_JUDGE,
              deadline_kind=DeadlineKind.RELEASE, verdicts=(v,), generation=gen, payload={"batch_windows": 1})
    row = event_row(e)
    assert row["type"] == "JOB_DONE" and row["job_kind"] == "INJ_JUDGE" and row["schema_version"] == 1
    assert event_from_row(row) == e


def test_decision_rule_strict():
    assert is_unsafe(0.51, 0.5) and not is_unsafe(0.5, 0.5)


def test_window_spec():
    assert str(WindowSpec.parse("256x2")) == "256x2"
    assert WindowSpec() == WindowSpec(128, 2)
