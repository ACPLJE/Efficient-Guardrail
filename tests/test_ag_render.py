"""SPEC 2.2: rendering keeps the length, maps trust and payload char spans to token spans."""
from ag.data import mock_lc_inject
from ag.data import window_map as wm
from ag.data.render import render, window_text
from ag.targets.mock import MockTokenizer

BIG = 1 << 24


def test_mock_requests_render_to_target_length():
    reqs = mock_lc_inject.generate(n_pairs=2, length_classes=["2k"], seed=3)
    tok = MockTokenizer(BIG)
    lens = {r.request_id: render(r, tok).n_tokens for r in reqs}
    assert set(lens.values()) == {2048}


def test_payload_tokens_and_trust():
    reqs = mock_lc_inject.generate(n_pairs=1, length_classes=["1k"], seed=0)
    a = next(r for r in reqs if r.is_attack)
    b = next(r for r in reqs if not r.is_attack)
    tok = MockTokenizer(BIG)
    rp = render(a, tok)
    (s, e), = rp.payload_spans
    assert list(rp.prompt_ids[s:e]) == tok.encode(mock_lc_inject.PAYLOAD)
    untr = rp.untrusted_bool()
    assert untr[s:e].all()
    assert not untr[:5].any()                         # system header + text are trusted
    assert rp.payload_rule == "half" and 1 <= len(rp.payload_windows) <= 3
    assert rp.eligible_windows == int(wm.eligible_windows(rp.block_untrusted_array()).sum())
    rb = render(b, tok)
    assert rb.payload_windows == () and rb.payload_rule == "none" and rb.n_tokens == rp.n_tokens


def test_window_text_decodes_window_tokens():
    reqs = mock_lc_inject.generate(n_pairs=1, length_classes=["1k"], seed=0)
    tok = MockTokenizer(BIG)
    rp = render(reqs[0], tok)
    w = rp.payload_windows[0]
    text, ids = window_text(rp, w, tok)
    assert len(ids) == 256 and "PWNED" in text.split()
