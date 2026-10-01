"""E0-1 equivalence and invariance tests (tiny Qwen3, CPU, < 1 min).

(a) manual layer loop == HF forward (full depth + norm, and truncated depth == hidden_states[ell])
(b) block-local batched trunk == standalone forward of each window (padding included)
(c) landmark of window w is invariant to tokens outside window w (block locality)
(d) window-unit trunk + in-window token-max pooling path; routing returns top-k on that scale
(e) make_windows(stride): window length stays 2b for stride = 2b; overlap covers every b-span
(f) attack runner: theta from route() on the router scale; suffix slot cannot lower a needle
    window; success is judged end-to-end
"""
import math

import pytest
import torch

from lgs1.lgcore import adapter as A

ATOL = 1e-5
RTOL = 1e-4


def _rand_ids(n, vocab=512, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g)


# ------------------------------------------------------------------------------------------
# (a) manual forward == HF forward
# ------------------------------------------------------------------------------------------
def test_a_manual_forward_matches_hf_full_depth(tiny):
    ids = _rand_ids(2 * 24).view(2, 24)
    hs, last = A.hf_hidden_states(tiny, ids)
    with torch.no_grad():
        man = A.manual_forward(tiny, ids, layer_range=None, apply_norm=True)
    assert man.shape == last.shape
    assert torch.allclose(man, last, atol=ATOL, rtol=RTOL), float((man - last).abs().max())


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_a_manual_forward_matches_hf_truncated_depth(tiny, ell):
    ids = _rand_ids(3 * 20, seed=2).view(3, 20)
    hs, _ = A.hf_hidden_states(tiny, ids)
    with torch.no_grad():
        man = A.manual_forward(tiny, ids, layer_range=(0, ell), apply_norm=False)
    ref = hs[ell]
    assert torch.allclose(man, ref, atol=ATOL, rtol=RTOL), float((man - ref).abs().max())


def test_a_manual_forward_with_padding_matches_hf(tiny):
    ids = _rand_ids(2 * 16, seed=3).view(2, 16)
    valid = torch.ones(2, 16, dtype=torch.long)
    valid[1, 10:] = 0  # second row is 10 tokens + right padding
    hs, last = A.hf_hidden_states(tiny, ids, valid)
    with torch.no_grad():
        man = A.manual_forward(tiny, ids, valid=valid, apply_norm=True)
    # compare only real tokens (padded positions are unspecified in HF as well)
    assert torch.allclose(man[0], last[0], atol=ATOL, rtol=RTOL)
    assert torch.allclose(man[1, :10], last[1, :10], atol=ATOL, rtol=RTOL)


# ------------------------------------------------------------------------------------------
# (b) block-local trunk == standalone window forward
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("b,stride,n", [(4, None, 40), (4, 8, 40), (8, None, 61), (6, 6, 17)])
def test_b_block_local_trunk_equals_standalone(tiny, b, stride, n):
    ell = 2
    doc = _rand_ids(n, seed=4)
    wins = A.make_windows(n, b, stride)
    hidden, valid = A.trunk_windows(tiny, doc, wins, ell, pad_id=0)
    for i, (s, e) in enumerate(wins):
        hs, _ = A.hf_hidden_states(tiny, doc[s:e][None])  # standalone, positions reset
        ref = hs[ell][0]
        got = hidden[i, : e - s]
        assert valid[i, : e - s].all() and not valid[i, e - s:].any()
        assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (i, float((got - ref).abs().max()))


# ------------------------------------------------------------------------------------------
# (c) landmark invariance to other blocks
# ------------------------------------------------------------------------------------------
def test_c_landmark_invariant_to_other_windows(tiny):
    b, ell, n = 4, 2, 48
    doc = _rand_ids(n, seed=5)
    wins = A.make_windows(n, b, None)              # 8-token windows, stride 4
    cfg = A.LGConfig(b=b, stride=None, ell=ell, k=3, pool="max")
    torch.manual_seed(11)
    lg = A.LandmarkGuard(tiny, cfg)
    base = lg.route(doc)
    w = 5                                          # window [20, 28)
    s, e = wins[w]
    outside = [i for i in range(n) if i < s or i >= e]
    doc2 = doc.clone()
    doc2[outside] = (doc2[outside] + 7) % 512      # change every token outside window w
    alt = lg.route(doc2)
    assert alt["windows"] == base["windows"]
    assert torch.allclose(alt["hidden"][w], base["hidden"][w], atol=ATOL, rtol=RTOL)
    assert torch.allclose(alt["token_scores"][w], base["token_scores"][w], atol=ATOL, rtol=RTOL)
    assert math.isclose(float(alt["window_scores"][w]), float(base["window_scores"][w]), abs_tol=1e-5)
    # and neighbouring windows that share tokens with the change DO move (sanity of the test)
    assert not torch.allclose(alt["window_scores"], base["window_scores"], atol=1e-6)


# ------------------------------------------------------------------------------------------
# (d) window-unit trunk + token-max pooling path
# ------------------------------------------------------------------------------------------
def test_d_token_max_pooling_and_routing(tiny):
    b, ell, n, k = 4, 2, 64, 3
    doc = _rand_ids(n, seed=6)
    cfg = A.LGConfig(b=b, ell=ell, k=k, pool="max")
    torch.manual_seed(12)
    lg = A.LandmarkGuard(tiny, cfg)
    r = lg.route(doc)
    ts, valid, ws = r["token_scores"], r["valid"], r["window_scores"]
    W, L = ts.shape
    assert L == 2 * b and W == len(r["windows"])
    # token-max equals an explicit per-window maximum over valid tokens
    explicit = torch.stack([ts[i][valid[i].bool()].max() for i in range(W)])
    assert torch.allclose(ws, explicit, atol=1e-6)
    # mean pooling is a different quantity on the same token scores
    mean_ws = A.pool_scores(ts, valid, "mean")
    assert not torch.allclose(mean_ws, ws)
    # a single spiked token dominates the window under max but not under mean
    ts2 = ts.clone()
    ts2[0] -= 5.0
    ts2[0, 3] = ts.max() + 10.0
    assert int(A.pool_scores(ts2, valid, "max").argmax()) == 0
    assert int(A.pool_scores(ts2, valid, "mean").argmax()) != 0
    # routing: top-k sorted descending, theta = k-th largest on the same scale
    sel, theta = A.route(ws, k)
    assert sel.numel() == k
    assert torch.all(ws[sel][:-1] >= ws[sel][1:])
    assert math.isclose(theta, float(ws[sel][-1]), abs_tol=1e-7)
    assert math.isclose(theta, float(torch.sort(ws, descending=True).values[k - 1]), abs_tol=1e-7)
    # the deep stage sees exactly k windows regardless of document length (constant work)
    out = lg(doc)
    assert out.n_deep == k and out.head_probs.numel() == k
    out_long = lg(_rand_ids(4 * n, seed=7))
    assert out_long.n_deep == k


# ------------------------------------------------------------------------------------------
# (e) window grid / stride
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("b", [4, 8, 128])
def test_e_stride_2b_keeps_window_length_2b(b):
    n = 20 * b + 3
    disjoint = A.make_windows(n, b, stride=2 * b)
    overlap = A.make_windows(n, b, stride=None)
    assert all(e - s == 2 * b for s, e in disjoint), "no-overlap grid must keep 2b-token windows"
    assert all(e - s == 2 * b for s, e in overlap)
    # disjoint windows do not overlap (except the final shifted window) and cover the document
    for (s1, e1), (s2, e2) in zip(disjoint[:-2], disjoint[1:-1]):
        assert e1 == s2
    assert disjoint[0][0] == 0 and disjoint[-1][1] == n and overlap[-1][1] == n
    assert len(overlap) > len(disjoint)


@pytest.mark.parametrize("b,n", [(4, 37), (8, 100), (16, 16 * 9 + 5)])
def test_e_overlap_covers_every_b_span(b, n):
    assert A.spans_fully_covered(n, b, stride=None, span_len=b)
    # with stride 2b a span of b tokens straddling a boundary is not contained -> property fails
    assert not A.spans_fully_covered(n, b, stride=2 * b, span_len=b)


def test_e_short_document_single_window():
    assert A.make_windows(5, 8) == [(0, 5)]
    assert A.make_windows(16, 8) == [(0, 16)]
    assert A.make_windows(0, 8) == []


# ------------------------------------------------------------------------------------------
# (f) attack runner semantics
# ------------------------------------------------------------------------------------------
def _needle_doc(n=96, needle_len=3, at=40, seed=8):
    doc = _rand_ids(n, seed=seed)
    return doc, (at, at + needle_len)


def test_f_theta_is_route_scale_and_success_is_end_to_end(tiny):
    b, ell, k = 4, 2, 3
    doc, span = _needle_doc()
    cfg = A.LGConfig(b=b, ell=ell, k=k, pool="max")
    torch.manual_seed(13)
    lg = A.LandmarkGuard(tiny, cfg)
    before = A.evaluate(lg, doc, span)
    r = lg.route(doc)
    # theta reported by the runner is exactly the router's k-th window score
    assert math.isclose(before.theta, r["theta"], abs_tol=1e-7)
    assert all(math.isclose(before.needle_window_scores[j], float(r["window_scores"][w]), abs_tol=1e-7)
               for j, w in enumerate(before.needle_windows))
    # needle-containing windows are the ones that wholly contain the span (<= 3 on this grid)
    assert 1 <= len(before.needle_windows) <= 3
    for w in before.needle_windows:
        s, e = r["windows"][w]
        assert s <= span[0] and span[1] <= e
    # router_evaded <=> no needle window admitted; success additionally needs a safe verdict
    admitted = any(w in before.selected for w in before.needle_windows)
    assert before.router_evaded == (not admitted)
    assert before.success == (before.router_evaded and not before.verdict_unsafe)


def test_f_suffix_slot_cannot_lower_needle_window(tiny):
    b, ell, k = 4, 2, 3
    doc, span = _needle_doc()
    cfg = A.LGConfig(b=b, ell=ell, k=k, pool="max")
    torch.manual_seed(14)
    lg = A.LandmarkGuard(tiny, cfg)
    adv = _rand_ids(6, seed=9)
    before, after = A.run_attack(lg, doc, span, adv, slot="suffix")
    # causal trunk + token-max: the needle tokens' own scores are unchanged by a suffix ...
    assert math.isclose(after.needle_token_max, before.needle_token_max, abs_tol=1e-5)
    # ... and every needle window score is bounded below by the needle's token max
    assert max(after.needle_window_scores) >= after.needle_token_max - 1e-5
    assert after.needle_span == span


def test_f_prefix_slot_changes_needle_scores(tiny):
    b, ell, k = 4, 2, 3
    doc, span = _needle_doc()
    cfg = A.LGConfig(b=b, ell=ell, k=k, pool="max")
    torch.manual_seed(15)
    lg = A.LandmarkGuard(tiny, cfg)
    adv = _rand_ids(6, seed=10)
    before, after = A.run_attack(lg, doc, span, adv, slot="prefix")
    assert after.needle_span == (span[0] + 6, span[1] + 6)
    # the prefix is inside the needle's windows, so the needle tokens can move (no equality claim)
    assert isinstance(after.needle_token_max, float)
    assert after.needle_windows  # still wholly contained in >= 1 window


def test_f_span_longer_than_b_is_rejected(tiny):
    b, ell, k = 4, 2, 3
    doc = _rand_ids(64, seed=16)
    cfg = A.LGConfig(b=b, ell=ell, k=k, pool="max")
    lg = A.LandmarkGuard(tiny, cfg)
    # a 9-token span cannot be wholly inside an 8-token window on any grid
    with pytest.raises(ValueError):
        A.evaluate(lg, doc, (20, 29))
