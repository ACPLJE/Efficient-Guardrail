"""Window-streaming forward (S6 64k/128k rows, E4-3 flat memory) == the all-windows forward; batched forward; sliding
scan; phase timer; cost-model switches (no-skip, pad-to-k) leave the verdict unchanged."""
import torch

from lgs1.lgcore import adapter as A
from lgs1.lgcore.model import PhaseTimer, sliding_scan


def _ids(n, seed=3, vocab=512):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g)


def _lg(tiny, k=3, b=8):
    torch.manual_seed(0)
    cfg = A.LGConfig(b=b, ell=2, k=k, pool="max")
    lg = A.LandmarkGuard(tiny, cfg)
    with torch.no_grad():                                   # non-trivial router / head so top-k is not degenerate
        lg.router.proj.weight.normal_(0, 0.5)
        lg.head.proj.weight.normal_(0, 0.5)
    return lg.eval()


def test_stream_equals_full(tiny):
    lg = _lg(tiny)
    for n in (5, 16, 40, 97, 300):                          # shorter than a window, exact multiples, ragged
        ids = _ids(n)
        full = lg(ids)
        for chunk in (1, 4, 64):
            st = lg.forward_stream(ids, chunk_windows=chunk)
            assert st.windows == full.windows
            assert torch.allclose(st.window_scores, full.window_scores, atol=1e-5)
            assert torch.allclose(st.token_scores, full.token_scores, atol=1e-5)
            assert sorted(st.selected.tolist()) == sorted(full.selected.tolist()), (n, chunk)
            assert abs(st.theta - full.theta) < 1e-5
            assert abs(st.score() - full.score()) < 1e-4 and st.verdict_unsafe == full.verdict_unsafe
            assert st.n_deep == full.n_deep == min(3, len(full.windows))
            # probs aligned with `selected` in score order on both paths
            a = dict(zip(full.selected.tolist(), full.head_probs.tolist()))
            b = dict(zip(st.selected.tolist(), st.head_probs.tolist()))
            assert all(abs(a[w] - b[w]) < 1e-4 for w in a)
        via_forward = lg(ids, stream=True, chunk_windows=4)
        assert abs(via_forward.score() - full.score()) < 1e-4


def test_cost_switches_keep_verdict(tiny):
    lg = _lg(tiny, k=6)
    ids = _ids(40)                                          # 4 windows < k=6 -> pad_to_k pads the head batch
    full = lg(ids)
    ns = lg.forward_stream(ids, skipping=False)
    pd = lg.forward_stream(ids, pad_to_k=True)
    for o in (ns, pd):
        assert abs(o.score() - full.score()) < 1e-4 and o.n_deep == full.n_deep == len(full.windows)
        assert sorted(o.selected.tolist()) == sorted(full.selected.tolist())
    # head_batch: the deep head on 1 / 2 buffered windows at a time gives the same logits (windows are independent)
    lg3 = _lg(tiny, k=3)
    ids = _ids(300)
    ref = lg3.forward_stream(ids, chunk_windows=4)
    for hb in (1, 2, 3, 99):
        o = lg3.forward_stream(ids, chunk_windows=4, head_batch=hb)
        assert o.selected.tolist() == ref.selected.tolist() and torch.allclose(o.head_logits, ref.head_logits, atol=1e-4), hb
        assert o.n_deep == ref.n_deep == 3
    o = lg3.forward_stream(ids, chunk_windows=4, head_batch=2, pad_to_k=True)
    assert torch.allclose(o.head_logits, ref.head_logits, atol=1e-4)


def test_forward_batch_matches_per_doc(tiny):
    lg = _lg(tiny)
    docs = [_ids(n, seed=s) for s, n in ((1, 30), (2, 120), (3, 7))]
    outs = lg.forward_batch(docs, chunk_windows=4, head_batch=5)
    for ids, o in zip(docs, outs):
        ref = lg(ids)
        assert sorted(o.selected.tolist()) == sorted(ref.selected.tolist())
        assert abs(o.score() - ref.score()) < 1e-4 and o.verdict_unsafe == ref.verdict_unsafe
        assert torch.allclose(o.window_scores, ref.window_scores, atol=1e-5)


def test_sliding_scan_streams_every_window(tiny):
    lg = _lg(tiny, k=2)
    ids = _ids(200)
    o = sliding_scan(lg, ids, chunk_windows=5)
    assert o.n_deep == len(o.windows) == o.head_probs.numel() and sorted(o.selected.tolist()) == list(range(len(o.windows)))
    # every window through the head == the oracle of the full-k guard
    cfg = A.LGConfig(b=8, ell=2, k=10 ** 6, pool="max")
    allk = A.LandmarkGuard(tiny, cfg, lg.router, lg.head).eval()(ids)
    a = dict(zip(allk.selected.tolist(), allk.head_probs.tolist()))
    b = dict(zip(o.selected.tolist(), o.head_probs.tolist()))
    assert all(abs(a[w] - b[w]) < 1e-4 for w in a)
    assert abs(o.score() - allk.score()) < 1e-4


def test_phase_timer_breakdown(tiny):
    lg = _lg(tiny)
    tm = PhaseTimer(enabled=True, device="cpu")
    with tm.phase("tokenize"):
        ids = _ids(100)
    lg.forward_stream(ids, chunk_windows=4, timer=tm)
    assert set(tm.ms) >= {"tokenize", "trunk", "router", "head", "aggregate"}
    assert tm.calls["trunk"] == tm.calls["router"] - 1 >= 1      # one router phase per chunk + the final sort
    assert tm.total() > 0 and tm.reset().ms == {}
    off = PhaseTimer(enabled=False)
    with off.phase("x"):
        pass
    assert off.ms == {}
