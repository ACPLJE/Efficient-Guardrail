"""SPEC 1.1-1.3: block/window counts, eligibility, payload windows (D-019, D-026)."""
import numpy as np
import pytest

from ag.data import window_map as wm


@pytest.mark.parametrize("n,nb,nw", [(8192, 64, 63), (32768, 256, 255), (129024, 1008, 1007),
                                     (131072, 1024, 1023), (130, 2, 1), (100, 1, 1)])
def test_counts(n, nb, nw):
    assert wm.n_blocks(n) == nb
    assert wm.n_windows(n) == nw


def test_window_span_and_last_short_block():
    assert wm.window_token_span(0, 1000) == (0, 256)
    assert wm.window_token_span(6, 1000) == (768, 1000)


def test_eligibility_any_untrusted_block():
    untr = np.zeros(640, dtype=bool)
    untr[300] = True                                     # block 2 only
    bu = wm.block_untrusted(untr)
    assert bu.tolist() == [0, 0, 1, 0, 0]
    assert wm.eligible_windows(bu).tolist() == [False, True, True, False]


def test_payload_half_rule_one_to_three_windows():
    n = 1280
    # payload inside one block -> the two windows containing that block
    assert wm.payload_windows_for([(300, 310)], n) == ((1, 2), "half")
    # payload split evenly over a block boundary -> three windows hold >= half
    pw, rule = wm.payload_windows_for([(246, 266)], n)
    assert rule == "half" and pw == (0, 1, 2)


def test_payload_max_overlap_for_long_payload():
    pw, rule = wm.payload_windows_for([(0, 900)], 1280)
    assert rule == "max_overlap" and len(pw) == 1


def test_per_site():
    union, per, rule = wm.payload_windows([(10, 20), (900, 910)], 1280, n_sites=2)
    assert rule == "per_site" and per == ((0,), (6, 7)) and union == (0, 6, 7)


def test_window_scores_ineligible_inf():
    s = wm.window_scores_from_blocks(np.array([0.1, 0.2, 0.3]), np.array([True, False]))
    assert s[0] == pytest.approx(0.3) and np.isneginf(s[1])
