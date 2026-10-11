"""SPEC 1.4: serving chunks (first 8, then 32), tool calls as one chunk, standard chunk defs."""
from ag.data.types import ChunkingConfig
from ag.runtime.chunking import serving_chunks, standard_chunk


def spans(chs):
    return [(c.tok_start, c.tok_end) for c in chs]


def test_default_chunks():
    assert spans(serving_chunks(80)) == [(0, 8), (8, 40), (40, 72), (72, 80)]
    assert spans(serving_chunks(5)) == [(0, 5)]
    assert [c.chunk_id for c in serving_chunks(80)] == [0, 1, 2, 3]


def test_tool_call_is_one_chunk():
    chs = serving_chunks(60, tool_spans=[(20, 50)])
    assert spans(chs) == [(0, 8), (8, 20), (20, 50), (50, 60)]
    assert [c.is_tool_call for c in chs] == [False, False, True, False]
    assert spans(serving_chunks(30, tool_spans=[(0, 30)])) == [(0, 30)]


def test_sentence_boundaries():
    chs = serving_chunks(60, ChunkingConfig(8, 32, True), sentence_ends=[20, 45])
    assert spans(chs) == [(0, 8), (8, 20), (20, 45), (45, 60)]


def test_standard_defs():
    assert (standard_chunk("first8", 5).tok_end, standard_chunk("first32", 100).tok_end,
            standard_chunk("all", 77).tok_end) == (5, 32, 77)
