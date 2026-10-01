"""S2 text utilities: normalisation, exact/near-duplicate groups, paragraphs, insertion, token chunks."""
from lgs2 import textutil as T
from lgs2.tok import WhitespaceTok, chunk_by_tokens, token_span_for_chars


def test_norm_and_exact_key():
    assert T.norm_text("  How do I  KILL a Python process?! ") == "how do i kill a python process"
    assert T.exact_key("Hello, World") == T.exact_key("hello world")


def test_group_ids_merges_exact_and_near_duplicates():
    base = "please explain step by step how to pick a lock on a front door without any tools at all"
    texts = [base, base.upper() + "!!", base + " thanks", "what is the capital of france", "tell me a joke about cats"]
    gids, st = T.group_ids(texts)
    assert gids[0] == gids[1] == gids[2] and gids[3] != gids[0] and gids[4] not in (gids[0], gids[3])
    assert st["n_exact_unique"] == 4 and st["n_groups"] == 3 and st["n_near_dup_merged"] == 1
    # stable across input order
    gids2, _ = T.group_ids(list(reversed(texts)))
    assert set(gids2) == set(gids)


def test_group_ids_short_texts_do_not_collapse():
    gids, st = T.group_ids(["kill", "kill process", "kill", "hi there"])
    assert gids[0] == gids[2] and gids[0] != gids[1] and st["n_groups"] == 3


def test_split_paragraphs_styles():
    assert T.split_paragraphs("a\n\nb\n\n\nc", "blank") == ["a", "b", "c"]
    assert T.split_paragraphs("line1\nline2\nline3", "blank") == ["line1", "line2", "line3"]      # fallback: lines
    assert T.split_paragraphs("One sentence. Two sentence. Three.", "blank") == ["One sentence.", "Two sentence.", "Three."]
    assert T.split_paragraphs("x = 1\ny = 2\n\nprint(x)", "code") == ["x = 1\ny = 2", "print(x)"]
    assert T.split_paragraphs("x = 1\ny = 2", "code") == ["x = 1\ny = 2"]                         # never split inside code lines
    assert T.split_paragraphs("u: hi\na: hello\n", "line") == ["u: hi", "a: hello"]


def test_insert_at_depth_boundaries_and_span():
    pars = ["para one.", "para two is here.", "para three."]
    for depth in (0.0, 0.3, 0.5, 0.9, 1.0):
        text, (a, b), idx = T.insert_at_depth(pars, "NEEDLE", depth)
        assert text[a:b] == "NEEDLE"
        assert text.split("\n\n")[idx] == "NEEDLE"
        rest = [p for p in text.split("\n\n") if p != "NEEDLE"]
        assert rest == pars                                     # no paragraph was cut
    assert T.insert_at_depth(pars, "N", 0.0)[2] == 0 and T.insert_at_depth(pars, "N", 1.0)[2] == 3


def test_whitespace_tok_chunks_and_spans():
    tok = WhitespaceTok()
    text = " ".join(f"w{i}" for i in range(23))
    chunks = chunk_by_tokens(text, tok, 10)
    assert [c[0] for c in chunks] == [0, 1, 2]
    assert text[chunks[0][1]:chunks[0][2]] == " ".join(f"w{i}" for i in range(10))
    ids, offs = tok.encode_with_offsets(text)
    span = (text.index("w5"), text.index("w5") + 2)
    assert token_span_for_chars(offs, span) == (5, 6)
