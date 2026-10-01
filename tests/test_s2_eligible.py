"""Pool-size eligibility (2026-09-18): a domain is used at length L only when its split pool holds >= 2 x L tokens
(agentdojo screened down to 29k tokens -> usable up to L=8k; pg19/stack/wildchat/wiki_zh under budget but large)."""
import pytest

from lgs2 import synth as S


class _FakePools:
    """Only what Pools.pool_tokens / Pools.eligible need: haystack(split, domain) -> docs with n_tok."""
    def __init__(self, tokens):
        self.tokens = tokens

    def haystack(self, split, domain):
        n = self.tokens.get(domain, 0)
        return [{"id": f"{domain}-{i}", "n_tok": 1000} for i in range(n // 1000)]

    pool_tokens = S.Pools.pool_tokens
    eligible = S.Pools.eligible


def test_eligible_filters_by_pool_size():
    pools = _FakePools({"govreport": 23_000_000, "agentdojo": 29_000, "wiki_zh": 4_800_000})
    doms = ["govreport", "agentdojo", "wiki_zh"]
    assert pools.eligible("test", doms, 1024) == doms
    assert pools.eligible("test", doms, 8192) == doms                     # 29k >= 2 x 8192 = 16,384
    assert pools.eligible("test", doms, 16384) == ["govreport", "wiki_zh"]  # 29k < 32,768
    assert pools.eligible("test", doms, 131072) == ["govreport", "wiki_zh"]
    assert pools.eligible("test", doms, 4_000_000) == ["govreport"]


def test_eligible_or_exit_reports_once_and_exits_when_empty(capsys):
    pools = _FakePools({"a": 10_000, "b": 100_000})
    seen = set()
    assert S.eligible_or_exit(pools, "test", ["a", "b"], 8192, "benign_long", seen) == ["b"]
    assert S.eligible_or_exit(pools, "test", ["a", "b"], 8192, "benign_long", seen) == ["b"]
    out = capsys.readouterr().out
    assert out.count("a skipped") == 1 and "10,000 tok" in out               # printed once per (split, subset, domain, L)
    with pytest.raises(SystemExit):
        S.eligible_or_exit(pools, "test", ["a", "b"], 131072, "benign_long", seen)


def test_build_background_cuts_a_long_unpunctuated_paragraph_to_the_window():
    """2026-09-18 dry-run: 20 rows > 10% short because the last paragraph (one long chat line / minified code) had no
    sentence units to cut at.  The cut now falls back to word boundaries and, with a tokenizer, is measured exactly."""
    import random
    from lgs2.tok import WhitespaceTok
    tok = WhitespaceTok()
    words = " ".join(f"w{i}" for i in range(3000))              # one 3000-token line, no punctuation, no newlines
    docs = [{"id": "d0", "pars": [words], "par_tok": [3000], "n_tok": 3000}]
    for target in (1024, 2048):
        pars, cnt, ids = S.build_background(docs, target, random.Random(0), 0.10, tok)
        total = sum(cnt)
        assert abs(total - target) <= 0.10 * target, (target, total)
        assert tok.count(["\n\n".join(pars)])[0] == total            # counts are exact
        assert ids == ["d0"]
    # without a tokenizer the proportional estimate still lands inside the window for uniform text
    pars, cnt, _ = S.build_background(docs, 1024, random.Random(0), 0.10)
    assert abs(sum(cnt) - 1024) <= 0.10 * 1024
    # sentence units are preferred when they exist
    sents = " ".join(f"s{i} a b c d e f g h i." for i in range(400))   # 400 sentences x 10 tokens
    docs2 = [{"id": "d1", "pars": [sents], "par_tok": [4000], "n_tok": 4000}]
    pars, cnt, _ = S.build_background(docs2, 1024, random.Random(0), 0.10, tok)
    assert abs(sum(cnt) - 1024) <= 0.10 * 1024 and pars[0].rstrip().endswith(".")
