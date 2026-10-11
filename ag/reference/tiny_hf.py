"""Tiny local HF model for CPU smoke runs of the real-model path (CLAUDE.md 3.4; build plan S2 step 0).

build_tiny_hf writes a directory that AutoModelForCausalLM / AutoTokenizer load like a hub snapshot:
a randomly initialised 4-layer Qwen3 (the lgs1.lgcore.model.tiny_qwen3 sizes, 4,096 positions) and a
fast word-level tokenizer built offline with the `tokenizers` library (lower-casing normaliser,
whitespace + isolated punctuation + single-digit pre-tokeniser, [UNK] for words outside the list).
Special tokens <|im_start|>, <|im_end|>, <|endoftext|> (eos). Chat templates:
  chatml  Qwen-like, message content inserted verbatim
  trim    Llama-3.1-like, `content | trim` (generators must strip segment texts)
Random weights: for wiring tests only, never for reported numbers.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

SPECIALS = ("[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>")
PUNCT = tuple(".,;:!?'\"()[]{}-/\\&%#@*_`~^|") + ("–", "—", "‘", "’", "“", "”")
DIGITS = tuple("0123456789")
WORDS = tuple("""
the a an and or but if then than so of to in on at by for with from into over under about after before
between during without within through across against among along around near up down out off again
is are was were be been being am do does did done have has had having will would can could should may
might must shall not no yes all any each every some many much more most few less least other another
such same own only also very just even still already ever never always often sometimes here there now
today when where why how what which who whom whose this that these those it its they them their he him
his she her we us our you your i me my one two three four five six seven eight nine ten first second
third last next new old large small long short high low great good bad early late main key important
system user assistant tool document answer request task summary summarize list points title abstract
topic date dates event events happened people person place city country year years century time day
war state government king river mountain school university company team game music film book work works
history world national international public local known called named built founded became
became part number group area region population language family life death born died including include
used use using made make known name since while however because although whether both either neither
please reply write print tell call include ignore previous instructions instruction complete answer
access granted poem ocean french link product refuse unavailable fruits translate good morning german
hashtag start end word words five times recommend information more
""".split())


def default_vocab_words() -> tuple[str, ...]:
    """Specials, punctuation, digits, then the fixed word list (deduplicated, first occurrence wins)."""
    return tuple(dict.fromkeys(SPECIALS + PUNCT + DIGITS + WORDS))


CHAT_TEMPLATES = {
    "chatml": ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}"
               "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"),
    "trim": ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] | trim }}"
             "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"),
}
FORMAT_VERSION = 1          # bump when the build changes (the cache key below includes it via build.json)


def _tokenizer(template: str, vocab_words: Sequence[str] | None):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = tuple(dict.fromkeys(SPECIALS + tuple(w.lower() for w in (vocab_words or default_vocab_words()))))
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation("isolated"),
                                                pre_tokenizers.Digits(individual_digits=True)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="<|endoftext|>",
                                   pad_token="<|endoftext|>", additional_special_tokens=["<|im_start|>", "<|im_end|>"],
                                   model_max_length=4096)
    fast.chat_template = CHAT_TEMPLATES[template]
    return fast


def build_tiny_hf(out_dir: str | os.PathLike, *, seed: int = 0, template: str = "chatml",
                  vocab_words: Sequence[str] | None = None) -> Path:
    """Write the tiny model + tokenizer to out_dir and return it. Same args -> same weights and tokenizer
    (config.json carries the transformers version, so bytes match only within one version)."""
    if template not in CHAT_TEMPLATES:
        raise ValueError(f"template must be one of {sorted(CHAT_TEMPLATES)}, got {template!r}")
    import torch
    from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = _tokenizer(template, vocab_words)
    eos, im_end = tok.convert_tokens_to_ids("<|endoftext|>"), tok.convert_tokens_to_ids("<|im_end|>")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=False, eos_token_id=eos, pad_token_id=eos, bos_token_id=None)
    with torch.random.fork_rng(devices=[]):         # no side effect on the caller's RNG
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(cfg).eval()
    model.generation_config = GenerationConfig(eos_token_id=[im_end, eos], pad_token_id=eos, do_sample=False)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "build.json").write_text(json.dumps({"builder": "ag.reference.tiny_hf", "format_version": FORMAT_VERSION,
                                                "seed": seed, "template": template,
                                                "custom_vocab": vocab_words is not None}, sort_keys=True) + "\n")
    return out


def cache_root() -> Path:
    return Path(os.path.expanduser(os.environ.get("AG_CACHE") or "~/.cache/ag"))


def ensure_tiny_hf(template: str = "chatml", *, seed: int = 0) -> Path:
    """Cached build under ${AG_CACHE:-~/.cache/ag}/tiny_hf/<template>-s<seed> (built once, then reused).
    The build goes to a temporary sibling directory and is renamed into place, so concurrent callers
    never see a half-written model; the loser of a race discards its copy."""
    dest = cache_root() / "tiny_hf" / f"{template}-s{seed}"
    stamp = dest / "build.json"
    if stamp.exists():
        try:
            if json.loads(stamp.read_text()).get("format_version") == FORMAT_VERSION:
                return dest
        except ValueError:
            pass
        shutil.rmtree(dest, ignore_errors=True)         # stale or broken build
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    try:
        build_tiny_hf(tmp, seed=seed, template=template)
        try:
            os.rename(tmp, dest)
        except OSError:
            if not stamp.exists():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest
