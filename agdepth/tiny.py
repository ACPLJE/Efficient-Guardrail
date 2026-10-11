"""A tiny random Qwen3 + BPE tokenizer saved as a local 'repo' directory (offline smoke run and tests)."""
from __future__ import annotations

import random
from pathlib import Path

from . import synth


def make_tiny_repo(dst: Path, n_layers: int = 4, hidden: int = 64, seed: int = 0) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from lgs1.lgcore.model import tiny_qwen3
    dst = Path(dst)
    if (dst / "config.json").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    corpus = [synth.smoke_sentence(rng) for _ in range(400)]
    corpus += [t[2] for t in synth.INSTR_TEMPLATES]
    corpus += ["SMOKEHARM SMOKEREQ SMOKEINJ SMOKENOTINJ document user assistant summarize"] * 20
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=500, special_tokens=["<unk>", "<pad>", "<eos>", "<|im_start|>", "<|im_end|>"])
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    fast.save_pretrained(dst)
    model = tiny_qwen3(seed=seed, n_layers=n_layers, hidden=hidden, vocab=512, attn="eager")
    model.config.pad_token_id = fast.pad_token_id
    model.config.eos_token_id = fast.eos_token_id
    model.save_pretrained(dst)
    return dst
