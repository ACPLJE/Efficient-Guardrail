"""Minimal LoRA (no peft dependency; the compute nodes are offline).

    inject_lora(model, layer_indices, r=16, alpha=32, targets=(...))   -> list of LoRALinear modules
    lora_parameters(model)                                             -> trainable A/B tensors
    lora_state_dict(model) / load_lora_state(model, sd)                -> adapter-only checkpoints
The wrapped Linear keeps its frozen weight; forward = base(x) + (alpha/r) * B(A(dropout(x))).
B is zero-initialised so an untrained adapter is an identity (E0-1 equivalence still holds at load).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A.to(base.weight.device)
        self.lora_B.to(base.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        d = self.lora_B(self.lora_A(self.drop(x.to(self.lora_A.weight.dtype))))
        return out + d.to(out.dtype) * self.scaling


def _decoder_layers(model) -> nn.ModuleList:
    dec = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
    return dec.layers


def inject_lora(model, layer_indices: Iterable[int], r: int = 16, alpha: int = 32, dropout: float = 0.05,
                targets: Sequence[str] = DEFAULT_TARGETS) -> List[LoRALinear]:
    layers = _decoder_layers(model)
    made: List[LoRALinear] = []
    for li in layer_indices:
        layer = layers[li]
        for name, mod in list(layer.named_modules()):
            leaf = name.split(".")[-1]
            if leaf in targets and isinstance(mod, nn.Linear):
                parent = layer.get_submodule(".".join(name.split(".")[:-1])) if "." in name else layer
                wrapped = LoRALinear(mod, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, leaf, wrapped)
                made.append(wrapped)
    return made


def lora_modules(model) -> Dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_parameters(model) -> List[nn.Parameter]:
    out: List[nn.Parameter] = []
    for m in lora_modules(model).values():
        out += list(m.lora_A.parameters()) + list(m.lora_B.parameters())
    return out


def lora_state_dict(model) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for n, m in lora_modules(model).items():
        sd[f"{n}.lora_A.weight"] = m.lora_A.weight.detach().cpu()
        sd[f"{n}.lora_B.weight"] = m.lora_B.weight.detach().cpu()
    return sd


def load_lora_state(model, sd: Dict[str, torch.Tensor]) -> int:
    mods = lora_modules(model)
    n = 0
    for name, m in mods.items():
        a, b = sd.get(f"{name}.lora_A.weight"), sd.get(f"{name}.lora_B.weight")
        if a is None or b is None:
            raise KeyError(f"LoRA tensors for {name} missing in checkpoint")
        with torch.no_grad():
            m.lora_A.weight.copy_(a.to(m.lora_A.weight.device, m.lora_A.weight.dtype))
            m.lora_B.weight.copy_(b.to(m.lora_B.weight.device, m.lora_B.weight.dtype))
        n += 1
    return n
