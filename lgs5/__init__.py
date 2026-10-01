"""Stage S5 — evaluation of the trained checkpoints: E1 parity (short prompts), E2-1/E2-2 recall vs length and depth
(<=28k first, 64k/128k in S6), E2-7 three-layer decomposition, dense-FT / sliding / CD baselines, and the tables."""
import os as _os

_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
