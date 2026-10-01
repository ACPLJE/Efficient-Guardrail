"""Stage S4 — training: LG head (window-FT) + router (head-distillation), dense-FT baseline, E0-7 FLOPs check."""
import os as _os

# fewer fragmentation OOMs for the fp32-master + Adam training runs (must be set before the first CUDA allocation)
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
