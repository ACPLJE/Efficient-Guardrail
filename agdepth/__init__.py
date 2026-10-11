"""Asymmetric Guard -- depth-asymmetry proof of concept (plan 2026-10-10 §2.2 row 1, experiments E0.3 / E1.1).

Claim under test: on the SAME backbone, harmful-content windows become linearly separable from benign windows at
shallow layers, while prompt-injection windows only separate from instruction-heavy benign windows at deeper layers.

Pipeline (one command each, see depthctl.py and docs/depth_poc_runbook.md):
    build    -> windows.jsonl.gz      256-token windows: background + one inserted snippet, label = snippet class
    extract  -> feats/<bb>/<framing>/<pool>.f16   per-layer pooled hidden states (embeddings .. last layer)
    fit      -> per-layer logistic probes (harm task, injection task), test scores
    score    -> optional large negative pool scored with the frozen probes (low-FPR regime: 2e-4, 5e-5)
    analyze  -> layer curves, l95 per contrast, bootstrap CI, gate verdict, figures, report.md
"""
