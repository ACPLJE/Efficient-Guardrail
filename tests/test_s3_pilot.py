"""E0-5 pilot: all LG runner paths on the dry-run rows with a tiny backbone + tiny tokenizer."""
from __future__ import annotations

import json
import random

from lgs2 import haystack as H
from lgs2 import needles as N
from lgs2 import screen as S
from lgs2 import splits as SP
from lgs2 import synth as SY
from lgs2.tok import HFTok, WhitespaceTok
from lgs3 import pilot as PL
from lgs3 import variants as V
from tests.test_s2_pipeline import TEST_DOMAINS, _fake_screen_rows, _needle_files, root  # noqa: F401
from tests.test_zeroshot import _tiny_repo


def test_pilot_all_paths(root, tmp_path, monkeypatch):
    P = root
    tok_ws = HFTok(str(_tiny_repo(tmp_path)), "tiny")      # one tokenizer for data AND model, as on the cluster (qwen3)
    for d in TEST_DOMAINS:
        H.build_domain(d, P, tok_ws, budget_factor=1.0)
    _fake_screen_rows(P, [d.name for d in TEST_DOMAINS])
    S.summarize(P, n_review=6)
    files = _needle_files(tmp_path / "srcs", random.Random(1))
    monkeypatch.setattr(N, "source_files", lambda src, P_: files.get(src.name, []))
    N.build_pool(P, tok_ws, tok_ws)
    monkeypatch.setattr(SP, "external_texts", lambda P_: {"safetyniah": []})
    SP.run(P)
    monkeypatch.setattr(SY, "DRYRUN_PLAN", {"test": {"pairs": 3, "L": [512, 1024], "positions": [0.0, 0.5, 1.0], "benign_long_per_L": 2},
                                            "train": {"pairs": 2, "L": [512], "positions": "random2", "benign_long_per_L": 0},
                                            "native": 4})
    assert SY.run_dry(P, tok_ws, seed=1, allow_unscreened=False)["validation"]["pass"]
    cfg = json.loads(json.dumps(V.VARIANTS)); cfg["L"] = [512]; cfg["framed"]["pairs"] = 2
    V.run_variant(P, "framed", tok_ws, cfg=cfg, dry_run=True, limit=8)
    rep = PL.run(P, tok_ws, device="cpu", n_attack=2, attack_steps=2)
    assert rep["pass"], rep["checks"]
    assert {"e2_main", "e3_benign_long", "e1_native", "b5_sliding", "e7_framed", "e77_evasion"} <= set(rep["checks"])
    assert rep["summary_rows"] >= 1 and "recall@5pct" in rep["summary_example"]
    rows = [json.loads(l) for l in open(rep["paths"]["e2_main"])]
    assert all(r["hit"] is not None for r in rows if r["label"] == 1)          # needle window found for every harmful row
    assert all(r["n_deep"] <= 8 for r in rows)                                  # k = 8 in the pilot config
    att = [json.loads(l) for l in open(rep["paths"]["e77_evasion"])]
    assert att and all("before" in a and "after" in a and a["theta_scale"] == "route()" for a in att)
