"""D5b — splits and the leakage audit (report §5.6; plan S2 'D5 분할·감사').

    python -m lgs2.splits                # -> splits/needles.json, splits/haystack.json, splits/audit.json

Keys: needle GROUP (exact + near-dup cluster) and background DOC id.  Rules
  * public test sources (pool=test) are test only; train-pool groups -> val / calib / train by a seeded hash
  * a train-pool needle whose cluster contains a test-pool needle or an external long-context needle
    (SafetyNIAH origin_prompt/origin_response, LongSafetyBench HarmfulNIAH reference) is REMOVED
    (exact and near-duplicate counts recorded separately)
  * train-domain background pieces -> test / calib / val / train by token share; OOD domains -> test only;
    only pieces that passed screening are assigned (before screening the file is marked provisional)
Gate checks (E0-4 subset that S2 can already evaluate) are written to audit.json["gates"].
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from lgs1.zeroshot.datasets import _hub_cache_dir, _snapshot_dir
from .config import DOMAINS, EXTERNAL_AUDIT, MINHASH, SPLITS, TEST_SOURCES, S2Paths
from .haystack import load_docs
from .needles import load_pool
from .screen import load_keep
from .textutil import exact_key, near_dup_clusters, word_shingles

NEEDLE_SPLITS = ("train", "val", "calib", "test")


def _u(seed: int, key: str) -> float:
    h = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


# ----------------------------------------------------------------------------------------------
# needle splits
# ----------------------------------------------------------------------------------------------
def assign_needle_splits(pool: List[Dict], fractions: Dict[str, float] = SPLITS["needle_group_fractions"],
                         seed: int = SPLITS["seed"]) -> Dict[str, str]:
    """group_id -> split.  Test-pool groups are 'test'; train-pool groups hash into val / calib / train."""
    groups: Dict[str, Set[str]] = defaultdict(set)
    for r in pool:
        groups[r["group_id"]].add(r["pool"])
    out: Dict[str, str] = {}
    v, c = fractions.get("val", 0.0), fractions.get("calib", 0.0)
    for g, pools in groups.items():
        if "test" in pools:
            out[g] = "test"          # train members of a mixed group are removed by the audit
            continue
        u = _u(seed, g)
        out[g] = "val" if u < v else ("calib" if u < v + c else "train")
    return out


# ----------------------------------------------------------------------------------------------
# external needle sets (audit only)
# ----------------------------------------------------------------------------------------------
def _parquet_files(registry: str, hf_repo: str, P: S2Paths, file_glob: str = "*.parquet") -> List[Path]:
    roots = [x for x in (_snapshot_dir(registry), _hub_cache_dir(hf_repo)) if x]
    for root in roots:
        files = sorted(glob.glob(str(root / "**" / file_glob), recursive=True))
        if files:
            return [Path(f) for f in files]
    return []


def external_texts(P: S2Paths) -> Dict[str, List[str]]:
    """{external set name: unique needle texts}.  Missing sets are reported (empty list), not fatal."""
    import pyarrow.parquet as pq
    out: Dict[str, List[str]] = {}
    for name, spec in EXTERNAL_AUDIT.items():
        files = _parquet_files(spec["registry"], spec["hf_repo"], P, spec.get("file_glob", "*.parquet"))
        texts: Set[str] = set()
        for f in files:
            pf = pq.ParquetFile(f)
            cols = [c for c in spec["columns"] if c in pf.schema_arrow.names]
            if not cols:
                continue
            for batch in pf.iter_batches(batch_size=2048, columns=cols):
                for col in batch.columns:
                    for v in col.to_pylist():
                        if v and str(v).strip():
                            texts.add(str(v).strip())
        out[name] = sorted(texts)
    return out


# ----------------------------------------------------------------------------------------------
# audit: remove train-pool needles that collide with test-pool / external needles
# ----------------------------------------------------------------------------------------------
def audit_overlap(pool: List[Dict], external: Dict[str, List[str]], group_split: Dict[str, str]) -> Tuple[Dict[str, Dict], Dict]:
    """Returns (removed: needle id -> {reason, against, match}, counts)."""
    train_rows = [r for r in pool if r["pool"] == "train"]
    test_rows = [r for r in pool if r["pool"] == "test"]
    texts: List[str] = [r["text"] for r in train_rows]
    tags: List[Tuple[str, str]] = [("train", r["id"]) for r in train_rows]
    for r in test_rows:
        texts.append(r["text"])
        tags.append(("test:" + r["source"], r["id"]))
    for name, lst in external.items():
        for i, t in enumerate(lst):
            texts.append(t)
            tags.append((f"external:{name}", f"{name}:{i}"))
    keys = [exact_key(t) for t in texts]
    shingles = [word_shingles(t, MINHASH["shingle_words"]) for t in texts]
    roots = near_dup_clusters(texts, n=MINHASH["shingle_words"], num_perm=MINHASH["num_perm"], bands=MINHASH["bands"],
                              threshold=MINHASH["jaccard"], seed=MINHASH["seed"], shingle_cache=shingles)
    by_root: Dict[int, List[int]] = defaultdict(list)
    for i, root in enumerate(roots):
        by_root[root].append(i)
    by_key: Dict[str, List[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    removed: Dict[str, Dict] = {}
    counts: Dict[str, Counter] = {"exact": Counter(), "near": Counter()}
    for i, (kind, nid) in enumerate(tags):
        if kind != "train":
            continue
        # exact collisions first
        against = [tags[j][0] for j in by_key[keys[i]] if tags[j][0] != "train"]
        if against:
            removed[nid] = {"reason": "exact", "against": sorted(set(against)), "match": tags[[j for j in by_key[keys[i]] if tags[j][0] != "train"][0]][1]}
            for a in set(against):
                counts["exact"][a] += 1
            continue
        near = [j for j in by_root[roots[i]] if tags[j][0] != "train"]
        if near:
            against = sorted({tags[j][0] for j in near})
            removed[nid] = {"reason": "near_dup", "against": against, "match": tags[near[0]][1]}
            for a in against:
                counts["near"][a] += 1
    summary = {"n_train_pool": len(train_rows), "n_test_pool": len(test_rows),
               "external_sizes": {k: len(v) for k, v in external.items()},
               "n_removed": len(removed), "n_removed_exact": sum(1 for v in removed.values() if v["reason"] == "exact"),
               "n_removed_near_dup": sum(1 for v in removed.values() if v["reason"] == "near_dup"),
               "removed_exact_by_set": dict(counts["exact"]), "removed_near_dup_by_set": dict(counts["near"])}
    return removed, summary


# ----------------------------------------------------------------------------------------------
# haystack splits
# ----------------------------------------------------------------------------------------------
def assign_haystack_splits(P: S2Paths, fractions: Dict[str, float] = SPLITS["haystack_fractions"],
                           seed: int = SPLITS["seed"]) -> Tuple[Dict[str, str], Dict, bool]:
    """doc_id -> split (train domains: test/calib/val/train by hash; OOD: test).  Returns (assign, counts, screened)."""
    assign: Dict[str, str] = {}
    counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    screened_all = True
    t, c, v = fractions.get("test", 0.3), fractions.get("calib", 0.15), fractions.get("val", 0.05)
    for dom in DOMAINS:
        keep = load_keep(P, dom.name)
        if keep is None:
            screened_all = False
        cnt: Dict[str, Dict[str, int]] = defaultdict(lambda: {"docs": 0, "tokens": 0})
        for rec in load_docs(P, dom.name, keep=keep):
            if dom.role == "ood":
                sp = "test"
            else:
                u = _u(seed, rec["id"])
                sp = "test" if u < t else ("calib" if u < t + c else ("val" if u < t + c + v else "train"))
            assign[rec["id"]] = sp
            cnt[sp]["docs"] += 1
            cnt[sp]["tokens"] += rec["n_tok"]
        counts[dom.name] = {k: dict(v2) for k, v2 in cnt.items()}
        counts[dom.name]["_screened"] = keep is not None
    return assign, counts, screened_all


# ----------------------------------------------------------------------------------------------
# gates (the part of E0-4 that S2 can evaluate before synthesis)
# ----------------------------------------------------------------------------------------------
def gate_checks(pool: List[Dict], needle_split: Dict[str, str], removed: Dict[str, Dict], hay_assign: Dict[str, str]) -> Dict:
    g_by_split: Dict[str, Set[str]] = defaultdict(set)
    test_src_in_train = 0
    for r in pool:
        sp = needle_split.get(r["id"])
        if sp in ("train", "val", "calib"):
            g_by_split[sp].add(r["group_id"])
            if r["source"] in TEST_SOURCES:
                test_src_in_train += 1
        elif sp == "test":
            g_by_split["test"].add(r["group_id"])
    inter = {f"{a}&{b}": len(g_by_split[a] & g_by_split[b]) for a in NEEDLE_SPLITS for b in NEEDLE_SPLITS if a < b}
    checks = {
        "needle_group_intersections": inter,
        "needle_groups_disjoint": all(v == 0 for v in inter.values()),
        "test_source_rows_in_train_val_calib": test_src_in_train,
        "external_overlap_removed": len(removed),
        "haystack_doc_single_split": len(hay_assign) == len(set(hay_assign)),   # a dict key is unique by construction
    }
    checks["pass"] = checks["needle_groups_disjoint"] and test_src_in_train == 0
    return checks


def run(P: S2Paths, seed: int = SPLITS["seed"]) -> Dict:
    P.ensure()
    pool = load_pool(P)
    group_split = assign_needle_splits(pool, seed=seed)
    external = external_texts(P)
    removed, overlap = audit_overlap(pool, external, group_split)
    needle_split: Dict[str, str] = {}
    for r in pool:
        if r["id"] in removed:
            needle_split[r["id"]] = "removed"
        elif r["pool"] == "test":
            needle_split[r["id"]] = "test"
        else:
            needle_split[r["id"]] = group_split[r["group_id"]]
    hay_assign, hay_counts, screened = assign_haystack_splits(P, seed=seed)
    gates = gate_checks(pool, needle_split, removed, hay_assign)
    counts = Counter((needle_split[r["id"]], r["source"], r["label"]) for r in pool)
    table: Dict[str, Dict] = {}
    for (sp, src, lab), n in sorted(counts.items()):
        table.setdefault(sp, {}).setdefault(src, {})[lab] = n
    usable = Counter((needle_split[r["id"]], r["label"]) for r in pool if r["n_tok_qwen3"] <= 128)
    needles_json = {"seed": seed, "fractions": SPLITS["needle_group_fractions"], "assign": needle_split,
                    "group_split": group_split, "counts_by_split_source_label": table,
                    "usable_le128_by_split_label": {f"{k[0]}/{k[1]}": v for k, v in sorted(usable.items())},
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    haystack_json = {"seed": seed, "fractions": SPLITS["haystack_fractions"], "screened": screened,
                     "provisional": not screened, "assign": hay_assign, "counts": hay_counts,
                     "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    audit = {"overlap": overlap, "removed": removed, "gates": gates, "haystack_screened": screened,
             "needle_counts": table, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (P.splits / "needles.json").write_text(json.dumps(needles_json, ensure_ascii=False), encoding="utf-8")
    (P.splits / "haystack.json").write_text(json.dumps(haystack_json, ensure_ascii=False), encoding="utf-8")
    (P.splits / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"needle_counts": table, "overlap": overlap, "gates": gates, "haystack": hay_counts, "screened": screened,
            "usable": needles_json["usable_le128_by_split_label"]}


def load_splits(P: S2Paths) -> Tuple[Dict, Dict]:
    n = json.loads((P.splits / "needles.json").read_text(encoding="utf-8"))
    h = json.loads((P.splits / "haystack.json").read_text(encoding="utf-8"))
    return n, h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D5b splits + audit")
    ap.add_argument("--seed", type=int, default=SPLITS["seed"])
    a = ap.parse_args(argv)
    P = S2Paths.load()
    r = run(P, a.seed)
    ov = r["overlap"]
    print(f"[splits] external sets: {ov['external_sizes']}")
    print(f"[splits] removed from train pool: {ov['n_removed']} (exact {ov['n_removed_exact']}, near-dup {ov['n_removed_near_dup']})")
    for sp in ("train", "val", "calib", "test", "removed"):
        srcs = r["needle_counts"].get(sp, {})
        n_h = sum(v.get("harmful", 0) for v in srcs.values())
        n_b = sum(v.get("benign", 0) for v in srcs.values())
        print(f"[splits] {sp:8s} harmful={n_h:7,d} benign={n_b:7,d}  usable<=128: "
              f"h={r['usable'].get(sp + '/harmful', 0):,} b={r['usable'].get(sp + '/benign', 0):,}")
    for d, c in r["haystack"].items():
        parts = " ".join(f"{k}={v['docs']}/{v['tokens'] // 1000}k" for k, v in c.items() if not k.startswith("_"))
        print(f"[splits] haystack {d:10s} {'screened' if c.get('_screened') else 'PROVISIONAL(unscreened)'}: {parts}")
    print(f"[splits] gates: {'PASS' if r['gates']['pass'] else 'FAIL'} {r['gates']}")
    print(f"[written] {P.splits}/needles.json haystack.json audit.json")
    return 0 if r["gates"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
