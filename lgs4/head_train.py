"""T1/T3 — LG head training (window-FT): deep layers [ell, N) + probe on window-level presence labels.

    python -m lgs4.head_train --backbone qwen3guard_0.6b --seed 0
    python -m lgs4.head_train --backbone qwen3guard_0.6b --seed 0 --max-steps 200 --limit-rows 500   (smoke)

Data: prepped train/main + train/main16k (presence) and train/native (see lgs4.sampling); windows are 2b = 256 tokens,
positions reset to 0, the trunk [0, ell) frozen (no-grad, identical to the base model), deep layers + probe trained
with BCE (pos_weight = n_neg/n_pos of the epoch, capped).  Output: runs/v2/<backbone>_s<seed>/head.pt (+ head.log.jsonl,
config.json, hashes.json).  Refuses to run on an unsealed lg_data_v2 unless --allow-unsealed (plan: no training before G0).

S8 ablation knobs (every one leaves the default run untouched: a variant trains into runs/v2/<backbone>_<variant>_s<seed>):
    --b 64|256                     window half-size (E6-2; windows 128 / 512 tokens; the router of the arm must use the same b)
    --hard-neg DIR --hard-per-row 4  mined hard negatives (lgs4.hardneg) replace that many random negatives per row (E6-6)
    --init-from PATH               start the deep layers (+ probe) from another head.pt or a dense_ft.pt (E6-6 init / hardneg_ft)
    --max-L 4096|8192              presence rows up to this length only (E6-7 training-length recipe)
    --needle-frac 0.1|0.25|0.5     seeded fraction of the pairs (E6-8 data amount; both rows of a pair stay together)
    --domains a,b                  background domains kept (E6-9 one-domain head)
    --no-twin                      drop the benign twins (E6-9 twin off)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from lgs1.lgcore.model import Head, LGConfig
from lgs4 import backbone as BB
from lgs4 import sampling as SM
from lgs4.common import Checkpointer, JsonlLog, device_auto, gpu_mem_gb, lr_lambda, make_optimizer, run_meta, set_seed, snapshot_sha, window_auprc
from lgs4.config import HEAD_HP, LG_K_DEFAULT, NATIVE_SETS, PRESENCE_SETS, VAL_SETS, WINDOW_B, S4Paths, backbone
from lgs4.prep import PrepSet

ROOT = Path(__file__).resolve().parents[1]


def load_sets(P: S4Paths, sets) -> List[PrepSet]:
    out = []
    for split, subset in sets:
        try:
            out.append(PrepSet(P, split, subset))
        except SystemExit as e:
            print(f"[head] {e}", file=sys.stderr)
    return out


@torch.no_grad()
def eval_val(model, head, val_sets: List[PrepSet], ell: int, pad_id: int, device: str, max_rows: int = 400,
             micro: int = 32, grad_ckpt: bool = False, b: int = WINDOW_B) -> Dict:
    """Window-level AUPRC on val/main: needle windows (harmful vs benign twin) and needle+random negatives."""
    import random
    rng = random.Random(0)
    units: List[SM.Unit] = []
    for si, ps in enumerate(val_sets):
        rows = list(range(len(ps)))
        rng.shuffle(rows)
        for r in rows[:max_rows]:
            units += SM.row_units(ps, si, r, rng, neg_per_row=2, near_neg=1, b=b)
    if not units:
        return {"n": 0}
    model.eval()
    logits: List[float] = []
    for chunk in SM.batches(units, micro):
        ids, valid, _ = SM.collate(chunk, val_sets, pad_id, b)
        lg = BB.lg_window_logits(model, head, ids.to(device), valid.to(device), ell, grad_ckpt=False)
        logits += lg.tolist()
    labels = [u.label for u in units]
    needle = [(l, u.label) for l, u in zip(logits, units) if u.kind == "needle"]
    res = {"n": len(units), "auprc_all": window_auprc(logits, labels),
           "auprc_needle_vs_twin": window_auprc([x for x, _ in needle], [y for _, y in needle]),
           "acc": sum(1 for l, y in zip(logits, labels) if (l > 0) == (y == 1)) / len(units)}
    model.train()
    return res


def init_from_checkpoint(model, head: Head, bb, ell: int, path: Path) -> Dict:
    """E6-6: initialise the trainable deep layers (+ the probe when its type matches) from another run's head.pt, or the
    deep layers from a dense_ft.pt (its layers >= ell; the dense probe is a different reading and is not copied)."""
    ck = BB.load_ckpt(path)
    kind = ck.get("kind")
    if bb.mode != "full":
        raise SystemExit(f"--init-from is implemented for full-FT backbones (got mode={bb.mode})")
    if kind == "lg_head":
        if int(ck.get("ell", ell)) != int(ell):
            raise SystemExit(f"{path}: head trained with ell={ck.get('ell')}, this run uses ell={ell}")
        n = BB.apply_trainable_state(model, bb, ck["trained"], ell)
        probe_ok = ck.get("head_probe", "last") == head.probe
        if probe_ok:
            head.load_state_dict(ck["probe"])
        return {"kind": kind, "tensors": n, "probe": probe_ok, "path": str(path)}
    if kind == "dense_ft":
        sd = {k: v for k, v in ck["trained"].items()
              if k.startswith("norm.") or (k.startswith("layers.") and int(k.split(".", 2)[1]) >= int(ell))}
        n = BB.apply_trainable_state(model, bb, sd, ell)
        return {"kind": kind, "tensors": n, "probe": False, "path": str(path)}
    raise SystemExit(f"{path}: unknown checkpoint kind {kind!r} (lg_head | dense_ft)")


def train(P: S4Paths, bb_name: str, seed: int, hp: Optional[Dict] = None, device: str = "auto", max_steps: Optional[int] = None,
          limit_rows: Optional[int] = None, repo_override: Optional[str] = None, allow_unsealed: bool = False,
          ell_override: Optional[int] = None, val_max_rows: int = 400, dtype: torch.dtype = torch.bfloat16,
          variant: str = "") -> Dict:
    hp = dict(HEAD_HP, **(hp or {}))
    bb = backbone(bb_name)
    if not P.sealed() and not allow_unsealed:
        raise SystemExit(f"{P.data.root}/SEALED missing: lg_data_v2 is not sealed (G0). Training before the seal is not allowed "
                         f"(pass --allow-unsealed for a smoke run only).")
    device = device_auto(device)
    run_dir = P.run_dir(bb_name, seed, variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    head_path = run_dir / BB.HEAD_FILE
    if head_path.exists():
        prev = BB.load_ckpt(head_path)
        if prev.get("meta", {}).get("complete"):
            print(f"[head] {head_path} already complete ({prev['meta'].get('steps')} steps) -> skip (delete it to retrain)")
            return {"skipped": True, "path": str(head_path)}
    set_seed(seed)
    presence, native = load_sets(P, PRESENCE_SETS), load_sets(P, NATIVE_SETS)
    if not presence:
        raise SystemExit("no presence sets prepared (python s4ctl.py prep)")
    val_sets = load_sets(P, [VAL_SETS[0]])
    model, tok = BB.load_backbone(repo_override or bb, device=device, dtype=dtype)
    ell = ell_override or bb.ell
    if ell >= BB.n_layers(model):
        raise SystemExit(f"ell={ell} >= n_layers={BB.n_layers(model)}")
    pad_id = BB.pad_id_of(model, tok)
    d = BB.hidden_size(model)
    groups = BB.prepare_lg_trainable(model, bb, ell)
    set_seed(seed)                                    # probe init depends on the seed only
    head = Head(d, probe=hp.get("probe", "last")).to(device)
    init_info = init_from_checkpoint(model, head, bb, ell, Path(hp["init_from"])) if hp.get("init_from") else None
    n_train = BB.count_params(groups["lr_deep"]) + BB.count_params(groups["lr_lora"]) + BB.count_params(head.parameters())
    b = int(hp.get("b") or WINDOW_B)
    cfg = LGConfig(b=b, ell=ell, k=LG_K_DEFAULT, pool="max", tau=0.5, pad_id=pad_id)
    # ---- S8 recipe knobs: presence-row filter, mined hard negatives ----
    row_filter = SM.make_row_filter(hp.get("max_L"), float(hp.get("needle_frac", 1.0) or 1.0), hp.get("domains"), bool(hp.get("twin", True)), seed)
    hard = None
    if hp.get("hard_neg"):
        hard = SM.HardNegIndex.load(Path(hp["hard_neg"]), [SM.set_name(ps) for ps in presence])
        for ps in presence:
            nm = SM.set_name(ps)
            if hard.n_rows(nm) and hard.n_rows(nm) != len(ps):
                raise SystemExit(f"{hp['hard_neg']}/{nm}.npz: {hard.n_rows(nm)} rows mined, prep has {len(ps)} (re-mine)")
        if int(hard.meta.get("b", b)) != b:
            raise SystemExit(f"hard negatives were mined with b={hard.meta.get('b')}, this run uses b={b}")
    hard_per_row = int(hp.get("hard_per_row", 0) or 0) if hard is not None else 0
    # ---- epoch plan ----
    ep_units = [SM.epoch_units(presence, native, seed, e, hp["neg_per_row"], hp["near_neg"], hp["native_ratio"], limit_rows, b=b,
                               row_filter=row_filter, hard=hard, hard_per_row=hard_per_row)
                for e in range(hp["epochs"])]
    kinds0 = {}
    for u in ep_units[0]:
        kinds0[u.kind] = kinds0.get(u.kind, 0) + 1
    steps_per_epoch = [max(1, -(-len(u) // hp["batch_windows"])) for u in ep_units]
    total_steps = sum(steps_per_epoch)
    if max_steps:
        total_steps = min(total_steps, max_steps)
    n_pos = sum(1 for u in ep_units[0] if u.label == 1)
    n_neg = max(1, len(ep_units[0]) - n_pos)
    pos_weight = min(hp["pos_weight_cap"], n_neg / max(1, n_pos)) if hp["pos_weight"] == "auto" else float(hp["pos_weight"])
    print(f"[head] {bb_name} s{seed} ell={ell} d={d} b={b} mode={bb.mode} probe={head.probe} variant={variant or '-'} trainable={n_train:,} "
          f"device={device}" + (f" init_from={init_info['path']} ({init_info['kind']}, {init_info['tensors']} tensors, probe={init_info['probe']})" if init_info else ""),
          flush=True)
    print(f"[head] rows: presence={sum(len(p) for p in presence):,} native={sum(len(p) for p in native):,} | epoch0 units={len(ep_units[0]):,} "
          f"(pos {n_pos:,}; kinds {kinds0}) batch={hp['batch_windows']} micro={bb.micro_windows} steps/epoch={steps_per_epoch[0]:,} total={total_steps:,} "
          f"pos_weight={pos_weight:.2f}" + (f" filters: max_L={hp.get('max_L')} needle_frac={hp.get('needle_frac', 1.0)} domains={hp.get('domains')} "
                                             f"twin={hp.get('twin', True)}" if row_filter else "") +
          (f" hard_neg={hp['hard_neg']} x{hard_per_row}/row" if hard is not None else ""), flush=True)
    opt = make_optimizer(groups, list(head.parameters()), hp)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total_steps, hp["warmup_frac"], hp["schedule"]))
    # ---- resume (mid-run checkpoint every ckpt_every_sec seconds and on SIGTERM; see backbone.save_resume) ----
    resume_path = run_dir / ("head" + BB.RESUME_SUFFIX)
    # a resume checkpoint is ~12 bytes/param (fp32 weights + Adam): 3.6 GB for the 0.6B head, ~29 GB for the 4B ->
    # scale the interval with the trainable size (0.6B head: 30 min, 4B head: ~4 h) to keep the shared FS sane
    ck = Checkpointer(hp.get("ckpt_every_sec", 1800) * max(1.0, n_train / 3e8))
    step, start_epoch, start_bi = 0, 0, 0
    if resume_path.exists():
        st = BB.load_resume(resume_path, model, bb, ell, head, opt, sched)
        step, start_epoch, start_bi = int(st["step"]), int(st["epoch"]), int(st["batch_idx"]) + 1
        if st.get("total_steps") != total_steps:
            raise SystemExit(f"{resume_path}: total_steps {st.get('total_steps')} != {total_steps} (data or hp changed) -> delete it to restart")
        print(f"[head] resumed from {resume_path.name}: step {step:,}/{total_steps:,} (epoch {start_epoch}, batch {start_bi:,}, "
              f"saved {st.get('saved_at')})", flush=True)
    log = JsonlLog(run_dir / "head.log.jsonl")
    meta = run_meta(ROOT, {"backbone": bb_name, "seed": seed, "variant": variant, "ell": ell, "b": b, "n_layers": BB.n_layers(model),
                           "tokens_per_step": hp["batch_windows"] * 2 * b, "model_sha": snapshot_sha(repo_override or bb.repo),
                           "prep": {f"{p.split}/{p.subset}": p.info for p in presence + native},
                           "pos_weight": pos_weight, "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
                           "trainable_params": n_train, "epoch0_kinds": kinds0, "init_from": init_info,
                           "recipe": {"max_L": hp.get("max_L"), "needle_frac": hp.get("needle_frac", 1.0), "domains": hp.get("domains"),
                                      "twin": hp.get("twin", True), "hard_neg": hp.get("hard_neg"), "hard_per_row": hard_per_row,
                                      "hard_meta": (hard.meta if hard is not None else None)},
                           "complete": False})
    (run_dir / "config.json").write_text(json.dumps({"backbone": bb.__dict__, "cfg": cfg.to_dict(), "hp": hp, "meta": meta}, indent=2, default=str),
                                         encoding="utf-8")
    pw = torch.tensor(pos_weight, device=device)
    sets = presence + native
    model.train()
    t0 = time.time()
    t_log, step_log = time.time(), step
    run_loss, run_n, run_correct = 0.0, 0, 0
    last_val: Dict = {}
    done = False

    def save_resume(epoch_: int, bi_: int) -> None:
        BB.save_resume(resume_path, model, bb, ell, head, opt, sched,
                       {"step": step, "epoch": epoch_, "batch_idx": bi_, "total_steps": total_steps, "seconds": round(time.time() - t0, 1)})
        ck.mark()
        print(f"   [ckpt] step {step:,} -> {resume_path.name}", flush=True)

    for epoch in range(start_epoch, hp["epochs"]):
        for bi, batch in enumerate(SM.batches(ep_units[epoch], hp["batch_windows"])):
            if epoch == start_epoch and bi < start_bi:
                continue
            opt.zero_grad(set_to_none=True)
            n_micro = -(-len(batch) // bb.micro_windows)
            for mi in range(n_micro):
                chunk = batch[mi * bb.micro_windows:(mi + 1) * bb.micro_windows]
                ids, valid, y = SM.collate(chunk, sets, pad_id, b)
                ids, valid, y = ids.to(device), valid.to(device), y.to(device)
                logits = BB.lg_window_logits(model, head, ids, valid, ell, grad_ckpt=bb.grad_ckpt)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw) * (len(chunk) / len(batch))
                loss.backward()
                run_loss += float(loss.detach()) * len(batch)
                run_n += len(chunk)
                run_correct += int(((logits > 0).float() == y).sum())
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], hp["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            if step % hp["log_every"] == 0 or step == total_steps:
                sps = (time.time() - t_log) / max(1, step - step_log)
                t_log, step_log = time.time(), step
                rec = {"step": step, "epoch": epoch, "loss": run_loss / max(1, run_n), "acc": run_correct / max(1, run_n),
                       "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1), "gpu_gb": gpu_mem_gb(),
                       "sec_per_step": round(sps, 3), "eta_h": round((total_steps - step) * sps / 3600, 2), "total": total_steps}
                log.write(**rec)
                print(f"   step {step}/{total_steps} loss={rec['loss']:.4f} acc={rec['acc']:.3f} lr={rec['lr']:.2e} {rec['sec']}s "
                      f"({sps:.2f} s/step, eta {rec['eta_h']} h, gpu {rec['gpu_gb']} GB)", flush=True)
                run_loss, run_n, run_correct = 0.0, 0, 0
            if val_sets and hp["eval_every"] and (step % hp["eval_every"] == 0 or step == total_steps):
                last_val = eval_val(model, head, val_sets, ell, pad_id, device, max_rows=val_max_rows, micro=bb.micro_windows, b=b)
                log.write(step=step, val=last_val)
                print(f"   val@{step}: {last_val}", flush=True)
            if step >= total_steps:
                done = True
                break
            if ck.due():
                save_resume(epoch, bi)
                if ck.stop:
                    print(f"[head] signal {ck.signal_name}: checkpoint saved at step {step:,}; exiting rc=3 -- resubmit "
                          f"(python s4ctl.py all --sbatch) to resume", flush=True)
                    log.close()
                    sys.exit(3)
        if done:
            break
    if val_sets and not last_val:
        last_val = eval_val(model, head, val_sets, ell, pad_id, device, max_rows=val_max_rows, micro=bb.micro_windows, b=b)
    meta.update(steps=step, seconds=round(time.time() - t0, 1), val=last_val, complete=True, gpu_gb=gpu_mem_gb())
    model.eval()
    BB.save_head_ckpt(head_path, model, head, bb, cfg, hp, meta)
    BB.write_hashes(run_dir, {"data_manifest_sha": {k: v.get("data_manifest_sha") for k, v in meta["prep"].items()}})
    if resume_path.exists():
        resume_path.unlink()
    log.write(step=step, done=True, seconds=meta["seconds"])
    log.close()
    print(f"[head] saved {head_path} ({step} steps, {meta['seconds']}s) val={last_val}", flush=True)
    return {"path": str(head_path), "steps": step, "val": last_val, "seconds": meta["seconds"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S4 LG head training")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repo", default=None, help="override the backbone repo/dir (tests)")
    ap.add_argument("--allow-unsealed", action="store_true")
    ap.add_argument("--ell", type=int, default=None)
    ap.add_argument("--probe", default=None, choices=[None, "last", "landmark"], help="head probe (default HEAD_HP.probe = last)")
    ap.add_argument("--variant", default="", help="run-dir suffix: runs/<backbone>_<variant>_s<seed> (e.g. lm)")
    # S8 ablation knobs
    ap.add_argument("--b", type=int, default=None, help="window half-size (E6-2; default 128)")
    ap.add_argument("--hard-neg", default=None, help="dir of mined hard negatives (lgs4.hardneg), E6-6")
    ap.add_argument("--hard-per-row", type=int, default=None, help="mined negatives per presence row (default 4 when --hard-neg)")
    ap.add_argument("--init-from", default=None, help="head.pt / dense_ft.pt to initialise the deep layers (+ probe) from")
    ap.add_argument("--max-L", type=int, default=None, help="presence rows up to this L_target only (E6-7)")
    ap.add_argument("--needle-frac", type=float, default=None, help="seeded fraction of the pairs kept (E6-8)")
    ap.add_argument("--domains", default=None, help="comma list of background domains kept (E6-9)")
    ap.add_argument("--no-twin", action="store_true", help="drop the benign twins (E6-9)")
    ap.add_argument("--neg-per-row", type=int, default=None)
    ap.add_argument("--near-neg", type=int, default=None)
    a = ap.parse_args(argv)
    P = S4Paths.load().ensure().for_backbone(a.backbone)      # prep arrays of the backbone's tokenizer family (T5: llama3)
    hp = {"epochs": a.epochs} if a.epochs else {}
    if a.probe:
        hp["probe"] = a.probe
    for key, val in (("b", a.b), ("hard_neg", a.hard_neg), ("init_from", a.init_from), ("max_L", a.max_L), ("needle_frac", a.needle_frac),
                     ("neg_per_row", a.neg_per_row), ("near_neg", a.near_neg)):
        if val is not None:
            hp[key] = val
    if a.hard_neg:
        hp["hard_per_row"] = a.hard_per_row if a.hard_per_row is not None else 4
    if a.domains:
        hp["domains"] = [x for x in a.domains.split(",") if x]
    if a.no_twin:
        hp["twin"] = False
    train(P, a.backbone, a.seed, hp, a.device, a.max_steps, a.limit_rows, a.repo, a.allow_unsealed, a.ell, variant=a.variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
