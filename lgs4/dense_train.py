"""T2 — dense-FT baseline: the same backbone fine-tuned full-context on the same rows, same number of optimizer steps.

    python -m lgs4.dense_train --backbone qwen3guard_0.6b --seed 0            # steps = head.pt of the same run (if present)
    python -m lgs4.dense_train --backbone qwen3guard_0.6b --seed 0 --steps 200 --limit-rows 300   (smoke)

Rows: train/main + train/main16k (presence, whole document, label = row label) + train/native at the same native_ratio as
the LG head.  Context cap Backbone.dense_max_tokens (0.6B 16k, 8B 8k); rows are batched under a padded-token budget
(Backbone.dense_micro_tokens) with gradient accumulation to `rows_per_step` rows per optimizer step.  Probe = LayerNorm +
Linear on the last valid token of the full causal forward (HF).  Output runs/v2/<bb>_s<seed>/dense_ft.pt.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from lgs1.lgcore.model import Head
from lgs4 import backbone as BB
from lgs4 import sampling as SM
from lgs4.common import Checkpointer, JsonlLog, device_auto, gpu_mem_gb, lr_lambda, make_optimizer, run_meta, set_seed, snapshot_sha, window_auprc
from lgs4.config import DENSE_HP, HEAD_HP, NATIVE_SETS, PRESENCE_SETS, VAL_SETS, S4Paths, backbone
from lgs4.prep import PrepSet

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def eval_val(model, head, val: PrepSet, pad_id: int, device: str, max_tokens: int, micro_tokens: int, max_rows: int = 200) -> Dict:
    rng = random.Random(0)
    rows = list(range(len(val)))
    rng.shuffle(rows)
    units = [SM.RowUnit(0, r, val.meta[r]["label"], min(val.meta[r]["n_tok"], max_tokens)) for r in rows[:max_rows]]
    model.eval()
    logits: List[float] = []
    labels: List[int] = []
    for chunk in SM.token_budget_batches(units, micro_tokens):
        ids, valid, y = SM.collate_rows(chunk, [val], pad_id, max_tokens)
        lg = BB.dense_logits(model, head, ids.to(device), valid.to(device))
        logits += lg.tolist()
        labels += [int(v) for v in y.tolist()]
    model.train()
    return {"n": len(units), "auprc": window_auprc(logits, labels),
            "acc": sum(1 for l, y in zip(logits, labels) if (l > 0) == (y == 1)) / max(1, len(units))}


def budget_steps(head_meta: Dict, hp: Dict, bb, presence, native, seed: int, limit_rows: Optional[int], rows_per_step: int):
    """Optimizer steps for dense-FT so that its training budget matches the LG head's (DENSE_HP.budget)."""
    from lgs4.config import HEAD_HP, WINDOW_B
    from lgs4 import backbone as _BB
    head_steps = int(head_meta["steps"])
    head_tok_per_step = int(head_meta.get("tokens_per_step") or HEAD_HP["batch_windows"] * 2 * WINDOW_B)   # older head.pt: default hp
    head_tokens = head_steps * head_tok_per_step
    rows = SM.epoch_rows(presence, native, seed, 0, hp["native_ratio"], bb.dense_max_tokens, limit_rows)
    mean_tok = sum(r.n_tok for r in rows) / max(1, len(rows))
    dense_tok_per_step = rows_per_step * mean_tok
    mode = hp.get("budget", "tokens")
    if mode == "steps":
        steps = head_steps
        detail = f"same optimizer steps as the head ({head_steps:,})"
    elif mode == "flops":
        n_layers = int(head_meta.get("n_layers") or 0)
        ell = int(head_meta.get("ell") or 0)
        if not n_layers:                       # older head.pt: read the layer count off the backbone config later; assume 3x
            head_lp, dense_lp = 1.0, 3.0
        else:
            head_lp = ell + 3 * (n_layers - ell)
            dense_lp = 3 * n_layers + (n_layers if bb.grad_ckpt or bb.dense_max_tokens > 8192 else 0)
        steps = max(1, round(head_tokens * head_lp / (dense_tok_per_step * dense_lp)))
        detail = (f"same training FLOPs: head {head_tokens:,} tok x {head_lp:g} layer-passes, "
                  f"dense {dense_tok_per_step:,.0f} tok/step x {dense_lp:g} -> {steps:,} steps")
    else:
        steps = max(1, round(head_tokens / max(1.0, dense_tok_per_step)))
        detail = (f"same training tokens: head {head_steps:,} x {head_tok_per_step:,} = {head_tokens:,} tok; "
                  f"dense {rows_per_step} rows x {mean_tok:,.0f} mean tok = {dense_tok_per_step:,.0f} tok/step -> {steps:,} steps")
    return steps, {"budget": mode, "head_steps": head_steps, "head_tokens": head_tokens, "dense_tok_per_step": round(dense_tok_per_step),
                   "mean_row_tokens": round(mean_tok), "detail": detail}


def train(P: S4Paths, bb_name: str, seed: int, hp: Optional[Dict] = None, device: str = "auto", steps: Optional[int] = None,
          limit_rows: Optional[int] = None, repo_override: Optional[str] = None, allow_unsealed: bool = False,
          rows_per_step: int = 8, val_max_rows: int = 200, dtype: torch.dtype = torch.bfloat16, variant: str = "") -> Dict:
    hp = dict(DENSE_HP, **(hp or {}))
    bb = backbone(bb_name)
    if not P.sealed() and not allow_unsealed:
        raise SystemExit(f"{P.data.root}/SEALED missing (G0); pass --allow-unsealed for a smoke run only")
    device = device_auto(device)
    run_dir = P.run_dir(bb_name, seed, variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / BB.DENSE_FILE
    if out_path.exists() and BB.load_ckpt(out_path).get("meta", {}).get("complete"):
        print(f"[dense] {out_path} already complete -> skip (delete it to retrain)")
        return {"skipped": True, "path": str(out_path)}
    head_path = run_dir / BB.HEAD_FILE
    set_seed(seed)
    presence = [PrepSet(P, s, u) for s, u in PRESENCE_SETS if (P.prep / s / u / "ids.npy").exists()]
    native = [PrepSet(P, s, u) for s, u in NATIVE_SETS if (P.prep / s / u / "ids.npy").exists()]
    val = PrepSet(P, *VAL_SETS[0]) if (P.prep / VAL_SETS[0][0] / VAL_SETS[0][1] / "ids.npy").exists() else None
    if not presence:
        raise SystemExit("no presence sets prepared (python s4ctl.py prep)")
    budget_info: Dict = {"budget": "steps(--steps)"}
    if steps is None:
        if not head_path.exists():
            raise SystemExit("--steps required when head.pt of the same backbone/seed does not exist yet (the budget is derived from it)")
        hm = BB.load_ckpt(head_path)["meta"]
        steps, budget_info = budget_steps(hm, hp, bb, presence, native, seed, limit_rows, rows_per_step)
        print(f"[dense] steps = {steps:,} ({budget_info['budget']} budget vs {head_path.name}: {budget_info['detail']})", flush=True)
    model, tok = BB.load_backbone(repo_override or bb, device=device, dtype=dtype)
    pad_id = BB.pad_id_of(model, tok)
    d = BB.hidden_size(model)
    groups = BB.prepare_dense_trainable(model, bb)
    if bb.grad_ckpt or bb.dense_max_tokens > 8192:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
        except Exception as e:                                        # tiny test models: fine without it
            print(f"[dense] gradient checkpointing unavailable: {e}")
    set_seed(seed)
    head = Head(d).to(device)
    n_train = BB.count_params(groups["lr_deep"]) + BB.count_params(groups["lr_lora"]) + BB.count_params(head.parameters())
    sets = presence + native
    opt = make_optimizer(groups, list(head.parameters()), hp)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, hp["warmup_frac"], hp["schedule"]))
    resume_path = run_dir / ("dense" + BB.RESUME_SUFFIX)
    # a resume checkpoint is ~12 bytes/param (fp32 weights + Adam): 3.6 GB for the 0.6B head, ~29 GB for the 4B ->
    # scale the interval with the trainable size (0.6B head: 30 min, 4B head: ~4 h) to keep the shared FS sane
    ck = Checkpointer(hp.get("ckpt_every_sec", 1800) * max(1.0, n_train / 3e8))
    step, epoch, start_bi = 0, 0, 0
    if resume_path.exists():
        st = BB.load_resume(resume_path, model, bb, None, head, opt, sched, dense=True)
        step, epoch, start_bi = int(st["step"]), int(st["epoch"]), int(st["batch_idx"]) + 1
        if st.get("total_steps") != steps:
            raise SystemExit(f"{resume_path}: total_steps {st.get('total_steps')} != {steps} -> delete it to restart")
        print(f"[dense] resumed from {resume_path.name}: step {step:,}/{steps:,} (epoch {epoch}, batch {start_bi:,}, saved {st.get('saved_at')})", flush=True)
    log = JsonlLog(run_dir / "dense.log.jsonl")
    meta = run_meta(ROOT, {"backbone": bb_name, "seed": seed, "mode": bb.dense_mode, "max_tokens": bb.dense_max_tokens,
                           "model_sha": snapshot_sha(repo_override or bb.repo), "total_steps": steps, "rows_per_step": rows_per_step,
                           "budget": budget_info,
                           "trainable_params": n_train, "prep": {f"{p.split}/{p.subset}": p.info for p in sets}, "complete": False})
    print(f"[dense] {bb_name} s{seed} mode={bb.dense_mode} max_tokens={bb.dense_max_tokens} trainable={n_train:,} steps={steps} "
          f"rows/step={rows_per_step} device={device}", flush=True)
    model.train()
    t0 = time.time()
    t_log, step_log = time.time(), step
    run_loss, run_n, run_correct = 0.0, 0, 0
    last_val: Dict = {}

    def save_resume(epoch_: int, bi_: int) -> None:
        BB.save_resume(resume_path, model, bb, None, head, opt, sched,
                       {"step": step, "epoch": epoch_, "batch_idx": bi_, "total_steps": steps, "seconds": round(time.time() - t0, 1)}, dense=True)
        ck.mark()
        print(f"   [ckpt] step {step:,} -> {resume_path.name}", flush=True)

    while step < steps:
        rows = SM.epoch_rows(presence, native, seed, epoch, hp["native_ratio"], bb.dense_max_tokens, limit_rows)
        for bi, r0 in enumerate(range(0, len(rows), rows_per_step)):
            if bi < start_bi:
                continue
            batch = sorted(rows[r0:r0 + rows_per_step], key=lambda u: -u.n_tok)     # similar lengths together: less padding
            opt.zero_grad(set_to_none=True)
            for chunk in SM.token_budget_batches(batch, bb.dense_micro_tokens):
                ids, valid, y = SM.collate_rows(chunk, sets, pad_id, bb.dense_max_tokens)
                ids, valid, y = ids.to(device), valid.to(device), y.to(device)
                logits = BB.dense_logits(model, head, ids, valid)
                loss = F.binary_cross_entropy_with_logits(logits, y) * (len(chunk) / len(batch))
                loss.backward()
                run_loss += float(loss.detach()) * len(batch)
                run_n += len(chunk)
                run_correct += int(((logits > 0).float() == y).sum())
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], hp["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            if step % hp["log_every"] == 0 or step == steps:
                sps = (time.time() - t_log) / max(1, step - step_log)
                t_log, step_log = time.time(), step
                rec = {"step": step, "epoch": epoch, "loss": run_loss / max(1, run_n), "acc": run_correct / max(1, run_n),
                       "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1), "gpu_gb": gpu_mem_gb(),
                       "sec_per_step": round(sps, 3), "eta_h": round((steps - step) * sps / 3600, 2), "total": steps}
                log.write(**rec)
                print(f"   step {step}/{steps} loss={rec['loss']:.4f} acc={rec['acc']:.3f} lr={rec['lr']:.2e} {rec['sec']}s "
                      f"({sps:.2f} s/step, eta {rec['eta_h']} h, gpu {rec['gpu_gb']} GB)", flush=True)
                run_loss, run_n, run_correct = 0.0, 0, 0
            if val is not None and hp["eval_every"] and (step % hp["eval_every"] == 0 or step == steps):
                last_val = eval_val(model, head, val, pad_id, device, bb.dense_max_tokens, bb.dense_micro_tokens, val_max_rows)
                log.write(step=step, val=last_val)
                print(f"   val@{step}: {last_val}", flush=True)
            if step >= steps:
                break
            if ck.due():
                save_resume(epoch, bi)
                if ck.stop:
                    print(f"[dense] signal {ck.signal_name}: checkpoint saved at step {step:,}; exiting rc=3 -- resubmit to resume", flush=True)
                    log.close()
                    sys.exit(3)
        start_bi = 0
        epoch += 1
    if val is not None and not last_val:
        last_val = eval_val(model, head, val, pad_id, device, bb.dense_max_tokens, bb.dense_micro_tokens, val_max_rows)
    meta.update(steps=step, epochs=epoch, seconds=round(time.time() - t0, 1), val=last_val, complete=True, gpu_gb=gpu_mem_gb())
    model.eval()
    BB.save_dense_ckpt(out_path, model, head, bb, hp, meta)
    BB.write_hashes(run_dir)
    if resume_path.exists():
        resume_path.unlink()
    log.write(step=step, done=True)
    log.close()
    print(f"[dense] saved {out_path} ({step} steps, {meta['seconds']}s) val={last_val}", flush=True)
    return {"path": str(out_path), "steps": step, "val": last_val}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S4 dense-FT baseline")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--rows-per-step", type=int, default=8)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--allow-unsealed", action="store_true")
    ap.add_argument("--budget", default=None, choices=[None, "tokens", "flops", "steps"], help="default DENSE_HP.budget (tokens)")
    ap.add_argument("--variant", default="", help="run-dir suffix (see head_train --variant)")
    a = ap.parse_args(argv)
    P = S4Paths.load().ensure().for_backbone(a.backbone)      # prep arrays of the backbone's tokenizer family (T5: llama3)
    train(P, a.backbone, a.seed, {"budget": a.budget} if a.budget else None, a.device, a.steps, a.limit_rows, a.repo, a.allow_unsealed,
          a.rows_per_step, variant=a.variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
