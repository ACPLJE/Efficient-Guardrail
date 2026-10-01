# S4 실행 순서 — LG 학습(T1 0.6B×3 seed, T3 4B), dense-FT(T2), E0-7 FLOPs 대조

전제: `lg_data_v2/SEALED`가 있어야 합니다(G0, `python s3ctl.py gates --seal`). 봉인 전에는 어떤 학습 명령도 시작하지 않습니다(`--allow-unsealed`는 tiny smoke 전용). 모든 명령은 `/home/212aig08/lg_s1`에서.

| 계획서 | 명령 | 자원 | 산출물 · 통과 기준 |
| --- | --- | --- | --- |
| 데이터 준비 | `python s4ctl.py prep --sbatch` | CPU(8 GB 티어), ~10분 | `$LG_OUT/s4/prep/<split>/<subset>/{ids.npy,offsets.npy,meta.jsonl.gz}` — train/main·main16k·native, val/main·benign_long, calib/main(28k만)·benign_long(28k만); `prep.json`의 `span_check.pass=true`(D8 tok/qwen3 span과 일치) |
| T1 LG 0.6B ×3 seed | `python s4ctl.py t1 --sbatch` | seed당 head(24 GB, ~3 h) → router(~1 h) → valcheck(~30 min) 체인 3개, 동시 | `runs/v2/qwen3guard_0.6b_s{0,1,2}/{head.pt,router.pt,hashes.json,val_report.json}`; **val_report `hit8_at_28k ≥ 0.95`** |
| T2 dense-FT 0.6B ×3 seed | `python s4ctl.py t2 --sbatch` | seed당 24 GB, head 잡 뒤 자동 시작(같은 step 수), ~6–12 h | `runs/v2/qwen3guard_0.6b_s{0,1,2}/dense_ft.pt` |
| T3 LG 4B | `python s4ctl.py t3 --sbatch` | 48 GB 티어, head ~10 h → router ~2 h → valcheck | `runs/v2/qwen3guard_4b_s0/…` |
| E0-7 FLOPs | `python s4ctl.py flops --sbatch` | GPU 1, ~10분 | `reports/e07_flops.{json,md}` — L∈{4k,28k,128k} 분석식 vs FlopCounterMode ±10% PASS, deep 항이 L에 불변 |
| 전부 | `python s4ctl.py all --sbatch` | prep → (t1, t2, t3, e07) 의존성 체인 | `python s4ctl.py status` |

## 1. 순서대로

```bash
cd /home/212aig08/lg_s1
ls /home/212aig08/lg_data_v2/SEALED          # 없으면: python s3ctl.py gates --seal
python s4ctl.py all --sbatch                 # prep 1 + head/router/valcheck 3 seed + dense 3 + 4B 체인 + e07
python s4ctl.py status                       # run별 head/router/dense 상태, val_report, E0-7
```

개별 재제출(잡 하나가 죽었을 때 — 완료된 체크포인트는 건너뜁니다):

```bash
python s4ctl.py head   --backbone qwen3guard_0.6b --seed 1 --sbatch
python s4ctl.py router --backbone qwen3guard_0.6b --seed 1 --sbatch --dep <head jobid>
python s4ctl.py valcheck --backbone qwen3guard_0.6b --seed 1 --sbatch --dep <router jobid>
python s4ctl.py dense  --backbone qwen3guard_0.6b --seed 1 --sbatch            # head.pt가 있으면 step 수를 거기서 읽음
```

## 2. 무엇이 어떻게 학습되나 (Table A2)

* **head (window-FT)** — 배경 ≤16k의 presence 행 34,000개(train/main 32,000 + main16k 2,000) + native 행(presence 행당 0.5개 추출). 행마다 needle을 온전히 담는 윈도우(≤3, 라벨 = 행 라벨: harmful 1 / benign twin 0) + 인접 윈도우 2개(0) + 무작위 윈도우 6개(0); native 행은 모든 윈도우가 행 라벨. 윈도우 = 256 토큰, 위치 0부터(block-local), trunk `[0, ell)`은 동결(no-grad, 기본 모델과 동일), `[ell, N)` + 최종 norm은 fp32 master로 학습(0.6B·4B) 또는 LoRA r=16(8B·LG3-8B), probe = LayerNorm+Linear(마지막 유효 토큰). BCE(pos_weight = 음/양 비율, 상한 8), AdamW(deep 2e-5 / LoRA 2e-4 / probe 1e-3, wd 0.01), warmup 3% + cosine, 3 epoch, 배치 32 윈도우(micro 32 / 4B·8B는 8 + grad checkpointing), bf16 autocast, seed = probe 초기화 + 데이터 순서. `head.log.jsonl`에 50 step마다 loss/acc, 2,000 step마다 val/main 윈도우 AUPRC.
* **router (head-distillation)** — 반드시 그 seed의 `head.pt`(FT 체크포인트) 위에서. Stage A: head 학습 행의 seed 고정 부분표본(presence 12,000 + native 6,000행; 9/20, 전량은 3090에서 8–10 h)의 모든 윈도우를 학습된 head에 통과시켜 P_head(window)를 `router_targets/*.npy`에 캐시. Stage B: 동결 trunk 특징 → LayerNorm(d)→Linear(d,1) 토큰 점수 → 윈도우 max; loss = BCE(score, 1[P_head>0.5]) + 1.0·KL(softmax(logit P_head) ‖ softmax(scores)) (행 단위 listwise), Adam 1e-3, 2 epoch, 행 8개/step. epoch마다 val/main hit@8/16/32.
* **dense-FT** — 같은 백본을 같은 행(전체 문서, 라벨 = 행 라벨)으로 full-context FT(0.6B 전 층 ≤16k, 8B는 LoRA ≤8k), 마지막 토큰 probe, step당 행 8개(길이순으로 묶어 패딩 토큰 예산 24k로 micro-batch), 그 외 하이퍼파라미터 동일. **학습 예산은 같은 seed의 head.pt에서 유도**(`DENSE_HP.budget`, 9/20): 기본 `tokens` = head와 같은 학습 토큰 수(58,120 × 8,192 = 4.76억 → 행 8개 × 평균 2.8k 토큰 = 22.6k/step → ≈21k step); `flops` = 같은 학습 FLOPs(≈12k step); `steps` = 같은 step 수(계획서 원문, 58k step = 3090에서 240 h라 폐기). `python s4ctl.py dense … --budget flops`로 바꿀 수 있고 meta에 기록됩니다.
* **valcheck** — val/main(≤8k) hit@k, calib/main 28k(2,000행) hit@k, calib/benign_long 28k(500행)로 임계값을 잡은 matched-FPR 5%/1% recall(엄격 `score > tau`, 달성 FPR 함께). `pass = hit@8@28k ≥ 0.95`.
* **E0-7** — 분석식 `W·2b·ell·c + k·2b·(N−ell)·c`, `c = 2·P_layer + 4·2b·d_q`(윈도우 안 attention 포함) vs `FlopCounterMode`(eager attention으로 계측: fused SDPA 커널은 일부 백엔드에서 카운터가 못 봄). 가중치와 무관하므로 기본 백본으로 잽니다.

## 3. 무엇을 확인하나

* `python s4ctl.py status` — run별 `head: N steps auprc=…`(진행 중이면 `running <step>/<total> eta <h>`), `router: … hit@8=…`, `dense: N steps`, `val: hit@8@28k=… PASS/FAIL recall@5pct=…`.
* head 로그 `$LG_OUT/logs/s4_head_qwen3guard_0.6b_s0_<jobid>.out`: 첫 줄에 `trainable=…`, `epoch0 units=… steps/epoch=… total=…`; loss가 0.69에서 내려가고 `val@…: auprc_needle_vs_twin`이 0.9 이상으로 가면 정상.
* router 로그: `[router] targets train/main: N windows, mean P=…`(head가 flag하는 윈도우 비율, 0.05–0.2 사이가 보통) → `val hit@k after epoch 1: {'hit@8': …}`.
* T1 판정: 세 seed 모두 `val_report.json`의 `pass: true`. 실패하면 `calib_hit`(router)와 `matched.recall@5pct`(head)를 보고 계획서 §3.3 대응(router면 k=64·listwise 가중, head면 b=256) — 이건 S5 G1 뒤에 결정.
* E0-7: `reports/e07_flops.md` 표에서 세 L 모두 PASS, `deep measured TF`가 L과 무관하게 같음.

## 3.5 잡이 죽거나 느릴 때 (9/19)

* GPU 종류부터: 로그 3번째 줄 `GPU 0: NVIDIA …`. **TITAN RTX / RTX 2080 Ti(Turing)면 bf16이 느려 10배 걸립니다** — 새 코드는 이 파티션을 자동으로 빼므로 그 잡은 `scancel` 후 다시 제출. 로그의 `(N s/step, eta H h)`가 예상: 0.6B head ~0.3 s/step·5–6 h(Ampere/Ada), 0.6 s/step이면 12 h.
* head/dense는 30분마다(4B는 ~4시간마다)·제한 25분 전 신호 때 `runs/v2/<run>/head.resume.pt`를 씁니다. 제한에 걸리면 **스스로 같은 잡 번호로 requeue**되어 `[head] resumed from head.resume.pt: step …`로 이어지고 router/valcheck 의존성도 그대로 유지됩니다(로그는 이어 붙음). 노드 장애 등으로 그냥 죽었을 때만 같은 명령을 다시 내면 됩니다(`python s4ctl.py t1 --seeds <s> --sbatch`, `t2 --seeds <s> --sbatch`). 완료된 단계(head.pt·router.pt·dense_ft.pt)는 다시 내도 건너뜁니다.
* 한 seed만 다시: `scancel <그 seed의 head/router/valcheck/dense jobid>` → `python s4ctl.py t1 --seeds 0 --sbatch` → `python s4ctl.py t2 --seeds 0 --sbatch`(head 잡을 squeue에서 찾아 의존으로 겁니다).
* head는 끝났는데 router/valcheck가 죽었으면(`DependencyNeverSatisfied`): `scancel <valcheck jobids>` → `python s4ctl.py t1 --seeds 0,1,2 --from router --sbatch`. router는 12k presence + 6k native 행의 부분표본으로 학습하며(9/20), targets는 2,000행 part 파일로 캐시되어 이어서 돕니다.

* 4B head가 OOM이면(`.err`에 `torch.OutOfMemoryError` … `opt.step`): 새 코드(fused AdamW)로 다시 내고, 그래도 나면 `python s1ctl.py env-set LG_BB_MODE_qwen3guard_4b=lora` 뒤 `python s4ctl.py t3 --sbatch`(LoRA r=16, 메모리 ~10 GB).

## 4. 자원·시간 감 (RTX 6000 Ada/A6000 기준 추정)

* prep ~35분(CPU; train/main 64k행 17분). head 0.6B: 58,120 step(620k 윈도우/epoch × 3) × ~0.3 s ≈ 5–6 h(24 GB, Ampere/Ada; TITAN RTX는 2.8 s/step이라 제외). router 0.6B: targets ~1 h + fit ~40 min. valcheck ~20 min. dense-FT 0.6B: 같은 step 수 × 행 8개 ≈ 6–12 h. 4B head(fp32 master 2.4B params + Adam ≈ 32 GB, grad ckpt): ~10 h(48 GB 티어).
* 체크포인트 크기: head.pt 0.6B ≈ 0.8 GB(deep 층 bf16), 4B ≈ 5 GB; router.pt 수십 KB; dense_ft.pt 0.6B ≈ 1.2 GB.

## 5. head v2 ablation — landmark probe (9/21)

S5 framing 진단: 마지막 토큰 probe는 needle 뒤에 16토큰 이상이 있어야 제 점수를 내고(문서 끝 depth 1.0 = pos-min 0.66), benign twin도 같은 위치 효과. 창 끝에 학습되는 landmark 벡터를 붙여 항상 같은 자리에서 읽는 head(`probe=landmark`)를 **별도 run**으로 학습:

```bash
python s4ctl.py t1 --seeds 0 --variant lm --probe landmark --sbatch      # runs/v2/qwen3guard_0.6b_lm_s0: head(24 h 제한, 자동 재개) → router → valcheck
python s4ctl.py status                                                    # qwen3guard_0.6b_lm_s0 행
python s5ctl.py e1 --models lg:qwen3guard_0.6b_lm_s0 --sbatch             # valcheck PASS 뒤
python s5ctl.py e2 --models lg:qwen3guard_0.6b_lm_s0 --sbatch
python s5ctl.py e27 --runs qwen3guard_0.6b_lm_s0 --sbatch
python s5ctl.py tables                                                    # lg_k32 | qwen3guard_0.6b_lm_s0 행, framing 진단의 lm 행
```

판단(calib/main에서, test 아님): v1 s0 대비 pos-min@28k(문서 끝) ≥ +10 pt, E2-7 framing 손실 절반 이하, R@28k 같거나 위 → v2를 주 설계로 승격(그러면 s1/s2·4B·8B 재학습 ~35 h). 아니면 v1 유지, v2는 ablation 행.

**결과(9/22)**: 승격 안 함 — R@28k 0.820(v1 0.850), pos-min 0.635(0.665), tail 0–15 결손 그대로(0.729). benign-side framing은 사라졌지만(0.055→0.002) harmful-side가 커짐(0.015→0.064): 문서 끝 결손은 probe 위치가 아니라 needle 앞 benign 문맥의 희석. 자세한 표는 `docs/S5_first_results_2026-09-20.md` §D.
