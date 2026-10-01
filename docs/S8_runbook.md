# S8–S9 실행 순서 — E6 ablation 13축, Table 4, Fig 1 (+ tail-pad, hard-negative head)

전제: LG 0.6B s0(`runs/v2/qwen3guard_0.6b_s0/{head.pt,router.pt,dense_ft.pt}`)와 그 S5/S6 e2 rows가 있음. 모든 명령은 `/home/212aig08/lg_s1`에서. S8 코드는 S7 결과값에 의존하지 않음(9/27): 학습·평가 코드 위에 얹은 것이라 S7 잡이 도는 동안 제출해도 됨. 단 **평가 잡은 48 GB 노드가 필요 없고(0.6B, 24 GB)**, head 재학습은 3–4 h·router 재학습 1 h·평가 5–8 h/arm.

| 계획서 | 명령 | 자원 | 산출물 |
| --- | --- | --- | --- |
| 등록된 arm 보기 | `python s8ctl.py arms` / `--axis E6-4` | — | 35 arm: 축·종류(infer/router/head)·평가 spec |
| E6-6 hard negative 채굴 | `python s8ctl.py mine --sbatch` | 24 GB, ~40 min | `runs/v2/qwen3guard_0.6b_s0/hardneg/{train.main,train.main16k}.npz` + `meta.json` (`run`에 명령의 hardneg arm이 자동으로 먼저 냄) |
| 추론만 (E6-1 k=64·all, E6-2 stride, tail-pad) | `python s8ctl.py run --arms k64,kall,stride256,tp32,tp64 --sbatch` | 24 GB, 5–8 h/arm | `eval/v2/e2/lg_k64__qwen3guard_0.6b_s0.rows.jsonl`, `sliding__…`, `lg_k32_s256__…`, `lg_k32_tp32__…` |
| router 재학습 (E6-3 ℓ, E6-4 pooling, E6-5 목적×용량) | `python s8ctl.py run --axis E6-3 --sbatch` (E6-4, E6-5도 같은 꼴) | 24 GB, router 1 h + 평가 | `runs/v2/qwen3guard_0.6b_<arm>_s0/{head.pt→base 심볼릭 링크, router.pt}` + rows |
| head 재학습 (E6-2 b, E6-6 init·hardneg, E6-7·8·9 레시피) | `python s8ctl.py run --arms hardneg,hardneg_ft --sbatch` / `--axis E6-8` | 24 GB(b256는 48 GB), head 3–4 h(hardneg_ft는 1 epoch ≈ 1 h) → router 1 h → 평가 | `runs/v2/qwen3guard_0.6b_<arm>_s0/{head.pt, router.pt}` + rows |
| 전부 | `python s8ctl.py run --arms all --sbatch` | 32 arm ≈ 25 학습 잡 + 32 평가 잡 | (큐 상황 보며 축 단위로 내는 걸 권함) |
| Table 4 · Fig 10 · Fig 1 | `python s8ctl.py tables [--sbatch]` | CPU | `reports/s8/table4_ablation.{md,json}`, `reports/s8/fig10_k_sweep.json`, `reports/s7/fig1_pareto.{md,json}` |
| 상태 | `python s8ctl.py status` | — | arm별 head/router/rows/평가 진행(`STOPPED` = 진행 중 로그인데 잡 없음, `FAILED before the first rows` = 시작 로그만 있고 잡 없음 → `.err`), squeue의 `s8_` 잡 |
| 몇 arm만 보기 | `python s8ctl.py table --arms ep1,ep2` / `--axis E6-10` / `--no-base` | — | 마지막 `tables`의 JSON에서 base·s1·s2 + 고른 arm(그 arm의 `+rg3`/`+k8` 행 포함)만 출력 |

## 1. 순서대로 (권장)

```bash
cd /home/212aig08/lg_s1
python s8ctl.py mine --sbatch                                        # 1) hard negatives (40 min)
python s8ctl.py run --arms k64,kall,stride256,tp32,tp64 --sbatch      # 2) 추론 arm 5개 (학습 없음)
python s8ctl.py run --arms hardneg,hardneg_ft --sbatch                # 3) G2 (a)의 핵심 후보: 채굴이 안 끝났으면 mine 잡에 afterok
python s8ctl.py run --axis E6-4 --sbatch                              # 4) router 축들 (E6-3, E6-5 같은 꼴; 각 1 h + 평가)
python s8ctl.py run --axis E6-2 --sbatch                              # 5) head 축들 (E6-6 init_dense, E6-7, E6-8, E6-9)
python s8ctl.py status
python s8ctl.py tables --sbatch                                       # 끝난 arm만으로도 표는 나옴 ("no rows" = 아직)
```

`run`은 arm마다 (mine →) head → router → 평가를 afterok 체인으로 묶고, 이미 있는 head.pt/router.pt는 학습기가 건너뛰며(`meta.complete`), rows는 id로 재개됨. head 잡은 SIGTERM에 체크포인트 + rc=3로 같은 id에 requeue, 평가 잡은 `lgs1.interrupt`로 requeue.

## 2. 무엇이 어떻게 계산되나

* **평가 부분집합**(`lgs8.config`): test/main의 depth **{0, .2, .5, .8, 1}** × 200쌍 × 8 L = 16,000행 + test benign-long 4,000행 + calib benign-long 4,000행(임계값) — `lgs5.run --exp e2 --L all --depths 0.0,0.2,0.5,0.8,1.0 --max-pairs 200 --no-calib-main`. 기준선(s0 k32)은 전량 rows에서 **같은 행만 골라** 비교(`subset_row`). test 격자는 0.0, 0.1, …, 1.0이라 계획서의 {0, .25, .5, .75, 1}은 0.25/0.75가 비어 실제로는 3 위치(9,600행)였음(9/27 발견) → 격자값 5개로 바꿈. 9/27 낮에 끝난 추론 arm(k64·stride256·tp32·tp64)은 같은 파일에 0.2/0.8 행만 id로 이어 붙음(`run --arms … --sbatch` 다시). B1-long(64k/128k dense 부분집합)·B2 sweep의 `--depths 0,0.25,0.5,0.75,1`도 같은 이유로 3 위치(0, .5, 1)인데, 이미 Table 2/3에 들어간 행이라 그대로 두고 3 위치로 보고(`s6ctl.SUB_LONG` 주석). 이제 `lgs5.run`/`lgs3.longbench`는 어떤 depth도 행에 없으면 `[depths] WARNING`을 찍음.
* **arm 종류**
  * infer: 기준 체크포인트 + 추론 옵션. spec 문법 `lg:<run>[:kK][:sS][:tpN]` — `s256` = stride 2b(겹침 없음), `tpN` = 중립 접미사 N 토큰을 문서 끝에 붙인 뒤 창 분할(`lgs5.runners.TAIL_PAD_TEXT`; harmful·twin·benign-long 모두에 붙어 임계값도 같은 조건). method 이름에 옵션이 붙어(`lg_k32_s256`, `lg_k32_tp32`) 기본 rows와 섞이지 않고, k′<k·rg3 유도도 접미사를 보존.
  * router: 기준 head 고정(arm 디렉터리에 `head.pt` 심볼릭 링크) + router 재학습. `lgs4.router_train` 옵션 `--pool max|mean|top2|top4|lse`, `--router-ells 6|12|16|6,12`(트렁크 상태를 읽는 깊이; 여러 개면 concat, `Router(in_dim=2d)`), `--arch linear|mlp`, `--objective distill_listwise|distill|window_bce|token_bce`. 체크포인트가 pool/depth/arch를 들고 있어 `load_lg`가 그대로 복원 → 평가·비용 코드는 손대지 않음(`LandmarkGuard._trunk`가 필요한 깊이만 한 번에 계산).
  * head: `lgs4.head_train` 옵션 `--b 64|256`, `--hard-neg DIR --hard-per-row 4`, `--init-from head.pt|dense_ft.pt`, `--max-L`, `--needle-frac`, `--domains`, `--no-twin` → 새 head + 그 head로 router 재학습(계획서 §3).
* **hard negatives**(`lgs4.hardneg`): presence row마다 기준 run의 router가 non-needle 창을 순위 매기고 상위 8개를 기준 head에 넣어 (창, head_prob, router_score)를 저장. 학습 시 head_prob 높은 순으로 행당 4개가 무작위 음성을 대신함(`Unit.kind == "hard"`). 근거: 128k 하락 = 라우터가 고른 benign 상위 창을 head가 높게 봐서 생기는 임계값 팽창(S6 9/23; 8B는 손실 0–2 pt, 0.6B는 10–13 pt). `hardneg`는 Gen 초기값에서 3 epoch 재학습, `hardneg_ft`는 s0 head를 초기값으로 1 epoch만.
* **Table 4**(`lgs8.table4`): arm과 기준을 같은 행에서 `lgs5.tables.table2`로 계산 — R@28k·R@128k(matched 5 %, 위치 평균), pos-min@128k, infl(1k 임계값의 128k benign-long FPR), hit@k, ε̂, ms@128k, Δ(기준 대비). k=4/8/16과 rg3는 기준 rows에서 유도. Fig 10 = k별 recall(+ S7 ksweep의 지연). 평가가 덜 끝난 arm은 `partial n/N`으로 표시하고 **기준도 그 n행에서 다시 재서** Δ를 냄(진행 중에 `tables`를 돌려도 비교가 어긋나지 않음).
* **Fig 1**(`lgs8.fig1`): Table 2(정확도) × Table 5(비용)의 이름 대응(`match_cost`: LG는 c16+h8 측정값, CD `chunk_or_W_o`↔`cdWoO`, 32k-native 가드의 ≤28k는 yarn 지연) → x = p50 ms@28k, y = R@28k, 크기 = benign-long FPR@128k, inset 128k, $/1M docs = ms/doc × GPU $/h(가정치, 기본 1.2 $/h). `--png`는 matplotlib이 있을 때.

## 3. 어떤 결과를 기대하나 (읽는 법)

* G2 (a) 후보 두 개가 이 단계의 핵심: `tp32/tp64`가 depth 1.0 결손을 지우면 pos-min@128k가 0.44 → 0.75쯤(0.6B), `hardneg*`가 팽창을 줄이면 R@128k·infl이 함께 좋아져야 함. 둘을 합친 arm이 `hardneg_tp32`(hardneg head + tail-pad 추론; `run --arms hardneg,hardneg_tp32`로 함께 내면 hardneg 체인 뒤에 붙고, 따로 내면 hardneg의 router.pt가 있어야 함).
* E6-1: k64·kall이 k32보다 낫지 않으면(S6에서 hit@32 0.95, top2_mean 붕괴) "k는 원인이 아님"의 근거. E6-2 stride256은 겹침 없음의 비용(ms)과 경계 손실. E6-3/4/5는 hit@k 열을 보는 축. E6-7/8/9는 R@128k의 외삽 근거.
* **결과(9/28 19:22, 0.6B s0, 5 depth 전량; 기준 R@1k 0.890 / R@28k 0.833 / R@128k 0.734 / pos-min@128k 0.435 / infl 0.164 / hit@32 0.954)** — b64·b256·init_dense·len4k·len8k·data50은 아직.
  * **G2 (a)는 0.6B 전 arm FAIL**(필요 pos-min ≥ R@1k − 0.05 ≈ 0.86). 최고는 tail-pad: tp64 R@128k 0.809 / pos-min 0.795, tp64+rg3 0.812 / 0.780 / infl 0.124(R@1k 0.922), tp32+k8 0.822 / 0.795(hit 0.90). tail-pad는 R@1k도 0.89 → 0.91로 올려 기준선이 같이 올라감. 남은 격차는 위치(pos-min ≈ pos-mean)가 아니라 128k 전체 하락(ε̂ ≈ 0.10) = 임계값 팽창 + 라우팅 미스 4 %. → 주 구성 후보 **k32 + tp64 + rg3**; PASS 가능성은 8B + tp64(전량 격자 e2 제출 9/28 12:51: 0.6B 1025415, 8B 1025416).
  * **E6-1 k**: recall@128k는 k에 단조 감소(k4 0.785 → k8 0.779 → k16 0.762 → k32 0.734 → k64 0.726 → all 0.689), hit@k는 단조 증가(0.85 → 1.0), pos-min은 0.66 → 0.24. 창이 많을수록 max 집계 팽창(infl 0.116 → 0.252). k=32 유지 + rg3가 팽창을 0.164 → 0.120으로.
  * **E6-2**: stride256(겹침 없음) R@28k −8.4 pt, R@128k −5.5 pt(시간 −28 %) → 겹침 필수.
  * **E6-3 깊이**: ell6 hit 0.903(−5 pt), ell12/ell16/ell6_12는 hit 0.928/0.864/0.931이지만 R@128k가 셋 다 정확히 0.693·pos-min 0.260 — **임계값이 head 천장에 붙음**(infl 0.21 → matched 5 % τ@128k ≈ 1 → recall = head가 포화하는 needle의 비율, 라우터가 달라도 동률). Table 4에 `tau@128k` 열 추가. ℓ=9(기본)가 최선이고 더 깊은 상태는 벤ign 창의 점수만 올림.
  * **E6-4 pooling**: mean 붕괴(hit 0.698, R@128k 0.574), top2 0.716, lse 0.706, top4 0.745(≈ max 0.734). max/top4가 맞음.
  * **E6-5 목적·용량**: hit@32 — mlp_window 0.978 > mlp_distill 0.956 ≈ 기본(linear distill+listwise) 0.954 > obj_window 0.944 > mlp 0.931 ≈ mlp_token 0.932 > obj_distill 0.928(listwise 항 −2.6 pt) > obj_token 0.844. R@128k는 전부 기본 ±1.6 pt(mlp_distill 0.750 최고). 라우터 용량·목적은 병목이 아님.
  * **E6-6**: hardneg(재학습) R@28k −4 pt·팽창 그대로, hardneg_ft R@128k +3.4 pt이지만 infl 0.164 → 0.314. hard negative는 팽창을 못 고침.
  * **E6-8 데이터**: data10 R@128k 0.755(+0.02)·infl 0.346, data25 R@1k 0.832(−6 pt) — 단조가 아니라 **head 학습의 seed 분산이 큼**을 시사 → Table 4에 s1·s2(같은 레시피, 다른 seed) 행을 넣어 d의 잡음 바닥을 같이 보임.
  * **E6-9**: dom1(배경 1 도메인) R@128k 0.586(−15 pt)·infl 0.41 → 배경 다양성이 핵심. notwin(twin 제거) R@128k 0.890(+16 pt)·R@1k 0.947은 **twin FPR을 대가로 산 것**(임계값은 benign-long으로만 맞추므로 표의 recall만 보면 착시) → Table 4에 `twinFPR@28k` 열 추가; 논문에는 twin FPR과 함께 보고.
  * 부수: router arm의 R@1k는 전부 0.890(1k는 8창뿐이라 k=32가 전부 덮음 → 라우팅 무관) — 라우터 축의 효과는 28k·128k 열에서만 읽음.
* **9/29 12:07 표(seed 행·tau·twinFPR 포함)로 추가된 것**
  * **잡음 바닥**(같은 레시피 s1·s2): R@128k −0.4/−2.2 pt, pos-min +4.5/−2.0 pt, infl 0.230/0.188(기준 0.164) → R@128k ±2 pt·pos-min ±5 pt·infl ±0.07은 seed 잡음. 그 안에 드는 것: hardneg_ft +3.4, mlp_distill +1.6, pool_top4 +1.1, data10 +2.1. 밖: len4k +6.8, data50 +7.3, notwin +15.6, b64 −8.2, dom1 −14.8, pool_mean −16, stride256 −5.5, ell12/16 −4.1.
  * **tau@128k**: 기준 0.9998, ell12/16/6_12 0.9999(천장 → 동률), rg3 0.977, k8 0.980, k4 0.0036(top-4 benign 창은 head 확률이 거의 0이라 임계값이 바닥) — matched 임계값이 k에 따라 0.004 → 0.98 → 0.999 → 0.9998로 올라가는 것이 팽창 그 자체.
  * **twinFPR@28k**(기본 판정): 기준 0.158, notwin **0.695**(recall +16 pt의 대가), dom1 0.482, data10 0.268, hardneg_ft 0.109(hard negative는 twin은 줄이고 benign-long 팽창은 키움), b256 0.140.
  * **b64** R@1k −6.8·R@128k −8.2(창이 needle보다 짧아짐), **b256** R@1k +2.6·R@28k +4.9(128k는 다음 표) — 큰 창이 유리, head 비용 2×.
  * **init_dense** −2.2(dense-FT 초기화 무익). **len4k +6.8·data50 +7.3**: 둘 다 기준보다 optimizer step이 적음 → 기준 head가 128k에 대해 과학습일 가능성(val ≤ 8k라 못 봄) → `ep1`/`ep2` arm(E6-10, 전체 레시피 1·2 epoch) 추가해 확인.
  * **tp64의 비용**(S7, RTX 6000 Ada, 9/29): `lg_k32_tp64+c16+h8` 0.6B 1019 ms@128k(tp 없이 1099; 잡음 안), peak +8.3 % PASS; 8B 13.2 s, +1.5 % PASS → tail-pad는 공짜. G2 (b)는 여전히 구조적 FAIL(0.30).
  * Table 2 all-L·g2_check가 S8 arm 파일 33개를 같이 읽어 오염됐던 것(9/29 13:05 표) → `lgs5.tables.load_eval`이 depth 격자를 다 덮지 않는 e2 파일(부분집합, 가장 많은 파일의 75 % 미만)을 Table 4 몫으로 빼고 stderr에 알림. 전량 tp64 파일은 남음.
  * **ep1/ep2**(9/30 11:33 표, E6-10): ep1 R@128k 0.750(+1.6, seed 잡음 안)·pos-min 0.500·infl 0.230, ep2 0.728(−0.6) → "기준 head가 과학습" 가설 기각. len4k·data50의 +7 pt는 step 수가 아니라 레시피·부분집합 분산 → 주 구성 후보는 그대로 k32+tp64(+rg3).
  * **9/30 12:51 g2_check(tp64 8B 전량 포함) — G2 (a) 첫 PASS**: `lg_k32_tp64|qwen3guard_8b_s0` R@1k 0.933 → pos-min@128k 0.890(필요 0.883, +0.7 pt), `_rg3` 0.935/0.895(필요 0.885, +1.0 pt), k16_tp64·k8_tp64도 0.885(+0.4 pt) PASS; ε̂ 0.034, hit@32 0.961. tp64 없이는 0.695. 여유는 depth당 200쌍의 표준오차(≈2 pt) 안이라 논문에는 이진 PASS보다 "R@1k 대비 −4.3 pt(CI)"로 쓰고, 8B는 s0뿐이라 seed 확장은 S10′ 이후 과제. 0.6B tp64는 0.795(필요 0.858) FAIL, 4B·LG3-8B tp64는 미평가 → S9′ 2번(`lg:qwen3guard_4b_s0:tp64`, `0.6b_s1/s2:tp64` 전량 e2) 제출. 주 구성 = **k32 + tp64 + rg3**(8B 기준), Fig 1에 k8_tp64 8B 점 추가.
  * **b256_tp64**(10/1 11:49 표): 0.6B 최고 arm — R@1k 0.933·R@28k 0.885·R@128k 0.832(+9.8 pt)·pos-min 0.815(+38 pt), `+rg3` 0.836/0.825/infl 0.112 → b128 tp64+rg3(0.812/0.780)보다 +2.4/+4.5 pt. G2 (a) 0.6B는 여전히 FAIL(0.825 vs 필요 0.878)이지만 격차 9 → 5 pt. `b256+k8`의 tau 0.065는 k4처럼 임계값 바닥(무시). 주 구성은 b128+k32+tp64+rg3 유지(8B PASS·35 arm·비용·4B/LG3-8B/seed가 전부 b128 기준); b256은 창 ablation + 0.6B 개선 옵션으로 보고, b256 비용(Table 5)이 거의 공짜면 재고.
  * **10/1 12:07 g2_check — 0.6B seed·b256 비용**: tp64 전량 0.6B s0/s1/s2 pos-min@128k 0.795/0.775/0.740(R@1k 0.908/0.890/0.920; rg3 0.780/0.755/0.775) → 0.6B는 seed와 무관하게 FAIL(평균 0.77, 격차 ≈13 pt)로 보고. b256 비용(RTX 6000 Ada, `lg_k32+c16+h8|qwen3guard_0.6b_b256_s0`): 1295 ms@128k(b128 1099, **+18 %**; tp64 1305 vs 1019, +28 %), peak 1467 MB·growth 0.218 → G2 (c) FAIL → b256은 공짜가 아님. **주 구성 b128+k32+tp64+rg3 확정**, b256은 Table 4 창 ablation(+2–4 pt on 0.6B, +18 % 지연, +12 % peak)으로만. 4B tp64 행(0.903 → 0.790)은 평가 중인 파일의 ~10 % 행이라 예비값(저녁 재계산); g2_check에 `n@128k`·`rows` 열 추가해 부분 파일이 보이게.
  * 전량 격자(Table 2, 11 depth) `lg_k32_tp64` 0.6B: R@1k 0.908 → R@128k 0.816, pos-min 0.795, FPR@τ1k bl 0.176, ε̂ 0.092; `_rg3` 0.917/0.814/0.780/0.124/0.104 → G2 (a) FAIL 확정(필요 0.858/0.867). 8B tp64는 9/30.
* **9/28–29 코드**: Table 4에 `R@1k`·`G2(a)`·`tau@128k`·`twinFPR@28k` 열, seed 행(s1·s2), arm마다 `<arm>+rg3`/`+k8` 유도 행(그 arm의 rows에서), ms 열은 평가 노드 시간이라 참고용(비용은 Table 5). S7 비용 spec에 `:sS`·`:tpN`(`lg:<run>:k32:tp64:c16:h8`; method `lg_k32_tp64+c16+h8`, g2_cost·Fig 1이 옵션 붙은 행을 같은 옵션의 비용 행에 대응, rg3는 원본 rows의 비용) → 주 구성 후보의 Table 5 행. arm 35개(E6-10 ep1·ep2 추가). 유도 행(k′·rg3)은 run당 **하나의 native pass**(행이 가장 많은 K, 동률이면 작은 K)에서만 만듦 — k64 파일이 생기자 lg_k8·lg_k32_rg3가 두 번 유도돼 n=33,600으로 찍혔던 것 수정(`lgs5.tables.native_source_k`; S5/S6 표에도 적용).

## 3.5 9/27 저녁 메모 — cfg_hash 회귀와 부분집합 depth

* **LineageMismatch(cfg_hash)**: 9/27 14:46 zip의 S8 코드가 `LGConfig.to_dict()`에 `router_ells: None`, LGEval 해시에 `tail_pad: 0`을 항상 넣어 **모든 LG rows 파일의 cfg_hash가 바뀜** → LG3-8B e2가 requeue 뒤 재개를 거부(`existing=937865eb716f… current=61a08307b478…`). 고침: 두 키는 값이 있을 때만 해시에 들어가서 기본 구성은 9/27 이전과 같은 해시(`lgs5.runners.lg_cfg_dict`). 9/27 낮에 그 코드로 쓰인 파일(S8 추론 arm 4개)은 자기 해시(`lg_cfg_dict_20260927`)를 `write_rows`가 알아보고 그 해시로 이어 씀(`[s5] … rows carry the 2026-09-27 cfg_hash … -> resuming under it`) — 손으로 할 일 없음. 실패한 LG3-8B e2는 그냥 다시 제출.
* **depth**: 위 §2. `s8ctl status` 머리에 목표 행 수(16,000 test/main)와 depth를 찍음.
* **ECC 노드**(9/30): 평가 잡이 시작 직후 `CUDA error: uncorrectable ECC error`로 죽으면(`status`에 `FAILED before the first rows`) `.out` 머리의 `[common] host=nNN`을 보고 `python s1ctl.py env-set LG_EXCLUDE=nNN` 뒤 재제출(`--exclude`가 모든 sbatch에 붙음; 여러 노드는 쉼표). 풀려면 `env-set LG_EXCLUDE=`.

## 4. 테스트

`tests/test_s8.py`: 다중 깊이 트렁크(구간 forward = 단일 forward), `Router(in_dim)`·`RouterMLP`·`make_router`, spec 옵션·유도의 접미사 보존, hard-negative 채굴(needle 창 제외·head_prob 정렬·캐시)과 `row_units`/필터(`max_L`·`needle_frac` 쌍 유지·`twin`·`domains`), head 재학습 knob(`init_from` head/dense, `b`, hard_neg + b 불일치 거부), router 변형(mean·mlp·token_bce·depths 1,2 → `load_lg` 복원 → 평가 end-to-end, 다른 목적/pooling), tail-pad·stride 추론 행, Table 4(유도 행·arm 행·`no rows`), Fig 1(이름 대응·Pareto), `s8ctl` dry-run(mine → head → router → eval 체인, 옵션 문자열, requeue 시그널), cfg_hash 형식(9/27 이전 dict와 동일)·9/27 파일의 alias 재개·부분집합 depth 경고·Table 4 `partial`·G2(a) 열·조합 행, 유도 행의 단일 native pass 규칙. 전체 147 tests.
