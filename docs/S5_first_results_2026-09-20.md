# S5 첫 결과 메모 (2026-09-20, ≤28k) — §A 20:00 기본 판정, §B 20:24 matched-FPR 갱신

## 나온 것

| | R@1k | R@28k | pos-min@28k | FPR@28k(twin+bl) | hit@k@28k | ms/row@28k |
| --- | --- | --- | --- | --- | --- | --- |
| LG 0.6B k=32 (3 seed) | 0.831/0.831/0.839 | 0.844/0.853/0.853 | 0.64/0.67/0.65 | 0.172/0.194/0.202 | 0.976/0.977/0.973 | 609–630 |
| LG k=16 (유도) | 같음 | 0.842–0.843 | | 0.17–0.19 | 0.957–0.965 | |
| LG k=8 (유도) | 같음 | 0.830–0.832 | | 0.17–0.19 | 0.938–0.948 | |
| oracle 윈도우 (matched 5%, twin) | 0.804–0.827 | 0.805–0.817 | 0.53–0.61 | 0.145–0.163 (τ=0.5) | 1.0 | 482–494 |
| needle 단독 (matched 5%) | 0.855/0.840/0.875 (길이 무관) | | | | | |
| PolyGuard 7B | 0.863 | 0.818 | 0.74 | 0.359 | | 9,231 |
| Qwen3Guard-8B native | 0.752 | 0.696 | 0.63 | 0.382 | | 10,546 |
| Qwen3Guard-8B CD 512/256 (110행) | 0.848 | 0.800 | 0.80 | 0.509 | | 26,533 |
| Qwen3Guard-4B | 0.683 | 0.568 | 0.48 | 0.238 | | 7,874 |
| Nemotron v3 head / tail (ctx 8k) | 0.717 | 0.075 / 0.122 | | 0.011 / 0.028 | | 2,400 |
| LlamaGuard3-8B | 0.309 | 0.085 | 0.005 | 0.004 | | 9,302 |
| WildGuard 7B | 0.814 | 0.015 (312행만 판정) | | 0.000 | | 11,267 |
| Granite 3.3-8B | 0.369 | 0.180 (409행만) | | 0.043 | | 12,203 |
| GuardReasoner 8B (부분) | 0.652 | 0.377 | | 0.112 | | 47,522 |
| Qwen3Guard-0.6B / Stream-8B / gpt-oss-20b | 0.65 / 0.72 / 0.61 | 16k·28k(gpt-oss는 4k) 이후 판정 없음 | | | | |

## 읽기

1. **길이 불변(주장 1)은 그대로 나옴.** LG는 1k→28k에서 R이 오히려 +1.5 pt(ε̂=0), 11개 깊이 최소도 0.64–0.67. dense 가드는 (a) recall 붕괴형(LG3-8B, WildGuard, Nemotron, Granite, 0.6B/Stream — 문맥·OOM 포함) 또는 (b) recall 유지 대신 장문 benign FPR 폭등형(PolyGuard 0.36, Qwen3Guard-8B 0.38, CD 0.51). LG는 28k에서 R 0.85로 (b)군보다 높고 FPR 0.19로 (b)군의 절반 — 같은 기본 판정 기준.
2. **하지만 FPR 0.17–0.20은 그 자체로 높다.** oracle(정답 윈도우만 head)도 twin에서 0.15 → 오탐의 대부분은 head가 hard-negative twin을 못 가르는 것이고, 32 윈도우 max 집계가 3–5 pt를 더한다. matched-FPR 5%로 맞추면 R@28k는 내려갈 것(oracle이 matched로 0.81 → LG도 0.8 근처 예상). G1 (c)(dense-FT FPR + 1pt)는 dense-FT 행이 나와야(3–4일) 판단 가능. 실패 시 계획서 §3(c): debounce/top-2 집계.
3. **손실 분해**: needle 단독 0.86 → oracle 윈도우 0.81(framing −4 pt, seed별 1–7 pt) → LG(라우팅 손실은 calib/main 행이 붙은 뒤). 라우터는 hit@32 0.975(최대 2.5 pt), k=8도 hit 0.94에 R −1.5 pt → 라우팅은 거의 공짜, **병목은 head(framing + twin 오탐)**. 계획서 §3(b)의 "head면 b=256"이 다음 손잡이.
4. **k**: k=32→16→8에서 R@28k 0.85→0.84→0.83, 비용은 같은 pass에서 유도(ms 동일). Table 2에 k=8 행을 주 결과로 쓸 수도 있음(ms/row는 아직 k=32 pass 기준).
5. **baseline 주의**: recall은 판정된 행만으로 계산 — WildGuard(28k 2,768/3,080 unscored), Granite(3,221/3,630), gpt-oss(≥4k 전부), 0.6B/Stream(≥16k 전부)은 "처리된 행 안"의 값. JSON의 `recall_effective`(unscored=miss)가 배포 관점 수치. 원인(OOM / 문맥 / 파싱)은 rows의 `error` 필드로 확인 필요.
6. **E1 parity**(`reports/s5/table1_e1.md`)는 만들어졌으나 아직 검토 전 — G1 (a)는 dense-FT W-avg F1이 있어야.

## 진행·남은 것

* e2 LG 3 seed: 20:22–20:26 완료 예정 → matched-FPR 행 → `e2 --models lg:…` 한 번 더(calib/main 12,000행, E2-7 LG 열) → `tables`.
* dense-FT 3 seed: 550/21,382 step, ETA 90–108 h(자동 requeue). 4B head(28,050/58,120 재개), 8B 체인: 대기.
* B1: b1-calib는 배열 목록 파일 충돌로 24 GB 12 task가 돌지 않음 → 수정 zip으로 재제출. LG4 v2main 행 없음(966381_0 25 h) → 로그 확인.

## B. 20:24 갱신 — matched-FPR 5% (calib benign-long) 행

| LG 0.6B k=32 | s0 | s1 | s2 |
| --- | --- | --- | --- |
| R@1k → R@28k (matched) | 0.893 → 0.850 | 0.889 → 0.831 | 0.915 → 0.850 |
| pos-min@28k | 0.660 | 0.665 | 0.660 |
| ε̂ | 0.044 | 0.058 | 0.064 |
| FPR τ=0.5 twin / bl @28k | 0.169 / 0.024 | 0.184 / 0.032 | 0.197 / 0.026 |
| FPR@τ1k twin / bl @28k (길이 팽창) | 0.351 / 0.084 | 0.352 / 0.130 | 0.366 / 0.102 |
| hit@32 @28k | 0.978 | 0.977 | 0.974 |
| k=8 R@28k / bl 팽창 | 0.841 / 0.058 | 0.813 / 0.098 | 0.834 / 0.072 |

E1 (Table 1): lg_k32 W-avg F1 0.807/0.816/0.806 (AUPRC 0.82) vs base Qwen3Guard-0.6B strict 0.768 / loose 0.819 (AUPRC 0.863); XSTest FPR 0.096/0.072/0.108, OR-Bench-Hard 0.062/0.057/0.055 vs base 0.220/0.732 (strict), 0.060/0.487 (loose); WGTest adv 0.834–0.840 vs 0.832/0.821.

E2-7 (모두 calib twin 임계값): needle 단독 0.855/0.840/0.875 → oracle 윈도우 0.78–0.82 → framing 손실 s0 5–7, s1 2–4, s2 7–9 pt. LG 열은 calib/main 재실행 뒤.

E2-8 집계 ablation (오프라인): second_max 붕괴(R@28k 0.53–0.61, needle은 한 윈도우), lse_0.5 발산(s1 k32 FPR 1.0), top2_mean = max와 같음, router_gated_3: bl 팽창 0.066/0.074/0.086, R@28k 0.847/0.858/0.850, R@1k 0.909/0.890/0.927 — 가장 좋음(선택은 calib/main 행에서 확정).

### 해석

* 0.19 "FPR"는 twin(hard benign 프롬프트)이었고 **긴 benign 문서(bl) FPR은 τ=0.5에서 2–3%**. LG의 hard-benign 과잉거부(XSTest 7–11%, OR-Hard 6%)는 base 가드보다 훨씬 낮음.
* 길이 의존 효과는 둘: (1) 팽창 — 1k 임계값을 28k에 두면 bl FPR 5% → 8–13%(라우터가 224개 중 가장 수상한 32개를 고르니 benign max가 L과 함께 오름), matched recall로는 ε̂ 4–6 pt. router_gated_3이 절반쯤 줄임. (2) framing — needle 뒤 배경 토큰을 지나는 마지막 토큰 probe가 2–9 pt 잃음(길이 무관).
* 다음 head 실험(하나로 둘 다 겨냥): 윈도우 내 토큰별 점수 max pooling(Stream head 방식) + 라우터 상위 benign 윈도우 hard-negative mining. dense-FT(3–4일)가 같은 팽창을 보이면 데이터 문제.
* G1 현황: (a) LG vs base −1 pt(dense-FT 대기), (b) LG R@28k 0.83–0.85 ≥ CD 0.71*(154행) 통과 중, (c) bl 팽창 0.08–0.13 vs dense-FT 대기.

## C. 9/21 11:52 갱신 — baseline matched-FPR, E2-7 완성, 집계 선택, framing 원인

**Table 2 (matched-FPR 5%, calib benign-long; 이진 판정 가드는 `*` 기본 판정 유지)**

| method | R@1k | R@28k | pos-min@28k | FPR@τ1k twin/bl @28k | ε̂ |
| --- | --- | --- | --- | --- | --- |
| LG 0.6B k=32 (s0/s1/s2) | 0.893/0.889/0.915 | 0.850/0.846/0.852 | 0.665/0.680/0.665 | 0.35/0.084, 0.35/0.130, 0.37/0.102 | 0.043/0.046/0.063 |
| LG k=32 router_gated_3 (calib 선택) | 0.909/0.890/0.927 | 0.854/0.853/0.854 | 0.695/0.720/0.670 | 0.34/0.066, 0.33/0.074, 0.36/0.086 | 0.055/0.037/0.074 |
| Qwen3Guard-8B native | 0.862 | **0.534** | 0.423 | **0.752/0.728** | 0.328 |
| Qwen3Guard-8B CD 512/256 (1,144행) | 0.870 | 0.773 | 0.712 | 0.554/– | 0.121 |
| Qwen3Guard-4B native | 0.826 | 0.365 | 0.235 | 0.733/0.760 | 0.460 |
| LlamaGuard3-8B | 0.381 | 0.182 | 0.045 | 0.013/0.006 | 0.200 |
| Qwen3Guard-0.6B native | 0.675 | 0.385@8k (≥16k 판정 없음) | | | |
| PolyGuard / WildGuard / Nemotron / Granite / GuardReasoner / gpt-oss / LG4 | 이진 판정 → `*` 행 유지 (0.863→0.818 / 0.787→0.010 / 0.717→0.075 / 0.371→0.167 / 0.641→0.340(부분집합) / 0.610→– / 0.250→0.000) | | | | |

읽기: 같은 규약으로 놓으면 dense Qwen3Guard-8B는 1k 임계값에서 28k benign 문서의 **73%**를 잡고(길이 팽창), 임계값을 L마다 맞추면 recall이 0.86→0.53. CD가 그걸 0.77까지 회복하지만 twin FPR 팽창 0.55. LG는 0.85, bl 팽창 8–13%(rg3로 7–9%). G1 (b) 통과(0.85 ≥ 0.77). (a)(c)는 dense-FT 대기.

**E2-7 완성 (calib twin 임계값)**: 28k에서 needle 단독 0.855/0.840/0.875 → oracle 0.784/0.798/0.789 → LG 0.784/0.744/0.754. framing 손실 7.1/4.2/8.6 pt 중 harmful-side 1.5/1.0/2.5, benign-side 5.5/3.2/6.2; routing 손실 0.0/5.4/3.5 (L 평균 ~1/4/3 pt). twin 창의 tail별 점수(0–15: 0.10–0.12 → 16+: 0.13–0.17)가 needle과 같은 방향으로 오르므로, benign-side의 상당 부분은 "문서가 benign을 유해하게 보이게" 하는 게 아니라 **probe 위치 효과**(단독 needle 행은 전부 tail 0 → 점수 눌림 → 임계값이 낮게 잡힘).

**E2-8 집계 선택 (calib/main)**: 세 seed 모두 router_gated_3 (k32, k32, k8). test: R@28k 0.854/0.853/0.854, bl 팽창 0.066/0.074/0.086.

**설계 결론**: 마지막 토큰 probe는 needle 뒤 ≥16토큰이 필요(문서 끝 depth 1.0 = pos-min 0.66의 정체). 창 끝 고정 landmark/suffix 토큰(또는 토큰별 max)으로 head를 다시 학습하면 pos-min ~0.66→~0.85, benign-side 3–6 pt 일부 회복 예상 — 주 결과가 아니라 ablation(head v2, s0 먼저 ~1일)으로.

## D. 9/22 16:05 — head v2 (landmark probe, `qwen3guard_0.6b_lm_s0`) 결과: **승격하지 않음**

| | v1 s0 (last-token) | v2 lm_s0 (landmark) |
| --- | --- | --- |
| Table 2 R@1k / R@28k (matched bl) | 0.893 / **0.850** | 0.902 / **0.820** |
| pos-min@28k | 0.665 | 0.635 |
| ε̂ | 0.043 | 0.082 |
| FPR@τ1k twin/bl @28k | 0.351/0.084 | 0.330/0.116 |
| FPR τ=0.5 twin/bl | 0.169/0.024 | 0.134/0.026 |
| rg3 R@28k | 0.854 | 0.819 |
| Table 1 W-avg F1 / AUPRC / OR-Hard FPR | 0.807 / 0.819 / 0.062 | 0.816 / 0.831 / 0.044 |
| E2-7 @28k needle→oracle→LG | 0.855→0.784→0.784 | 0.885→0.820→0.802 |
| framing 손실 = harmful + benign | 0.071 = 0.015 + 0.055 | 0.065 = 0.064 + 0.002 |
| valcheck hit@8 / recall@5pct | 0.974 / 0.915 | 0.971 / 0.906 |
| framing 진단 tail 0–15 / 128+ (창) | 0.726 / 0.855 | 0.729 / 0.837 |
| twin 창 tail 0–15 / 16–63 / 64–127 / 128+ | 0.100 / 0.136 / 0.147 / 0.131 | 0.097 / 0.103 / 0.120 / 0.098 |

읽기:
* 짧은 프롬프트(E1)는 v2가 약간 낫고(+0.9 F1, 과잉거부 ↓), twin 점수는 위치에 무관하게 낮아짐(benign-side framing 손실 0.055 → 0.002). 즉 landmark는 "읽는 위치" 문제는 해결함.
* 그러나 needle 창의 **tail 0–15 결손은 그대로**(0.729 vs 나머지 0.83) → 원인은 probe 위치가 아니라 **왼쪽 문맥 희석**: tail이 작을수록 needle 앞에 benign 배경이 많고(256 − n − tail), needle 토큰의 표현이 그 배경에 희석됨. v1에서 뒤 토큰이 도움이 된 건 needle을 "최근 내용"으로 다시 읽어 회복했기 때문이고, landmark 한 자리는 그 회복을 못 함(harmful-side 0.015 → 0.064).
* 긴 L에서 v2가 더 나쁨(R@28k −3 pt, ε̂ 0.082, bl 팽창 0.116): 라우터가 고른 benign 창의 landmark 점수가 L과 함께 더 오름.
* 결론: **v1 유지**. v2는 Table 2/E2-7 ablation 행("end-of-window landmark probe")으로, 그리고 "문서 끝 needle의 결손은 왼쪽 문맥 희석"이라는 메커니즘 근거로 씀. 남는 손잡이는 창을 줄이는 것(b=64: 희석 ↓, 같은 FLOPs면 k=64) 또는 토큰 단위 감독(needle 토큰 자체를 배경 뒤에서도 유해로 유지하도록 학습) — S6 뒤 ablation 후보. 4B/8B는 v1 그대로 진행.
