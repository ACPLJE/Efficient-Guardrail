# S5 실행 순서 — E1 parity, E2-1·2 ≤28k, E2-7, CD·dense-FT baseline, LG 8B 학습, G1 판정

전제: S4 T1 통과(`python s4ctl.py status`에서 0.6B 세 seed `val: … PASS`). 모든 명령은 `/home/212aig08/lg_s1`에서. 결과 rows는 `$LG_OUT/eval/v2/<exp>/<method>__<run>.rows.jsonl`(행마다 E0-2 계보), 표는 `$LG_OUT/reports/s5/`.

| 계획서 | 명령 | 자원 | 산출물 · 통과 기준 |
| --- | --- | --- | --- |
| E1-1·2·3 parity | `python s5ctl.py e1 --models lg:qwen3guard_0.6b_s0,…,dense:…,base:qwen3guard_gen_0.6b --sbatch` | 모델당 GPU 1(24 GB), ~1 h | `eval/v2/e1/*.rows.jsonl`(7 벤치 + XSTest·OR-Bench-Hard, 풀의 test 소스 전 행) → Table 1 |
| E2-1·2 ≤28k | `python s5ctl.py e2 --models lg:…,dense:… --L le28k --sbatch` | 모델당 GPU 1, LG ~3–4 h(3090) | test main 26,400 + benign-long 3,000 + calib benign-long(임계값) → Table 2 ≤28k 열, Fig 3·4 데이터 |
| E2-7 3층 분해 | `python s5ctl.py e27 --runs qwen3guard_0.6b_s0,… --sbatch` | run당 GPU 2(oracle·needle) | needle 단독 → oracle 윈도우 → LG: framing 손실 / routing 손실. 세 층 모두 calib/main **twin**(같은 음성)으로 임계값 → LG e2 rows에 calib/main 12,000행이 포함돼야 함(9/20 이전에 만든 rows 파일은 `e2 --models lg:…`를 **잡이 끝난 뒤** 한 번 더: todo=12,000, ~1 h) |
| B2 CD | `python s5ctl.py b2 --sbatch` (+`--sweep`) | 48 GB 티어(8B) ~10 h | `longbench/v2main/qwen3guard_gen_8b.chunk_or_512_256.rows.jsonl` → Table 2 CD 행; sweep = W∈{256,1k,2k}×overlap{0,W/2}, LG3-8B·gpt-oss W=512, 128쌍×5위치 부분집합 |
| B1 calib | `python s5ctl.py b1-calib --sbatch` | 12종 × calib benign-long 3,000행, 각 ≤1 h | `longbench/v2calib/*.rows.jsonl` — zero-shot·CD 행도 matched-FPR 5%로(없으면 표에 `*` 기본 판정) |
| B4 dense-FT | `e2 --models dense:qwen3guard_0.6b_s0`(dense_ft.pt가 생기면) | 24 GB | Table 2 dense-FT 행 |
| T4 LG 8B | `python s5ctl.py t4 --sbatch` | 48 GB, head ~20 h(LoRA) → router → valcheck | `runs/v2/qwen3guard_8b_s0/…` |
| E2-8 집계 ablation | (`tables`에 포함, GPU 없음) | CPU | `agg_ablation_le28k.md` — 있는 LG rows의 윈도우별 head 확률로 max / top2_mean / second_max(debounce) / lse / router_gated × k∈{8,16,32}를 같은 규약으로 재평가 |
| framing 진단 | (`tables`에 포함, GPU 없음) | CPU | `framing_by_offset_le28k.md` — oracle 행의 needle 윈도우를 "needle 뒤 배경 토큰 수"(0–15/16–63/64–127/128+)로 묶어 head 확률·τ=0.5 통과율. 128+에서 떨어지면 마지막 토큰 probe의 pooling 문제 → 토큰별 head |
| 표 | `python s5ctl.py tables` (`--L all` = 64k·128k 행이 붙은 뒤, `s6ctl tables`가 같이 부름) | CPU | `reports/s5/table1_e1.md`, `table2_le28k.md`, `e27_decomposition_le28k.md`, `agg_ablation_le28k.md`, `framing_by_offset_le28k.md`, `fig3_heatmap_le28k.json`, `fig4_recall_vs_L_le28k.json` |

## 1. 순서대로

```bash
cd /home/212aig08/lg_s1
python s5ctl.py all --sbatch          # 준비된 것만: LG 0.6B 3 seed(+4B) e1·e2·e27, dense_ft.pt가 있으면 dense, base 0.6B, CD 기본 config, b1-calib
python s5ctl.py status                # rows 파일별 행 수 + 진행(running n/todo, ms/row, eta·예상 끝 시각 | done | queued) + s5_ 잡
python s5ctl.py tables                # 잡이 끝난 뒤(끝난 것만으로도 표는 나옴)
```

dense-FT가 나중에 끝나면 그 seed만:

```bash
python s5ctl.py e1 --models dense:qwen3guard_0.6b_s0 --sbatch
python s5ctl.py e2 --models dense:qwen3guard_0.6b_s0 --sbatch
python s5ctl.py tables
```

64k·128k 행(S6)은 `python s6ctl.py e2-long --models … --sbatch`가 **같은 rows 파일**에 이어 붙입니다(`docs/S6_runbook.md`; 그 seed의 S5 e2 잡이 끝난 뒤).

모든 잡은 id 단위로 이어서 돌므로(`done=… todo=…`), 죽었거나 일부만 됐으면 같은 명령을 다시 내면 됩니다. `status`가 `STOPPED (no job in queue) -> <log>.err`를 보이면 그 로그를 보세요. `LineageMismatch … model_sha`(9/21: dense-FT가 같은 run 디렉터리에 `dense.resume.pt`를 쓰면서 디렉터리 전체 지문이 바뀐 것)면 `python s5ctl.py relineage --exp e2 --models lg:…` 한 번(rows의 model_sha를 head.pt+router.pt 지문으로 고쳐 씀; router_sha가 다르면 거부) 뒤 같은 e2 명령을 다시.

## 2. 무엇을 계산하나

* **모델 지정** `lg:<run>[:kK]`(기본 k=32; k=8·16은 k=32 rows의 라우팅 순서별 head 확률에서 표가 유도), `sliding:<run>`(B5, 모든 윈도우), `oracle:<run>`(needle이 든 윈도우만 head — 라우터를 정답으로 대체), `needle:<run>`(needle 원문만), `dense:<run>`(dense_ft.pt, 전체 문맥 ≤32k, 마지막 토큰 probe), `base:<guard>`(zero-shot; Qwen3Guard는 strict/loose 두 행).
* **임계값**(E0-3): 기본 τ=0.5(prob) 판정과, calib **benign-long**의 (method, L)별 엄격 quantile(FPR 5%·1%, 달성 FPR 기록) matched-FPR recall. zero-shot·CD 행(longbench rows)은 `b1-calib`의 calib benign-long 점수가 있으면 같은 규약으로, 없으면 기본 판정만(표에 `*`). 점수가 이진(파싱한 판정 1/0: GuardReasoner·gpt-oss·LG4)인 가드는 quantile이 무의미하므로 calib이 있어도 기본 판정을 유지(`*`, JSON `calib_binary`). recall·FPR은 **판정이 있는 행**만으로 계산하고, `unscored@max L` 열 = 판정이 없는 행(OOM·문맥 초과·파싱 실패)/전체 — unscored가 큰 행(granite·wildguard·gpt-oss)은 그 recall이 "처리된 행 안에서의" 값임.
* **Table 2 열**: R@L = matched-5% recall의 위치 평균(11 depth), pos-min, FPR twin/bl(기본 판정; twin = 같은 배경 안의 benign twin — 같은 풀의 길이 맞춘 benign 프롬프트(XSTest·OR-Bench-Hard·WGTest adversarial-benign 포함), bl = needle 없는 긴 benign 문서), **FPR@τ1k twin/bl** = 1k에서 잡은 matched-5% 임계값을 max L에 그대로 적용했을 때의 FPR(길이에 따른 오탐 팽창; G1 (c)는 이 값으로 dense-FT와 비교), ε̂ = max_L[R(1k)−R(L)], hit@k, ms/row. Fig 3 = (L × depth) recall 히트맵 JSON, Fig 4 = recall vs L JSON.
* **E2-7**: (i) needle 단독 recall(길이 불변) → (ii) oracle 윈도우 → (iii) LG top-k; framing 손실 = (i)−(ii), routing/희석 손실 = (ii)−(iii). 세 층의 임계값은 모두 calib/main의 benign twin(L별 2,000행)에서 — needle 단독·oracle은 benign-long을 점수화할 수 없고, 층마다 다른 음성으로 맞추면 쉬운 음성을 쓴 층이 유리해지므로. 그래서 LG e2 rows에는 calib/benign_long(Table 2용) + calib/main(E2-7용)이 둘 다 들어감.

## 2.5 FPR가 높게 보일 때 (9/20: LG τ=0.5에서 twin+bl FPR 0.17–0.20)

τ=0.5는 운영점이 아님(E0-3: 임계값은 calib benign-long의 L별 5% quantile). 순서대로 보기:

1. `table2_le28k.md`의 FPR twin/bl — twin 쪽이 높으면 hard-negative(benign이지만 민감 주제 프롬프트) 과잉거부, `table1_e1.md`의 XSTest·OR-Bench FPR과 같은 현상. bl 쪽이 높으면 평범한 긴 문서를 잡는 것 → head 문제.
2. FPR@τ1k twin/bl — 1k 임계값이 28k에서 몇 pt 오르는지(길이 팽창). JSON `fpr@tau1k` per L.
3. `agg_ablation_le28k.md` — max 대신 second_max(두 윈도우가 동시에 울려야) / top2_mean / router_gated로 바꾸면 같은 matched recall에서 팽창이 얼마나 주는지(오프라인, GPU 없음). 집계는 **calib/main 행에서 고르고**(표의 `calib R`·`calib infl bl`, `**` 표시 = 선택) test로 보고. 9/20 결과: second_max는 recall 붕괴(needle은 한 윈도우), lse는 발산, router_gated_3이 bl 팽창을 0.08–0.13 → 0.07–0.09로 줄이며 R@28k 유지/상승. 좋은 집계가 있으면 `lgs4`의 verdict 한 줄 + Table 2 ablation 행.
4. 그래도 남으면 head 재학습(계획서 §3). 9/20 진단: bl FPR(τ=0.5) 0.02–0.03이라 평범한 긴 문서는 문제없고, 남는 건 (i) framing 손실 2–9 pt(needle 단독 → 정답 윈도우; 마지막 토큰 probe가 needle 뒤의 배경 토큰을 지나며 신호를 잃는 구조) → 윈도우 안 토큰별 점수의 max(Stream head 방식)로 pooling, (ii) 길이 팽창(1k 임계값이 28k에서 bl FPR 5%→8–13%, matched recall ε̂ 4–6 pt; 라우터가 224개 중 가장 수상한 32개를 고르니 benign max가 L과 함께 오름) → head 학습의 benign 윈도우를 라우터 상위 윈도우에서 뽑는 hard-negative mining, (iii) twin FPR은 hard benign 프롬프트 과잉거부(Table 1 XSTest 7–11%, OR-Hard 6%로 base 가드보다 낮음)라 별도 축. dense-FT 행이 같은 팽창을 보이면 데이터 문제이지 구조 문제가 아님.

## 3. G1 판정 (10/1)

`reports/s5/table1_e1.md`·`table2_le28k.md`에서:

* (a) E1-1: LG 0.6B(3 seed 평균) W-avg F1 ≥ dense-FT W-avg − 1pt
* (b) 28k: LG k=32 R@28k(matched-FPR 5%) ≥ CD 기본(`qwen3guard_gen_8b/chunk_or_512_256`) 행의 R@28k(b1-calib 뒤로는 같은 matched-FPR; 9/21: LG 0.85 vs CD 0.77 → 통과 중). Table 2의 `lg_k32_rg3` 행 = calib/main에서 고른 router_gated_3 집계(같은 rows에서 유도).
* (c) 28k: LG FPR@τ1k@28k ≤ dense-FT FPR@τ1k@28k + 1pt (둘 다 1k에서 잡은 matched-5% 임계값을 28k에 적용 — 길이 팽창 비교; 기본 판정 FPR은 참고)

실패 시 계획서 §3: (a) native 비중↑·Stream head 초기화·학습률, (b) E2-7로 router(hit@k)/head 분리 → router면 k=64·listwise 가중, head면 b=256, (c) debounce/top-2 집계.
