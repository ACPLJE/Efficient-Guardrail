# S6 실행 순서 — E2-1·2 64k·128k, E2-4 희석, E2-6, E3-1·2·3 FPR, B1·B2 장문 baseline, B4·B5·B6·B7, T5·T6 (+ S7 E4 비용)

전제: S5 rows(`eval/v2/e2/*.rows.jsonl`)가 있고 S5 규약이 그대로 이어집니다. 모든 명령은 `/home/212aig08/lg_s1`에서. S6 코드는 S5 결과값에 의존하지 않으므로(9/22) S5 잡이 도는 동안 제출해도 됩니다 — 단 같은 rows 파일에 쓰는 잡(같은 `--models`의 e2)은 S5 e2 잡이 끝난 뒤에(`s5ctl status`가 done).

| 계획서 | 명령 | 자원 | 산출물 · 통과 기준 |
| --- | --- | --- | --- |
| E2-1 64k·128k (LG) | `python s6ctl.py e2-long --models lg:qwen3guard_0.6b_s0,lg:qwen3guard_0.6b_s1,lg:qwen3guard_0.6b_s2 --sbatch` | 24 GB, seed당 ~6–8 h(3090) | S5 rows 파일에 64k·128k 행이 **이어서** 붙음(test main 8,800 + benign-long 1,000 + calib bl 1,000 + calib main 4,000). 128k는 윈도우 스트리밍(`forward_stream`: 64창씩 trunk → 러닝 top-k만 보관, 메모리 평평) — 같은 판정을 `tests/test_lgcore_stream.py`가 검증 |
| B4 dense-FT 64k·128k | `--models dense:qwen3guard_0.6b_s0,dense:qwen3guard_0.6b_s0:tail,dense:qwen3guard_0.6b_s0:yarn` (dense_ft.pt가 생기면) | head/tail 24 GB, yarn 48 GB | `dense_ft`(앞 32k 유지 = 계획서 truncation 앞), `dense_ft_tail`(뒤 32k), `dense_ft_yarn`(YaRN ×4, 128k 전부) 세 행 |
| B5 sliding | `--models sliding:qwen3guard_0.6b_s0` (`e2-long` 또는 `s5ctl e2 --L all`) | 24 GB, ~4× LG | Table 2 sliding 행(같은 head를 모든 창에) |
| E2-2 전 길이, E2-6 ε̂·pos-min | `python s6ctl.py tables` | CPU | `reports/s5/table2_allL.md`·`fig3_heatmap_allL.json`·`fig4_recall_vs_L_allL.json`(8 L), ε̂·pos-min 열 자동, `reports/s6/g2_check.md` = G2 (a) 128k pos-min ≥ R(1k) − 0.05 |
| E2-4 희석 통제 | `python s6ctl.py var --models lg:qwen3guard_0.6b_s0,dense:qwen3guard_0.6b_s0 --sbatch` (+ `--subsets needle_repeat`만도 가능) | 24 GB, ~2 h | `eval/v2/var/*.rows.jsonl`(needle_repeat·decoy·framed·ood_domain, L 4k/28k/128k) → `reports/s6/fig5_dilution.md`: (a) Benign-Fill(test/main) vs Needle-Repeat recall(같은 calib 임계값), (b) needle 길이 버킷별 recall(needle share) |
| E3-1 장문 benign FPR | (e2 rows에서 자동; baseline은 `b1-long`) | — | `reports/s6/fig6_fpr.md`: benign-long FPR을 in-domain(학습 배경 5) / OOD(wiki_en·wiki_zh·pubmed·agentdojo·ledgar) / 도메인별로, 기본 판정과 matched-5%; 64k LG ≤ dense + 1pt |
| E3-2 decoy, E3-3 framed | (`var` rows에서 자동) | — | `fig6_fpr.md`의 decoy(소스별)·framed(frame별) 열, `table_variants.md` |
| B1 64k·128k | `python s6ctl.py b1-long --sbatch` | 48 GB 티어, ≈90 h 합계 | zero-shot 12종: 128쌍 × 5위치 × 2 L + benign-long 250/L(`--max-pairs 128 --depths 0,0.25,0.5,0.75,1 --max-bl 250`), 정책은 모델별(`long_policies`): 32k Qwen 계열 = **yarn + trunc 앞/뒤**, WildGuard·Stream·≤8k 모델 = trunc 앞/뒤, 128k 네이티브(LG3-8B·GuardReasoner·Granite·gpt-oss·LG4) = native + **chunked prefill 8k**(`lgs1/zeroshot/prefill.py`; 48 GB에서 8B–20B 128k). 같은 정책으로 calib benign-long 250/L(`v2calib`)도 → matched-FPR |
| B2 CD 64k·128k | `python s6ctl.py b2-long --sbatch [--sweep]` | 48 GB, ~25 h/config | 8B W=512 o=256 부분집합 + calib; sweep = W∈{256,1k,2k}×{0,W/2} + LG3-8B·gpt-oss |
| B1 변형 서브셋 | `python s6ctl.py b1-var --sbatch` | 24/48 GB | Qwen3Guard-0.6B·8B·LG3-8B + CD를 `v2:needle_repeat,decoy,framed,ood_domain`에(4k·28k native, 128k는 장문 정책) → `longbench/v2_needle_repeat+decoy+framed+ood_domain/` → fig5·fig6·table_variants의 baseline 행 |
| B6 소형 인코더 | `python s6ctl.py b6 --sbatch` | 24 GB | Prompt Guard 2-86M 512 창 chunk-OR(≤28k 전량 + 64k·128k 부분집합 + calib) |
| B7 cascade | (`tables`가 유도, GPU 없음) | CPU | `reports/s6/table2_cascade.md`: LG k=32의 head 확률이 calib에서 고른 순위 밴드(문서의 5/10/20 %가 에스컬레이션, `@e05/@e10/@e20`) 안이면 Qwen3Guard-8B 행(같은 id; ≤28k native, 64k·128k yarn → trunc)의 판정으로 — test·calib 모두 유도해 자체 matched 임계값, 에스컬레이션 비율·ms 기록 |
| T5 LG3-8B | `python s6ctl.py t5 --sbatch` | 48 GB, prep(CPU) → head ~20 h(LoRA) → router → valcheck | `s4/prep_llama3/`(같은 행을 Llama-3 토큰으로; D8 `tok/llama3` 스팬과 대조) → `runs/v2/llamaguard3_8b_s0/`. S5·S6 평가는 `lg:llamaguard3_8b_s0` |
| T6 dense-FT 8B | `python s6ctl.py t6 --sbatch` | 48 GB ~25 h | `runs/v2/qwen3guard_8b_s0/dense_ft.pt`(LoRA 전 층, ≤8k, 8B head.pt와 토큰 예산 일치 → head 잡에 afterok) |
| 상태 | `python s6ctl.py status` | — | e2 파일별 64k 이상 행 수 + 진행(`running n/todo`·`STOPPED`), var 파일, longbench 디렉터리별 장문 행 수, T5·T6 run 디렉터리 |

## 1. 순서대로

```bash
cd /home/212aig08/lg_s1
python s6ctl.py all --sbatch              # 준비된 것만: e2-long(LG 3 seed + 4B/8B/LG3-8B가 있으면 + dense 3 정책 + sliding s0), var(s0 + 큰 백본 + dense s0), b1-long, b2-long, b1-var, b6
python s6ctl.py all --sbatch --with-t5 --with-t6   # 학습 체인까지
python s6ctl.py status
python s6ctl.py tables --sbatch           # 잡이 끝난 뒤(끝난 것만으로도 표는 나옴) = s5ctl tables --L all + S6 표; CPU 잡(로그인 노드에선 ~2M 행에 10분+)
```

S5 e2 잡이 아직 도는 seed는 `e2-long`을 그 seed만 나중에(`--models lg:qwen3guard_0.6b_s2`). 모든 잡은 id 단위로 이어서 돌므로 죽으면 같은 명령을 다시. `tables --L all`의 Table 2는 LG rows 파일 하나에 8 L이 다 들어 있어야 한 행으로 나옵니다(`s6ctl status`의 "rows at L >= 64k"가 seed당 14,800이면 끝).

## 2. 무엇이 어떻게 계산되나

* **128k LG**: `LandmarkGuard.forward(stream=True)` = `forward_stream` — 64창 단위로 trunk(layers < ℓ) → 라우터 점수 → 러닝 top-k 버퍼(k개 창의 trunk 상태만 보관) → 마지막에 deep head. 판정·점수·선택 창이 비스트리밍과 같고(`tests/test_lgcore_stream.py`), 메모리는 L과 무관(E4-3, G2 (c)). `lgs5.run --stream auto`(기본)는 32k 초과 행에만 스트리밍, `on/off`로 강제. `sliding_scan`도 창 단위 스트리밍(head를 모든 창에, 메모리 평평·시간 ∝ L). oracle 행은 needle 창만 trunk에 넣음(block-local이라 전체 통과와 같은 상태 — 128k oracle 행이 1k 값과 같은 비용).
* **dense 정책**(`dense:<run>[:head|tail|yarn]`): 창 밖 문서에 대해 앞 32k 유지(기본, method `dense_ft`) / 뒤 32k(`dense_ft_tail`) / YaRN ×4(`dense_ft_yarn`, `lgs4.backbone.apply_yarn`; transformers 4.x `rope_scaling`·5.x `rope_parameters` 모두). ≤28k 행은 세 정책이 동일. 32k 토큰을 넘는 행은 KV 캐시로 2,048 토큰씩 chunked prefill(`dense_logits_chunked`; `--prefill-chunk`, 8B는 1024) — 단일 패스와 같은 마지막 토큰 상태이고 attention 메모리가 O(chunk × L)라 fused 커널이 안 잡히는 행에서도 [L×L] 마스크를 만들지 않음(9/23 OOM). 배치 묶기는 실제 토큰 수 기준(CJK 배경은 chars/3 추정이 3–4배 과소).
* **변형 rows**(`--exp var`): `synth/test/<subset>`의 행을 그대로(`needle_char_spans` 여러 개면 모두 토큰 스팬으로 → `needle_windows`는 합집합, `hit`는 그중 하나라도 라우팅). rows에 변형 축(`n_decoys`·`n_pieces`·`frame`·`multi_config`·`decoy_source`·`n_repeats`·`needle_share`·`n_tok_needle`)이 실려 표가 축별로 쪼갬. S8–S9의 fragmented·decoy_flood·multi_needle·long_needle·related도 같은 명령(`--subsets`)으로.
* **임계값**: 변형·장문 행의 matched-5% 임계값은 e2 rows의 calib benign-long **같은 L**(4k/28k/128k)에서 — 그래서 `var` 잡은 calib 행을 만들지 않고, 128k 변형 행은 `e2-long`이 끝나야 matched 열이 채워짐(그 전엔 `-`). 이진 점수 가드는 기본 판정 유지(`*`).
* **B1 장문 정책**(`s6ctl.long_policies`): Table 3의 "128k 처리" 열이 그대로 rows의 policy. chunked prefill은 계산이 같은 greedy 디코딩(캐시에 8k씩 prefill 후 `generate`가 이어감; `tests/test_s6_s7.py::test_chunked_prefill_matches_single_pass`)이라 cfg_hash에 들어가지 않고 행에 `prefill_chunk`만 기록.
* **B7 cascade**(`lgs6.tables.derive_cascade`): 오프라인 유도라 GPU가 안 들지만 8B 행이 있는 id에서만(64k·128k는 부분집합) — 표의 n이 줄어드는 이유. 밴드는 `cascade_bands`: L마다 calib 행(benign-long + main)의 점수 순위에서 matched-5% 임계값을 가운데 두고 비율만큼 잡음.
* **G2 판정(10/7)**: (a) `reports/s6/g2_check.md`, (b)(c) `reports/s7/g2_cost.md`(아래 S7).

## 3. S7 E4 비용 실측 (`s7ctl.py` + `lgs7/`)

```bash
python s7ctl.py e4 --sbatch                       # 계획서 방법 목록: LG 0.6B k8/16/32·noskip·pad·nostream, sliding, dense-FT yarn, LG 8B/LG3-8B/4B(있으면),
                                                  # zero-shot 0.6B/8B yarn·LG3-8B·gpt-oss·GuardReasoner native, CD 512/1k(8B), Prompt Guard chunk — latency+memory 전 방법,
                                                  # breakdown(LG), ksweep(LG 0.6B), throughput(LG·dense·CD); 잡마다 --exclusive
                                                  # 잡마다 --exclusive, 기본 --partition RTX6000ADA(한 GPU 종류; LG_S7_PARTITION)
python s7ctl.py e4 --sbatch --partition L40S      # 다른 GPU(3-GPU 소패널): 파티션 고정, rows는 cost/<장치명>/ 로 따로
python s7ctl.py cost --methods lg:qwen3guard_0.6b_s0,sliding:qwen3guard_0.6b_s0 --exps latency,memory,breakdown --sbatch   # 개별
python s7ctl.py tables                            # reports/s7/fig7_{latency,throughput,memory}.json, fig9.json, table5_cost.md, g2_cost.md
python s7ctl.py status
```

* `lgs7.cost`는 방법 하나·GPU 하나가 한 프로세스: L마다 test/benign_long 문서 20개(없으면 무작위 토큰)로 워밍업 5회 뒤 20회(CUDA event + synchronize), p50/p90/p99/mean ms(모델 시간; 토큰화는 `ms_tokenize`로 따로), `peak_alloc_mb`(측정 직전 `reset_peak_memory_stats`, 적재 후 휴지 메모리 대비 `peak_delta_mb`), 행마다 GPU·드라이버·CUDA·torch·transformers·dtype·attention 구현·배치·반복 횟수·코드 커밋·run 계보. 같은 key(exp|method|L|batch|k)는 건너뜀(`--force`로 다시).
* 방법 문자열: `lg:<run>[:kK][:noskip][:pad][:nostream]`(noskip = 모든 창이 deep layer까지, E6-13; pad = 창 수 < k면 head 배치를 k로 채움, E5-5; nostream = 모든 창 상태 보관), `sliding:<run>`, `dense:<run>[:head|tail|yarn]`, `base:<guard>[:native|yarn|trunc_keep_head|trunc_keep_tail]`, `cd:<guard>[:W[:overlap]]`.
* E4-2 throughput은 `forward_batch`(문서별 스트리밍 trunk·라우팅 → 모든 문서의 라우팅 창을 head 배치로) — 판정은 문서별 스트리밍과 같음(테스트). dense는 패딩 배치, OOM은 행에 기록하고 더 큰 배치는 건너뜀.
* E4-4 breakdown은 `PhaseTimer`(단계 경계마다 synchronize)로 tokenize/trunk/router/head/aggregate ms·% (3회 평균) — latency 행과는 별도 측정.
* Table 5의 `budget x@128k` = LG 환산 토큰 예산 ℓ/N + k·2b·(1−ℓ/N)/L (E4-7). G2 (b) 128k p50 LG ≤ sliding/4, (c) peak 1k→128k ≤ +10%는 `g2_cost.md`.

## 3.5 9/23 결과 메모 (S6 첫 표)

* **128k 하락의 정체**: LG k=32 R@1k 0.89–0.92 → R@128k 0.76–0.78, pos-min 0.42–0.48 → G2 (a) FAIL. heatmap: depth 0–0.9는 0.91 → 0.87 → 0.81(1k/28k/128k)로 고르게, depth 1.0만 0.81 → 0.67 → 0.43. E2-7 64k·128k: oracle 창 0.80 평평(block-local), framing 2–8 pt L 무관, **(ii)→(iii) 손실만 0–5 → 9–13 pt로 증가**하는데 hit@32@128k = 0.95 → 잃는 건 32창 max 집계의 임계값 팽창(benign 문서도 32창 max). 어떤 집계도 되돌리지 못함(rg3 0.80/0.58이 최선, top2_mean은 64k부터 붕괴 0.67/0.56). 대응: S8 hard-negative head 재학습(팽창), rg3 주 구성, depth 1.0은 tail-pad 검증 후보.
* **G2 (b)**: ms@128k LG 1,151 vs sliding 3,388 = 2.9× = N/ℓ 상한(28/9) → "sliding ÷ 4"는 ℓ=9로 불가; CD 58.5 s(51×)·8B yarn 94.6 s(82×) 대비로 기준을 다시 씀(E4 실측에서 확정).
* **변형**: 8B는 Needle-Repeat 0.93 유지 vs Benign-Fill 0.53 → 순수 희석; LG는 둘이 같음. decoy FPR LG 0.11–0.20 vs 8B 0.43–0.77. framed LG 0.87 → 0.81(quote 최저), 8B 0.75 → 0.58. OOD 배경 LG 0.84–0.91 ≈ in-domain(agentdojo FPR 0.34).
* **cascade**: 고정 밴드 (0.2, 0.8)은 0.1–0.3%만 에스컬레이션(LG 확률이 양극단) → calib에서 고른 순위 밴드(`cascade_bands`, @e05/@e10/@e20 = 문서의 5/10/20 %)로 바꾸고, 64k·128k는 8B native 행이 없으니 yarn → trunc 순으로 대체(`BIG_METHODS`).
* **dense yarn 128k OOM**: `scaled_dot_product_attention`이 [L×L] 마스크 28 GiB를 잡음(fused 커널이 안 잡힌 행) → dense 행이 32k 토큰을 넘으면 KV 캐시로 2,048 토큰씩 chunked prefill(`lgs4.backbone.dense_logits_chunked`; 같은 마지막 토큰 상태, 테스트로 검증). `e2-long --prefill-chunk 1024`는 8B dense용. 모든 S6 잡에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* 운영 메모: relineage는 파일 종류마다(oracle·needle도) 해야 함; `s6ctl status`가 이제 s5ctl로 낸 oracle/sliding 잡의 로그도 읽음.
* **`s7ctl e4` 9/23 버그**: 주 구성 필터가 `k32:noskip/pad/nostream`(콜론 3개)을 잡아 breakdown/ksweep/throughput이 그 셋에 돌고 각각 3번 더 제출됨(991469–71, 991474–79 취소). 이제 `E4_MAIN = lg:qwen3guard_0.6b_s0:k32`에만 돌고 `e4_plan`이 (method, exp) 중복을 막음; 빠진 셋은 `python s7ctl.py cost --methods lg:qwen3guard_0.6b_s0:k32 --exps breakdown,ksweep,throughput --sbatch`로 따로 냄.
* **9/24 E4 GPU 섞임**: 파티션을 티어 목록에 맡겨 24 GB 티어 잡이 A5000에, 48 GB 티어가 RTX 6000 Ada에 떨어져 rows가 `cost/rtxa5000/`·`cost/rtx6000adageneration/`로 갈림 — 표·G2 (b)(c)는 GPU 디렉터리 단위라 한 GPU에 전 방법이 있어야 함(계획서 GPU D = RTX 6000 Ada). 이제 S7 잡은 기본 `--partition RTX6000ADA`(`LG_S7_PARTITION`; `--partition any`가 티어 목록, L40S·A100 소패널은 `--partition L40S` 등)이고, A5000에서 돈 방법(k8/k16/noskip/pad/nostream/sliding/LG3-8B·GuardReasoner native/Prompt Guard + 주 구성 breakdown·ksweep·throughput)은 RTX6000ADA로 다시 냄(다른 GPU 디렉터리라 `--force` 불필요). 24 GB에서 돈 8B native 행은 device_map 오프로드 가능성이 있어 쓰지 않음.
* **9/24 cost 오류 2건**: Prompt Guard chunk(`cd:prompt_guard_2_86m:512:0`)는 분류기 파이프라인이 토크나이저를 모델에 들고 있어 `BaseCost.ctok`가 None → 전 L 오류 → longbench와 같은 fallback(`getattr(model, "tokenizer")`). 128k-native base 가드(LG3-8B·GuardReasoner·gpt-oss)는 B1과 같은 `--prefill-chunk 8192`를 기본으로(`_prefill_for`; 48 GB에서 128k OOM 방지).
* **9/24 `--exclusive` = 노드의 GPU 8장이 다 보임**: `base:qwen3guard_gen_0.6b:yarn` 전 L 오류(`index is on cuda:0 … other tensors on cuda:7`)의 원인 — transformers `device_map="auto"`가 보이는 GPU 전부에 가드를 균등 분할해서 8B·20B base/cd 잡도 8-GPU 파이프라인으로 재던 셈(LG·dense 드라이버는 `device="cuda"` 단일이라 무관). 이제 S7 sbatch가 첫 GPU만 남기고(`GPU_ENV`), `lgs7.cost`가 import 시점에 `CUDA_VISIBLE_DEVICES`를 하나로 줄이며(`_one_gpu`; 미설정이면 nvidia-smi로 확인), 그래도 2장 이상 보이면 중단. 행에 `n_gpus_visible`·`cuda_visible_devices` 기록. 그 전에 잰 base/cd 행(8B yarn·cd512·cd1024·A5000의 LG3-8B·GuardReasoner)은 `--force`로 다시 잼(같은 key의 마지막 행이 이김).
* **E2-7 rg3 열**: `e27_decomposition_*.md`에 `LG rg3`(같은 rows의 router_gated_3, 자체 twin 임계값)와 `routing loss (rg3)` 열 추가 — 주 구성 후보의 (ii)→(iii) 손실을 max 집계와 나란히.
* **9/25 E4 첫 표(RTX 6000 Ada, batch 1)**: LG 0.6B k32 p50 26 / 358 / 1,351 ms @1k/28k/128k(p99 1,407), 79k tok/s@32k, 예산 x@128k 0.364, peak 1,203 → 1,680 MB; sliding 3,857 ms → **LG/sliding = 0.35** = ℓ/N + (k/N_w)(1−ℓ/N) = 0.32 + 0.02의 하한 → G2 (b)(≤ 0.25) FAIL은 구조적(트렁크 9/28층이 하한; 통과하려면 ℓ ≤ 6). dense/CD 대비: 0.6B yarn 16.1 s(12×), 8B yarn 50.3 s(37×, peak 45 GB), CD-512 8B 82.5 s(61×), LG3-8B native 80.2 s, GuardReasoner 387 s, Prompt Guard chunk 2.3 s(1.7×), gpt-oss 1k만(4k부터 OOM). LG 4B 8.5 s, LG 8B 14.9 s(8B yarn의 3.4×). k8/k16 ≈ k32(head 비용 작음), noskip 3.9 s ≈ sliding, nostream 3.5 GB. **G2 (c) FAIL(+40 %)**: 1k → 128k 증가분은 64창(16k 토큰) chunk 하나의 활성 메모리(transient)라 L이 아니라 chunk 크기에 비례 → 계획서 fallback대로 chunk를 k 창 이하로 고정해 재측: `lg:<run>:k32:c8|c16|c32`(태그 `_c8`, method `lg_k32+c8`; g2_cost가 함께 판정). A5000 패널의 LG3-8B·GuardReasoner 행은 8-GPU 분할 전 측정이라 무효.
* **9/27 G2 (c) PASS**: `lg_k32+c16+h8` peak 1,203 → 1,306 MB(+8.6 %), 128k p50 1.10 s(c64 대비 −19 %); h16도 동일. 주 구성 = `k32:c16:h8`(`s7ctl.E4_CFG`; `e4`가 이 구성으로 목록을 냄) → breakdown·ksweep·throughput과 k8/k16/noskip/pad, 4B·8B·LG3-8B latency/memory/breakdown을 같은 구성으로 재측해 Table 5를 한 구성으로 맞춤. (b)는 같은 chunk의 sliding(`sliding+c16`)과 비교(`g2_cost`가 자동).
* **9/26 chunk 스윕**(RTX 6000 Ada, k32): c64 1,351 ms·+40 % → c32 1,174·+19 %, c16 **1,069**·+17 %, c8 1,159·+17 % — 작은 chunk가 더 빠르고(16k 토큰 chunk가 비효율) 메모리 증가엔 +17 %(≈ 200 MB) 바닥이 남음 = k=32 창을 한 번에 넣는 deep head 배치의 활성 메모리. → `forward_stream(head_batch=N)`(창별 독립이라 로짓 동일, 테스트)·cost 스펙 `:hN` 추가; 주 구성 후보 `lg:<run>:k32:c16:h8`(예상 +5 %). sliding도 같은 chunk(`sliding:<run>:c16`)로 재야 (b) 비율이 공정(c64 sliding 대비 0.277은 sliding 쪽 비효율이 섞인 값; FLOPs 하한 0.32).
* **9/25 시간 한도 자동 이어달리기**: eval(`s5ctl e1/e2`, `s6ctl e2-long/var`)과 longbench 배열(`s3ctl longbench`, `s6ctl b1-long/b1-var/b6/b2-long`) 잡이 `--signal=B:TERM@300|900 --requeue`로 나가고, `lgs1.interrupt.Stop`이 SIGTERM을 받으면 현재 행까지 쓰고 rc=3 → 같은 job id로 requeue(id 재개). 그 전 잡(yarn s1/s2 36 h, LG 8B e2 24 h, GuardReasoner 48 h)은 한도에서 죽어 손으로 다시 냈음.
* **gpt-oss-20b**: eager attention([64 × L × L] bf16)이라 48 GB에서 4k부터 OOM(≤28k rows: 1k 정상, 2k 97 %, 4k 이상 전부 OOM 행). Table 2엔 1k·2k, Table 3 처리 열 "OOM (48 GB)"; 64k·128k 안 돌림. `gpt_oss_safeguard_20b.native.rows.jsonl`은 동시 쓰기로 깨진 1줄을 제거(백업 `.bak0924`). GuardReasoner-8B는 5.8 min/행(128k)이라 부분집합(`--max-pairs 64 --depths 0,0.5,1 --max-bl 100`)으로.

## 4. 테스트

`tests/test_lgcore_stream.py`(스트리밍 = 비스트리밍, forward_batch, sliding 스트리밍, PhaseTimer, noskip/pad 판정 불변), `tests/test_s6_s7.py`(변형 rows·dense 3 정책·스트리밍 rows 동일·S6 표·cascade·G2 (a)·E4 러너 5종 + S7 표·s6ctl/s7ctl dry-run·prep 토크나이저 경로·YaRN 설정·chunked prefill = 단일 prefill).
