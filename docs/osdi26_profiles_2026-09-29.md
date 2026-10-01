# OSDI '26 유사 논문 13편 — 평가·발표 프로파일 (2026-09-29)

계획서 v2(OSDI)의 §1 근거. 각 논문의 PDF(usenix.org/system/files/osdi26-<slug>.pdf)에서 절 구성·평가 설정·그림 전수·표·headline·발표 장치를 뽑았다. 그림 종류는 렌더링한 페이지로 확인한 것.

## A. KV 캐시·장문 서빙 (성능 전용, "lossless by construction")

### ECHO — KV offload for native sparse attention (SGLang 0.5.4; SJTU+Huawei)
- 구조(21쪽): Intro 1.3 / Background+Motivation 4(측정 그림 4 + Table 1, "Motivation N" 볼드 + 이탤릭 takeaway) / Overview 0.5 / Cache Manager 1.3 / Prefetch 2 / **Eval 4** / Discussion 0.6 / Concl 0.2 / Refs 6 / Artifact Appendix 1.
- 설정: 8×H20 96 GB 1노드, 1.5 TB DRAM; 모델 1(DeepSeek-V3.2 671B AWQ); 워크로드 InfiniteBench 318 req 80–100K + ShareGPT 100 req(짧은 입력 대조); 요청률 0.1…0.9 + Inf; baseline 2(SGLang 0.5.4, vLLM 0.11.1); 지표 throughput, effective batch, TTFT/ITL/E2E, 단계 breakdown, hit rate.
- 평가 소절: Setup → Throughput(최대·무제한·HBM 제약·task별) → Latency(+breakdown) → In-depth(hit rate, intra/inter-query prefetch).
- 그림 20(평가 9): 요청률 vs tok/s 선 + effective batch 라벨 + speedup 화살표(2–4패널), latency 선 3패널, breakdown 누적 막대, hit rate 막대, 커널 latency 누적 막대 9패널, prefetch on/off 막대. 표 1(시스템 비교 격자).
- headline: SGLang 대비 2.1×(최대 4.1× vLLM); HBM 압박 사다리(전량 → 200K → 110K pool)에서 3.1×/4.1×; ShareGPT 오버헤드 TTFT ≤ +7.9 %, E2E +16–19 %(저부하)·≤ 7.2 %(0.3 req/s 이상); offload 1.15 ms = 0.28 %/step.
- 베낄 것: 메모리 압박 사다리, 저부하 오버헤드 대조를 숫자로, 마이크로벤치 → E2E 짝 ablation("왜 E2E 이득이 작은가"), task별 음수 결과 + 히스토그램 설명, Discussion(일반화·throughput vs latency·미래 인터커넥트).

### Strata — hierarchical context caching (SGLang 0.4.5; Stanford/SJTU/NVIDIA 등)
- 구조(16쪽): Intro 1.5(Fig 1 측정) / Background 1(Fig 2 측정) / Challenges 1(Fig 3 측정) / Design+Impl 4 / **Eval 4.5** / Discussion 0.4(한계 3개 명시) / Related 0.4 / Concl 0.2.
- 설정: 테스트베드 3(8×H200, 8×H20+NVMe, GH200); 모델 3(Llama-3.1-8B, Qwen2.5-14B-1M, Llama-3.1-70B TP4) + DeepSeek-V3; 데이터 4(LooGLE 21.6K, NarrativeQA 54.8K, ReviewMT 17.7K, ShareGPT 0.7K) + Mooncake 트레이스 시뮬; Poisson, 요청률 스윕, in-flight 128; baseline 6(vLLM 0.8.5, vLLM-LMCache 0.2.1, TRT-LLM 0.17.0, TRT-LLM-HiCache, SGLang 0.4.5, SGLang-HiCache = 저자가 강화) + 자체 변형 4(-IO, -Schedule-Only, -IO-LPM, -Oracle); 지표 avg TTFT(P90), output throughput, hit rate, I/O stall %, GB/s.
- 평가 소절이 질문 제목: "How does Strata compare … on long-context workloads?", "… with a warm cache?", "… with short-context?", "How much do I/O and scheduling benefit?", "Can Strata alleviate the burden of choosing a page size?", "Can it adapt to varying cache distances?", "What causes delay hit?", "Does the decoupled layout benefit disk caching?", + GH200.
- 그림 15(평가 8): **throughput–TTFT frontier 격자 4 데이터셋 × 3 모델 = 12패널**(점선 = 비계층 baseline), ablation frontier, page size 민감도(자기 대비 정규화), **누적 기여 막대**(base +IO +DelayHit +Balance +StallHiding × 3 캐시 거리), hit rate vs resolve 지연, DeepSeek/SSD 막대, GH200 frontier + 대역폭 막대. 표 1(데이터셋 통계).
- headline: 최대 5×(vLLM-LMCache)·3.75×(TRT-LLM); "같은 TTFT에서" 3.2×/2.6×/1.9×; warm cache 2.3–2.6×; 동기: prefill 시간의 74 %가 KV 전송 대기.
- 베낄 것: 질문 제목 소절, "같은 TTFT에서" speedup 표현, warm-cache·short-context 행을 같은 격자 안에, 누적 기여 막대, oracle(무한 대역폭) 상한, 새 하드웨어 실험, 저자가 강화한 baseline, 프로덕션 배치 언급.

### DirectKV — zero-copy KV offload on GH200 (UVA, 2인 저자, 5.3k LOC CUDA)
- 구조(16쪽): Intro 1.3 / Background+Motivation 3.7(**측정 case-study 그림 6**, 각각 볼드 "Takeaway:") / Overview 0.5 / Kernel Gen 1 / Attention 2(Algorithm 1–3) / KV mgmt 0.3 / **Impl+Eval 3** / Discussion 0.7 / Related 0.5.
- 설정: GH200(+H100 PCIe 비교), 1노드; 모델 3(Llama-3.1-8B, OPT-13B/30B); ShareGPT·Alpaca, Poisson 10/20/30 req/s, 길이 1k–32k; baseline 4(SGLang, Pie, FlexGen, Neo; 버전 미기재); 지표 per-token latency, GPU/CPU 메모리, 전송량, HBM TB/s, L2 hit, 커널 ms.
- 그림 14(평가 5): 요청률 vs latency 선 3패널, **길이 vs latency 막대(OOM 표시) + 메모리 누적 막대(96 GB 천장선)**, 컴포넌트 막대 3개. 표 0.
- headline: 전송 −50 %, GPU 메모리 −43 %, E2E 최대 1.2×; 32k에서 baseline 3종 OOM; 커널 HBM throughput 3.5×.
- 베낄 것: takeaway 문장, 동기 마이크로벤치 ↔ 컴포넌트 ablation 거울 구조, OOM 표시 + 천장선, 하드웨어 카운터를 메커니즘 증거로. 약점(심사자가 볼 것): baseline 버전 없음, precision 미기재, 요청률 3점, 작은 모델.

## B. 근사 vs 정확도 (정확도 표를 시스템 지표 옆에)

### SPEX — speculative Tree-of-Thought (SGLang; PKU+ByteDance)
- 구조(16쪽): Intro 1.5 / Background 1.1 / Motivation 2(측정 4: CDF 4패널, roofline, reward 곡선, depth vs 정답률) / Design 3.4 / Impl 0.35 / **Eval 3.3** / Related 0.4.
- 설정: A6000(7B·8B)·A100(34B·30B-A3B); 모델 4; 벤치 6(GSM8K, MATH-500, AIME 24/25, BRUMO, HMMT); closed-loop 배치 1/4/16; baseline 8(ToT 알고리즘 4 × 구성 2) + CoT + MTP; 지표 throughput(questions/min), speedup, pass@1, TPOT, hit rate.
- 소절: Setup → E2E(배치·구성) → **Accuracy(6.3)** → Ablation → Compatibility with SD → Overhead → Prediction accuracy.
- 그림 15(평가 6): 이중축 throughput 막대 + speedup 마커(4패널), **누적 ablation 막대(+T1, +T1+T2, +T1+T2+T3)**, MTP 조합 막대, TPOT/reward latency 막대, hit rate 선 2개. **Table 2 정확도: 36행(모델 4 × 데이터 × CoT/ToT/SPEX) × 9열(알고리즘-구성 8 + AVG), 짝 중 나은 쪽 볼드.**
- headline: 1.2–3×(DFS 1.8–3×, BFS 1.2–1.9×), MTP와 합쳐 4.1×; TPOT 오버헤드 ≤ 15 %; 정확도 AVG +0.3…+4.1 pt(개별 셀은 하락도 있음, 원인 = 조기 종료가 탐색 공간을 바꿈).
- 베낄 것: **정확도 표를 E2E 그림 바로 뒤 소절로**, 하한 행(CoT)으로 품질 lift 보존 증명, 누적 ablation, 직교 최적화와의 조합, 오버헤드 소절, 예측기 품질 → speedup 원천 사슬.

### Kairox — GPU–CPU 뉴런 balancing (llama.cpp 5.7k LOC; Sun Yat-sen 등)
- 구조(18쪽): Intro 1.5 / Background 1.1 / Motivation 1.7(측정 3) / Overview 0.4 / Design 3.2 / Impl 0.3 / **Eval 4.4** / Related 0.5 / Artifact Appendix 1.
- 설정: RTX 3080 Ti 12 GB PC + RTX 4090 24 GB PC; 모델 13(6.7B–66B, sparse 10 + non-ReLU 3); ShareGPT 프롬프트 batch 1 + speculative; baseline 4(llama.cpp, PowerInfer 포팅, Neuralink 재구현, Q-Infer); 지표 tok/s, geomean speedup, CPU/reload latency, GPU util, TPOT trace, λ trace, 예측기 recall/precision, 정확도.
- 소절: Setup → E2E(표준·speculative) → Analysis(balancer, λ, GPU util, **VRAM budget sweep**, TPOT variability) → Ablation → Non-ReLU → Sparse analysis(**예측기 정확도 → 오버헤드 → downstream 정확도**).
- 그림 16(평가 8): 막대 + speedup 라벨 4패널, 정책 막대 + 정규화 latency 선, λ 시계열, GPU util 막대, **VRAM budget vs throughput 선("14 GB에서 남들의 24 GB를 이김")**, per-token latency trace, 누적 ablation(+LP, +NB, +ANB), **recall/precision per layer + P–R 산점도**. 표 4(표기, PCIe 대역폭, non-ReLU throughput, **정확도 5 모델 × 5 벤치 + Avg**).
- headline: llama.cpp 대비 최대 7.57×, geomean 3.15×/3.93×; 정확도 저하 "<0.5 %, 통계 분산 안"(개별 −1.3 pt); 예측기 오버헤드 1–2 % E2E.
- 베낄 것: 예측기 recall-우선 → 정확도 보존 서사(FN은 출력 크기 ≈ 0), 원본/변형 짝 행 + Avg, 막대마다 배수, 예산 스윕, 강건성 절(다른 모델군·양자화·2 하드웨어), artifact appendix의 "expected results = 상대 추세".

### ADAngel — 임의 정밀도 양자화 커널 (FasterTransformer, 15k LOC; SJTU)
- 구조(17쪽): Intro 1.6(Fig 1 측정) / Background 1.3 / Motivation 1.15(측정 2, "Observation N" 볼드 4개) / Design 3.9 / Impl 0.55 / **Eval 4** / Related 0.35.
- 설정: Jetson AGX Orin 64 GB + A100; 모델 1(Llama-3-8B, W2/3/4/5A8); 합성 길이·배치 스윕; baseline 4; 지표 TTFT, TPS, total time, peak memory breakdown, dispatch 오버헤드; **정확도 없음**(정확 재구성이라 논증 생략 — 우리는 못 따라함).
- 그림 14(평가 7): 2 지표 × 2 스윕 격자(길이 × 배치 4패널 ×2), 정밀도 교차 6패널 ×2, 플랫폼 교차 정규화(llama.cpp = 1) 2패널 ×2, ablation(고정 전략/규칙 dispatcher/축소 포트폴리오 + prefill+decode 누적 막대). 표 0.
- 베낄 것: 동기를 이질성 축 4개 × 교차 그림 + Observation, 클리핑 막대 주석, 교차 하드웨어 정규화, 세 종류 ablation(고정/단순 규칙/축소), 오프라인 프로파일링 비용 명시.

## C. 품질–시스템 결합, SLO·비용

### Murakkab — agentic workflow 오케스트레이션 (vLLM 0.9; MIT+Azure)
- 구조(21쪽 + 부록 3): Intro 1.3 / Background+Motivation 3.2(측정 3, "Insight N" 박스) / Design 4.3 / **Eval 5.3** / Related 0.6 / 부록(수학 워크플로우, 구성표, 트레이스, MILP).
- 설정: Azure 8×A100/8×H100 VM; 모델 11; Azure 24 h 실제 트레이스 + VideoMME·HumanEval·MATH·LiveCodeBench; baseline 2(LangGraph, +auto-scaling); 지표 GPU 수, MWh·Wh/req, $·$/req, 정확도·pass@1, E2E, TTFT/TPOT P90, TPS/Wh; **SLO 4 tier(best/95/80/50 pct의 정확도·지연)**.
- 소절: Setup → Single-workflow → Multi-workflow → Dynamic coding pipeline → Dynamic resource availability → DAG-aware scheduling → Optimization frequency sensitivity → Profiling generality.
- 그림 19(평가 8): **정확도 vs 지연/$/Wh Pareto 산점도(6패널, GPU/TP/모델 인코딩)**, SLO tier별 GPU·MWh·k$ 누적 막대(임계값을 x축 라벨에), 시계열 + 요청군별 정확도·지연 막대 + SLO 선·위반 ✗, pass@1 vs 토큰 Pareto, Gantt + 전력/사용률, epoch 민감도, 프로파일 vs 테스트 정확도 막대. 표 6.
- headline: LangGraph 대비 GPU 2.8× ↓, 에너지 3.7× ↓, 비용 4.3× ↓, SLO 유지; tier 완화 best→fair로 비용 ≈4×.
- 베낄 것: **$와 MWh를 거의 모든 그림에 병기**, tier 표, 정확도 손실을 tier로 정당화(66.2→64.4 %가 에너지 −23.5 %), 미지 벤치 일반화.

### Local MoE CPU–GPU hybrid (2×EPYC + 2×RTX 5090; Tsinghua)
- 구조(18쪽): Intro 1.3 / Background+Motivation 3.2 / Design 5 / **Eval 3.4** / Related 0.7 / **Discussion 0.5** / Concl.
- 설정: 1노드; 모델 5(DeepSeek-R1 671B FP8/Q4, Kimi-K2 1T, DS-V3.1, K2-Instruct); prefill 256–64K, 컨텍스트 1K–128K, 배치 1–6 스윕, 동시성 마이크로벤치; baseline 5(KTransformers, llama.cpp, ik_llama.cpp, BLAS 2); 지표 prefill tok/s, TTFT, decode tok/s, TPOT, kernel µs·GB/s, Exact Match; **SLO 임계선(30 s TTFT, 20 tok/s)**.
- 그림 18(평가 6; 설계 다이어그램 12): prefill 막대, TTFT vs 길이(log-log) + 30 s 선, decode vs 컨텍스트 + 20 tok/s 선, 정규화 배치 스케일링 막대(절대값 주석), 양자화 비교 선, 동시성 정규화 막대 3패널. 표 2(커널, **정확도 vs 공식 점수**).
- 베낄 것: SLO 임계선을 길이 축 그림에 직접, "추정" baseline 막대의 정직한 표시, 공식 점수 대비 품질표, Discussion에서 저사양 하드웨어로 외삽.

### Sereno — 모바일 speculative decoding (PowerServe 6.4k LOC; SJTU IPADS)
- 구조(20쪽): Intro 1.4(**Fig 1 = Pareto 산점도 + 이득 영역**) / Background 0.5 / Interference analysis 2(측정 3 + 코드 시그니처 표) / Challenges 0.9 / Design 2.6 / **Impl 0.4** / **Eval 5** / Related+Discussion 0.8.
- 설정: OnePlus 13·12; 모델 2(8B target + 1B draft); 앱 30개 4 tier, GSM8K 프롬프트, 버스트 duty-cycle; baseline 8 + native; 지표 jank rate, 벤치 점수, tok/s, E2E, tokens/J·W, 오버헤드.
- **평가 질문 4개 명시**: "❶ End-to-End Effectiveness … ❷ Mechanism Analysis … ❸ Ablation Study … ❹ System Analysis: overheads and robustness".
- 그림 14(평가 10): 모든 평가 막대에 시스템 지표를 빨간 선(이중축)으로 결합, 100 % 누적 시간 막대, 산점도 + 회귀(R = 0.86), **누적 ablation(+Abort → +Batch → +Sleep → +N-gram)**, tokens/J·W, SoC 세대 비교.
- 베낄 것: 인트로 Pareto, QoS↔throughput 이중축, 4 질문 프레임, 강건성(다른 SoC·데이터셋·duty cycle), 오버헤드 문단, Discussion 범위.

## D. 서빙 스케줄링·자원 (SLO·goodput·트레이스 관례)

### LMetric — 요청 스케줄링 (Rust 라우터; SJTU+Alibaba)
- 구조(14+3+2쪽): Intro 1.7 / Background 1.3 / **Characterization 4.3(측정 그림 9)** / Design 3 / **Eval 3.3** / Discussion 0.45 / Related 0.3.
- 설정: 16×H20 + 프로덕션 canary; 모델 2; **실제 트레이스 4(내용 해시 + 타임스탬프; Azure/BurstGPT는 내용 없어 거부)**, 요청률 = 최대의 절반 기본 + 스윕; baseline 6(전부 재구현·튜닝); 지표 TTFT/TPOT mean·P50/90/95/99, CDF, hit ratio, 불균형.
- 그림 34(평가 8): CDF + 0.99 tail 인셋 8패널, **요청률 vs TTFT/TPOT mean·P99 16패널**, hit ratio 시계열, batch-size 히트맵, 프로덕션 대시보드. 표 0.
- 베낄 것: baseline 하이퍼파라미터 민감도 격자로 설계를 정당화, mean+P99 동시, 메커니즘 시계열, 실패 조건 도출 + 탐지기, canary.

### Prism — multi-LLM GPU memory ballooning (SGLang; UCLA/Berkeley 등, 10.4k LOC)
- 구조(14+3+2쪽): Intro 2 / Background 0.5 / Trace analysis 2(측정 2 + Table 1) / Overview 0.5 / Design 4 / **Eval 4** / Related 0.4; 부록 민감도.
- 설정: 4×8 H100; 모델 58; 프로덕션 트레이스 4 재생(rate scale); baseline 4(Static, MuxServe++ 포팅 + 검증표, QLM, ServerlessLLM); **지표: per-model SLO = 전용 GPU P95 × SLO scale, attainment, 99 % 도달 GPU 수(= 비용), throughput, activation latency, 오버헤드, GPU당 매출**.
- 그림 15(평가 7): **attainment vs rate / SLO scale / GPU 수 3×4 격자 + 99 % 마커**, 메모리 공유 시계열, placement·arbitration 효과, 대규모(16–32 GPU) 선 + "32+" 막대, activation 누적 막대, 프로덕션 시계열. 표 3(트레이스, 포팅 검증, 모델 크기).
- 베낄 것: attainment 격자, **99 % attainment에 필요한 GPU 수를 비용으로**, 포팅 baseline 검증표, 단일 A100-40G 최악 오버헤드 실험, 민감도 부록.

### Revisiting Pipeline Parallelism (Korea Univ., SGLang 0.4.1, **4×A100-40 PCIe 1노드**)
- 구조(14+3쪽): Intro 1.5 / Background 1.5(Fig 1 측정) / Challenges 1.5(측정 3) / Design 3 / **Eval 5.5** / Related 0.6.
- 설정: 모델 2(Qwen2.5-32B/14B); Azure Conversation·ShareGPT·CNN-DailyMail + 합성 2; **Poisson 스윕 + Azure 트레이스 15분 재생(0.9× scale)**; baseline TP + PP_static(오프라인 튜닝 최선) + 자체 3 변형; **goodput = P90 TTFT ≤ 2 s & P90 TPOT ≤ 200 ms(MLPerf)를 만족하는 최대 요청률**, goodput vs SLO scale(1.0→0.2), 반복 latency CDF, mean/P90/P95/P99 막대, attainment + 조건부 CDF, 예측기 MAPE.
- 그림 14(평가 7): goodput 막대 + P90 지연 vs 요청률 선 16패널, CDF 3, goodput vs SLO scale 4패널, percentile 막대 6패널(끊긴 축), 트레이스 재생 attainment + CDF. 표 1(예측기).
- headline: goodput TP 3.48 → PP 8.28 → DCP 8.47 req/s; SLO가 죄어질수록 이득 2.7×→5.4×; 트레이스 attainment TP 1.9 % / PP 92.5 % / DCP 94.6 %.
- **4-GPU 논문이 설득하는 법**: 정확한 동기 마이크로벤치, 실제 4 + 합성 2 워크로드 × 2 모델, 16패널 스윕, SLO-scale 강건성, latency CDF, 트레이스 재생 attainment, percentile 막대, ablation 표 — 규모 대신 조건의 폭.

### Seer — 동기 RL rollout (vLLM+Mooncake; Moonshot+Tsinghua, 256×H800)
- 구조: Intro 2.3 / Background+Motivation 2(측정 3) / Design 3.2 / **Eval 5** / Related 1; 그림 15(평가 9), 표 6(단계 시간, acceptance, 설정, **누적 breakdown 표**, API 2).
- 베낄 것: 누적 ablation 표(워크로드별 배수), **품질 보존 = reward 곡선을 baseline과 겹쳐 그림**, "tail time" 정의, 막대 speedup 라벨.

## E. 집계(13편)

| 지표 | 중앙값 | 범위 |
| --- | --- | --- |
| 본문 쪽 | 13.5 | 12.5–14.5 (+ 참고문헌 3–6) |
| 그림 총 | 15 | 14–34 |
| 평가 절 그림 | 8 | 5–10 |
| 측정 동기 그림 | 3 | 1–9 |
| 표 | 1 | 0–6 (정확도 보고 논문 2–4) |
| 평가 절 쪽 | 4 | 3–5.5 (본문의 25–35 %) |
| baseline | 4 | 2–8 |
| 모델 | 3 | 1–58 |
| 워크로드 | 실제 2–4 + 짧은 입력 대조 | — |
| 정확도 명시 | 4/13 (표 3, Pareto 1) + 곡선 1 | 나머지 "lossless" 1문장 |
| SLO/goodput | 3/13 | Prism·PP·Murakkab |
| $·에너지 | 3/13 | Murakkab·Sereno·Prism |
| Discussion 절 | 6/13 | — |
| Artifact | 5/13 | 배지·appendix |
