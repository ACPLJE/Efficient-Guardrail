# 비대칭 가드 실험 계획서 — OSDI '27

Oct 10, 2026 · @Jieui Kang

값의 출처 표기: \[측정\] 우리가 잰 값, \[계산\] 산술로 얻은 값, \[문헌\] 논문·문서·소스 코드에서 확인한 값, \[추정\] 근거가 약한 어림값, \[해석\] 사실에서 끌어낸 판단. 통과 기준은 모두 제안값이다.

## 0. 한눈에 보기

이 계획서는 OSDI '27(초록 Dec 1, 2026, 본문 Dec 8, 2026)에 낼 새 논문의 실험, 구현, 일정을 정한다.

**논문의 한 문장.** 유해 콘텐츠와 prompt injection은 판정에 필요한 깊이, 증거, 마감이 다르다. 그래서 유해 콘텐츠는 타깃의 prefill과 나란히 모든 창을 얕게 훑고, injection은 타깃 자신의 attention으로 출력에서 거꾸로 추적한 몇 개 창만 내보내기 전에 판정하는 서빙 런타임을 만든다.

**전제.**

- 목표 학회는 OSDI '27 연구 트랙이고 본문은 12쪽이다.
- 대상은 10월 9–10일에 정한 새 방향이다. Landmark Guard v4와는 별개의 계획으로 잡고, Landmark Guard는 유해 콘텐츠 경로로 재사용한다. OSDI '27에 두 논문을 따로 낼지는 미정이고, 이 계획은 하나로 내는 쪽을 권한다(12절).
- 인력은 본인 1명과 collaborator, 장비는 RTX 6000 Ada 48 GB 클러스터다.
- 새 방향의 실험 수치는 아직 없다. 지금 있는 것은 유해 콘텐츠 경로의 측정값과 문헌 수치다.

**입증할 주장.**

1. 같은 detector를 쓰더라도 출력이 실제로 기댄 창만 판정하면 긴 입력에서 오탐이 준다.
2. 유해 콘텐츠 판정은 첫 토큰 전에 끝나고 injection 판정은 chunk를 내보내기 전에 끝나서, 생성 기준 TTFT는 늘지 않고 보이는 지연은 chunk 하나의 생성 시간과 판정 시간으로 묶인다.
3. 역추적은 서빙 엔진 안에서 타깃의 attention을 누산해 얻고, 귀속용 모델을 따로 돌리지 않는다.

**가장 큰 위험.**

- 타깃의 attention 역추적이 긴 입력에서 주입 창을 맞히는지 아직 재지 않았다. 이 값이 낮으면 논문이 성립하지 않는다.
- 본문 마감까지 8.5주다. 엔진 통합과 평가를 모두 새로 해야 한다.

**먼저 할 일.** 10월 12–18일에 확인 4개(E0)를 하고 10월 18일에 진행 여부를 정한다(G0, 10절). 기준은 두 가지다. 32k 입력에서 성공한 공격의 주입 창이 상위 8개 창에 드는 비율이 0.90 이상일 것, 같은 recall(0.80)에서 문서 FPR이 모든 창 판정의 5분의 1 이하일 것(제안값). 한쪽만 미달이면 10.2절의 규칙대로 다시 재고, 둘 다 미달이면 Landmark Guard v4 계획으로 돌아간다.

**규모.** 실험은 33개(E0–E6)다. GPU 시간은 재실행을 포함해 약 2,350 GPU-h로 가용 5,472 GPU-h의 43 %다 \[추정\]. 본문 그림 16개와 표 3개를 목표로 한다.

## 1. OSDI '26 논문 10편에서 뽑은 요구 사항

읽은 10편은 우리 논문과 닿는 면이 서로 다르게 골랐다. 검증 구조 2편(AEGIS, Semisolates/try), LLM 품질을 평가하는 2편(SMARTTalk, SPEX), LLM serving 5편(Sereno, Murakkab, Prism, Revisiting PP, LMetric), attention 커널을 고친 1편(DirectKV)이다. OSDI '26 프로그램 136편 중 guardrail, prompt injection, jailbreak를 다룬 논문은 없다 \[문헌: 프로그램 페이지\]. 이 주제의 평가 관례가 OSDI에 없으므로, 아래 요구 사항은 serving 논문의 관례에 보안 논문의 의무를 더한 것이다.

### 1.1 10편 요약

| 논문 | 한 일 | 그림/표, 평가 분량 | 평가에서 눈에 띄는 점 | 우리가 가져올 것 |
| --- | --- | --- | --- | --- |
| AEGIS | LLM 학습 중 silent data corruption을 sense-then-verify로 탐지 | 17/2, 3.0쪽 | 3.5×10⁷ GPU-h 운영에서 18건 탐지, 오버헤드 0.86 %. Table 2가 고장 장비 8대 × detector 5종. 오버헤드를 sampling rate에 대해 그림 | 공격 종류 × 단계 표, FP와 FN 분리, 단조 knob 하나에 대한 오버헤드 곡선, 검증 시간이 어디서 나오는지 |
| Semisolates/try (Best Paper) | 불투명한 컴포넌트의 효과를 붙잡아 검사한 뒤 commit | 3/2, 3.6쪽 | 범위를 한 문장으로 한정. 사례 5종이 평가 소절 5개로 돌아오고 소절마다 Control, Equivalence, Performance. 최악 82.6×를 그대로 적음 | 계약에 이름 붙이기, 범위 한 문장, 시나리오와 평가 소절의 1:1 대응, 지는 경우 공개 |
| SMARTTalk | SSD 고장 예측에 LLM 적용 | 10/7, 4.6쪽 | 품질만 평가. Table 5가 방법 22종 × 2 × 지표 5종이고 FPR 0.0019–0.0057을 headline 옆에 표기. 같은 backbone의 naive baseline 2종. seed와 CI 없음 | headline마다 FPR·FNR 병기, 운영점의 근거, 같은 backbone naive baseline |
| SPEX | Tree-of-Thought 추론을 speculative하게 가속 (SGLang) | 15/2, 3.2쪽 | challenge, insight, technique이 3쌍으로 대응. Table 2가 모델×데이터 12블록 × 3행(하한, 원본, 제안). 누적 ablation, speculative decoding과의 호환, 선택 적중률 소절 | 3행 품질표, 선택 품질 소절, 누적 ablation, 다른 최적화와의 호환 |
| Sereno | 모바일 LLM 추론의 memory bandwidth 간섭 제어 | 14/1, 4.9쪽 | 동기가 정량화, 소거로 귀속, 원인, naive 해법 실패(−8.4 %)의 사슬. Fig. 1이 Pareto와 gain 영역. 평가 질문 4개에 번호. naive 전략 3종을 자기 엔진에 재구현. "lossless by design"이라 정확도 표 없음. 약 6.4k LoC | 동기의 인과 사슬, Fig. 1 Pareto, naive 전략 재구현, lossless는 release gate에만 주장 |
| Murakkab | agentic workflow의 자원 최적화 (vLLM) | 19/6, 약 5쪽 | 상자에 넣은 Insight 3개. 품질·지연 SLO tier 4단. Azure 24시간 trace. Table 2에서 GPU 2,568 → 912. 요청 하나의 Gantt. 보지 않은 benchmark로 일반성 확인 | 단일 요청 timeline, class별 마감선과 위반 표시, 보지 않은 데이터로 일반성 |
| Prism | 여러 LLM의 GPU 메모리 공유 (SGLang) | 15/3, 약 4쪽 | SLO = 전용 GPU의 P95 TTFT·TPOT × 배수. Fig. 5가 3×4 격자(부하, SLO 배수, GPU 수). 99 % attainment에 필요한 GPU 수. baseline 이식 검증표. 최악 오버헤드 3 ms(4 %)와 4 ms(13 %). 약 10.4k Python + 774 C++ | SLO 정의, 격자 그림, 필요한 GPU 수, 이식 검증표, 이득이 없는 조건의 오버헤드 |
| Revisiting PP (DCP) | pipeline parallelism의 prefill chunk를 동적으로 조절 (SGLang) | 14/1, 약 5.5쪽 | A100-40GB 4장 한 노드. goodput = P90 TTFT와 TPOT가 SLO(2,000 ms, 200 ms)를 지키는 최대 요청률. Fig. 1에 SLO 선과 "limited by" 표시. SLO 배수 1.0–0.2 sweep. trace 재생 attainment 1.9, 92.5, 94.5 %. 정적 설정 전부를 같이 그림 | goodput 정의, 4-GPU 한 노드로 설득하는 폭, 정적 설정 전부와 offline 최선의 비교 |
| LMetric | KV cache를 고려한 요청 라우팅 | 34/0, 약 3쪽 (특성 분석 4.6쪽) | 특성 분석이 본문의 3분의 1이고 측정 그림 9개. 대안을 "Cons #"로 하나씩 기각. trace를 측정한 최대 부하의 절반에서 시작해 sweep. CDF에 0.99 위 확대. 실패 조건을 식으로 유도하고 fallback을 둠. baseline 재구현이 원본보다 느리지 않다고 명시 | 모든 guard 배치를 한 틀에서 측정해 기각, tail 보고, 운영점의 근거, 실패 조건과 fallback |
| DirectKV | KV cache zero-copy offload, attention 커널 수정 | 14/0, 2.7쪽 (동기 3.4쪽) | FlashAttention-3 위 약 5.3k LoC. 커널 변경을 loop 순서와 상태로 서술. Nsight Compute counter와 지연을 짝지어 보고. 등가성 실험이 없고 저부하 오버헤드를 재지 않음 | 커널 diff를 추가 상태와 자원 한계로 서술, counter와 지연의 짝, 등가성 실험과 저부하 오버헤드는 우리가 채움 |

이 10편의 그림 수 중앙값은 14.5(범위 3–34), 표는 2(0–7), 평가 분량은 3.8쪽이다 \[계산\]. 9월 29일에 정리한 13편 프로필의 중앙값은 본문 13.5쪽, 그림 15, 평가 그림 8, 측정된 동기 그림 3, 표 1, 평가 4쪽, baseline 4, 모델 3이었다. 모두 camera-ready 기준이고 제출 한도는 12쪽이다.

### 1.2 요구 사항

| 번호 | 요구 사항 | 근거 | 우리 논문에서 채우는 곳 |
| --- | --- | --- | --- |
| R1 | 동기는 우리 장비에서 잰 그림 3개 이상으로, 정량화 → 귀속 → 원인 → naive 해법 실패 순서로 쓴다 | Sereno, LMetric, DirectKV, AEGIS | 2절 관찰 1–3, 실험 E1.1–E1.5, Fig. 3–5 |
| R2 | challenge, insight, technique을 1:1로 맞추고 각 쌍에 측정 그림을 하나씩 둔다 | SPEX, Murakkab | 비대칭 3개 → 설계 3개. SPEX의 Insight 2처럼 그림 없는 쌍을 두지 않는다 |
| R3 | 계약과 범위를 각각 한 문장으로 앞에서 밝힌다 | Semisolates/try | 3절. 계약은 "chunk는 그 chunk에 걸린 판정이 모두 끝나기 전에 sink에 닿지 않는다" |
| R4 | baseline을 세 층으로 둔다: 상·하한, 기존 시스템, 우리 엔진에 재구현한 naive 전략. 이식은 원 수치로 검증한다 | Sereno, SMARTTalk, Prism, LMetric | 7절, Table 4 |
| R5 | SLO와 goodput을 문장으로 정의하고 SLO 배수를 sweep한다 | DCP, Prism | E4.3. SLO = guard 없는 타깃의 전용 GPU P95 × 배수 |
| R6 | production trace를 재생하고 운영점의 근거를 밝힌다 | LMetric, Prism, DCP, Murakkab | E4.4. Mooncake trace, 측정한 최대 부하의 절반에서 sweep |
| R7 | 품질표는 하한, 상한, 제안의 3행 블록으로 쓰고 FPR·FNR을 headline 옆에 둔다. 단계 × 공격 표를 따로 둔다 | SPEX, SMARTTalk, AEGIS | Table 2, Table 3. 표본 수와 95 % CI를 붙인다 |
| R8 | 선택 품질을 독립된 소절로 낸다 | SPEX §6.7, Sereno Fig. 11 | E3.3, Fig. 9. hit@k를 k, 길이, 위치, 공격 종류별로 |
| R9 | 정상 트래픽에서 항상 드는 오버헤드를 단조 knob에 대해 ms와 %로 낸다. 이득이 없는 조건도 낸다 | AEGIS, Prism, SPEX, StriaTrace | E2.3, E4.7, Fig. 14 |
| R10 | 커널 변경은 추가 상태와 자원 한계로 서술하고 LoC를 적는다. 출력 등가성은 실험으로 보인다 | DirectKV, Semisolates/try, Sereno | 4절, E2.1–E2.2 |
| R11 | 실패 조건을 유도하고 fallback과 마감을 놓쳤을 때의 정책을 정한다. 지는 경우를 숫자로 적는다 | LMetric, Sereno, Semisolates/try | E5, E4.7, 11절 |
| R12 | 누적 ablation, 정적 설정 전부의 sweep, 보지 않은 데이터, 다른 최적화와의 호환을 낸다 | SPEX, DCP, Murakkab | E4.5, E6.1–E6.4 |

### 1.3 10편에 없어서 따로 져야 하는 것

- **공격자.** 10편 중 공격자를 상대하는 논문은 없다. Semisolates/try는 범위에서 뺐다. guardrail 논문은 그럴 수 없으므로 적응 공격 절(E5)이 추가 의무다.
- **통계적 불확실성.** SPEX, SMARTTalk, DCP는 seed나 CI 없이 통과했다. 우리는 ASR과 FPR의 몇 pp 차이를 주장하므로 표본 수와 bootstrap 95 % CI를 모든 품질 수치에 붙인다.
- **CFP의 ML 논문 조건.** 기존 systems 연구와의 관계, 알려진 기법 대비 baseline, 새로 생기는 systems 문제를 써야 한다. 연결할 선행은 dynamic taint tracking(TaintDroid), 검증 후 commit(Semisolates/try, hS), sense-then-verify(AEGIS), 엔진 내 tracing(StriaTrace), LLM serving 스케줄링(PASCAL, SPORK, Sherlock)이다.

### 1.4 분량 목표

| 절 | 쪽 | 내용 |
| --- | --- | --- |
| 1 Introduction | 1.5 | 문제, 관찰 3개 요약, 기여, Fig. 1 |
| 2 Background and Motivation | 2.25 | 두 위협, 현재 배치, 관찰 1–3과 측정 그림 3개, naive 해법의 실패 |
| 3 Overview | 0.75 | 계약, 위협 모델, 구성도 |
| 4 Design | 3.0 | 설계 1–3, 커널 내 누산, release gate, 스케줄링 |
| 5 Implementation | 0.4 | 엔진, LoC, 등가성 |
| 6 Evaluation | 3.6 | Q1–Q6 |
| 7 Discussion, 8 Related Work | 0.5 | 한계, 선행 연구 |

합계 12.0쪽이다 \[계산\]. 그림은 본문 16개, 표는 3개를 목표로 한다.

## 2. 논문의 논리

### 2.1 문제 설정

긴 입력(문서, 검색 결과, 도구 출력)을 받는 LLM 서비스는 두 가지를 막아야 한다. 하나는 입력과 출력에 섞인 유해 콘텐츠이고, 다른 하나는 신뢰할 수 없는 문맥에 심긴 지시(prompt injection)다. 지금의 배치는 둘을 같은 파이프라인으로 다룬다. 입력을 창으로 잘라 모든 창을 생성 전에 판정하거나, 생성 뒤에 비동기로 판정한다. 논문은 이 배치가 긴 입력에서 세 군데로 어긋난다는 것을 재서 보이고, 어긋난 곳마다 설계를 하나씩 둔다.

### 2.2 관찰 3개와 설계 3개

| 비대칭 | 재서 보일 문장 | 지금 있는 근거 | 설계 | 기각할 대안 |
| --- | --- | --- | --- | --- |
| 깊이 | 같은 backbone에서 유해 콘텐츠는 얕은 층에서 갈린다. injection은 지시문이 많은 정상 텍스트와 섞이면 깊은 층에서야 갈린다 | \[문헌\] Zhou et al. Fig. 3: 유해와 정상이 앞쪽 5개 층 안에서 95 % 이상 분리. PIShield Fig. 4: Qwen3-4B의 FPR이 18/36층까지 약 0.35, 24–27층에서 0.05 이하. TaskTracker Table IX: AUROC 0.577(7층) → 0.989(15층). \[측정\] LG: 9/28층 probe의 hit@32 0.954. 반례 \[문헌\] LEC: 두 과업 모두 50–75 % 깊이가 최선 | 설계 1. 유해 콘텐츠는 모든 창을 얕게 읽고 상위 k개만 깊게 판정한다 (LG 그대로) | 모든 창을 끝 층까지 읽으면 연산이 1.00 대 0.34다. injection까지 모든 창을 18층으로 올리면 0.65다 \[계산\] |
| 증거 | injection 여부는 창만 봐서는 정해지지 않고 창, 과업, 출력의 관계로 정해진다. 창만 보고 모든 창을 판정하면 문서 단위 오탐이 창 수만큼 누적된다 | \[계산\] 문서 FPR 5 %를 지키려면 창 단위 FPR이 8k에서 8.1×10⁻⁴, 32k에서 2.0×10⁻⁴, 128k에서 5.0×10⁻⁵ 이하여야 한다. \[문헌\] LongPIBench: 300토큰 조각을 전부 판정하면 Prompt Guard의 FPR이 0.01 → 0.98. Attention Tracker Fig. 2: 공격 시 attention이 주입된 지시문으로 옮겨 간다. AgentWatcher: 귀속 후 판정으로 긴 문맥 ASR 0.00–0.06 | 설계 2. 출력 chunk가 기댄 창을 타깃 자신의 attention으로 역추적하고, 그 창만 창 + 과업 + chunk로 판정한다 | 모든 창 판정은 오탐이 누적된다. 두 번째 모델로 귀속하면 AgentWatcher 8.2 s/탐지, AttnTrace 10–20 s가 든다 \[문헌\] |
| 마감 | 유해 콘텐츠 판정은 첫 토큰 전에 필요하다. injection 판정은 출력이 sink(사용자 화면, 도구 호출)에 닿기 전에만 끝나면 된다. 두 마감 앞에 각각 여유가 있다 | \[측정\] 128k에서 LG scan 1.10 s, 8B dense forward 50.3 s. scan은 그 2.2 %다 \[계산\]. \[문헌\] SentGuard: 문장 단위 hold-and-release의 추가 지연이 비동기 36 ms, blocking 576 ms. Bedrock의 비동기 모드는 검사 전 텍스트가 노출된다 | 설계 3. 유해 경로는 prefill과 병렬로, injection 경로는 chunk마다 hold-then-release로 돌린다. guard 작업에 마감 종류를 붙여 타깃과 같은 GPU에서 스케줄한다 | 전부 생성 전에 하면 injection은 판정할 출력이 없다. 전부 생성 후 비동기로 하면 노출된다. 전부 blocking으로 하면 TTFT가 는다 |

깊이 비대칭은 셋 중 근거가 가장 약하다. 문헌 수치는 backbone과 데이터가 서로 다르고 LEC는 반대 방향이다. 이 관찰이 우리 측정에서 안 나와도 증거와 마감의 비대칭만으로 설계 2와 3은 선다. 그 경우 설계 1은 유해 경로의 비용 절감으로만 쓴다.

### 2.3 주장과 필요한 값

통과 기준은 제안값이고 E0의 측정이 나오면 다시 정한다.

| 주장 | 필요한 값 | 실험 | 그림·표 | 통과 기준 (제안) |
| --- | --- | --- | --- | --- |
| C1. 역추적된 창만 판정하면 같은 detector로 오탐이 준다 | 성공한 공격의 주입 창이 상위 k개에 드는 비율(hit@k). 같은 ASR에서 문서 FPR, 모든 창 대 역추적 창. 8k, 32k, 128k | E0.1, E0.2, E3.1, E3.3, E3.4 | Table 2, Fig. 9, Fig. 10 | hit@8 ≥ 0.90(32k), ≥ 0.85(128k). 같은 recall에서 FPR 비 ≥ 5배(32k, 128k). 원천을 합친 ASR 차의 95 % CI 상한 ≤ 3 pp |
| C2. 유해 판정은 첫 토큰 전에, injection 판정은 release 전에 끝나고 지연은 chunk 하나로 묶인다 | 생성 기준 TTFT, 보이는 첫 chunk 시각, chunk별 release lag의 P50·P95·P99, SLO 배수별 goodput | E0.4, E4.1–E4.4 | Fig. 5, Fig. 11–13 | 생성 기준 TTFT 증가 ≤ 3 %(128k). 입력 8k 이상 workload에서 SLO 배수 1.25일 때 goodput이 guard 없는 타깃의 0.9배 이상 |
| C3. 역추적은 엔진 안에서 얻고 귀속용 모델을 따로 돌리지 않는다 | 누산을 켰을 때 TPOT 증가, 요청당 추가 메모리, 출력 토큰 동일성, 요청당 판정한 창 수 | E2.1–E2.3 | Fig. 14 | TPOT +2 % 이하(기본 설정), +5 % 이하(최악). 같은 분할에서 누산을 켜고 꺼도 토큰 100 % 동일 |
| C4. 유해 경로의 품질과 비용은 injection 경로와 같이 돌려도 유지된다 | LG의 R@128k와 scan 시간, 타깃 prefill에 준 간섭 | E3.5, E4.6 | Fig. 5, 본문 수치 | prefill 지연 ≤ 3 % |
| C5. attention을 피하는 공격은 공격 성공률을 스스로 깎는다 | 무방어 ASR 대 주입 창의 attention 순위 곡선, 방어 시 ASR, 최악 release lag | E5.1–E5.3 | Fig. 15 | 통과 기준 없음. 곡선과 실패 조건을 그대로 보고 |

0절의 주장 3개는 C1–C3이다. 기본 설정은 128토큰 분할, 추적 층 8개, head 균등 평균, k = 8, chunk 32토큰(첫 chunk는 8토큰)을 뜻하고, C1과 C3은 같은 설정에서 잰다. C4는 기존 측정을 유지한다는 확인이고, C5는 성립하지 않으면 한계로 적는다.

### 2.4 평가 질문

1. Q1. 타깃의 attention 역추적은 주입 창을 찾는가? 두 번째 모델을 쓰는 귀속과 비교해 어떤가?
2. Q2. 판정 창을 줄이면 오탐이 얼마나 주는가? ASR과 정상 과업 성능은 유지되는가?
3. Q3. 사용자와 도구가 겪는 지연, SLO 아래 goodput은 얼마인가?
4. Q4. 엔진 안 누산과 release gate의 비용은 얼마인가? 출력은 그대로인가?
5. Q5. 구성 요소 각각의 기여와 k, chunk 크기, 분할 크기의 민감도는 어떤가?
6. Q6. 적응 공격과 이득이 없는 조건에서 어떻게 되는가?

### 2.5 가장 가까운 선행과 남는 차이

서빙 엔진 안에서 guard 작업을 마감이 있는 job으로 돌리는 systems 논문은 찾지 못했다(OSDI '26 전체, SOSP '26 채택 목록, MLSys '26의 LLM serving 세션 범위). 구성 요소마다 가까운 선행은 있고 injection 경로가 가장 붐빈다.

| 선행 | 한 것 \[문헌\] | 남는 차이 |
| --- | --- | --- |
| SingProbe | 타깃 3개 층의 probe를 SGLang·vLLM fork에 넣음. token 간 지연 +0.18–0.45 %, TTFT +0.86–2.25 % | 유해 콘텐츠만 다룸. 창, 긴 입력, 마감 없음 |
| CausalArmor (ICML 2026) | 권한 있는 tool call에서 proxy 모델로 leave-one-out 귀속 후 sanitize. 지연 1.22–1.30배 | 두 번째 모델의 추가 forward. tool 결과 단위. 엔진 설계 없음 |
| MindGuard | 타깃 자신의 attention으로 tool 결정의 출처 그래프를 만듦. TPR/FPR 96.5/5.2 | tool metadata poisoning 한정. 엔진 구현과 오버헤드 수치 없음 |
| Rennervate (NDSS 2026) | 응답 첫 32토큰의 attention으로 토큰 단위 탐지. FPR 0.46–2.42 % | attention tensor 전체를 내보내야 함. 긴 문맥과 지연 수치 없음 |
| BASIS | prefill attention probe로 실제로 뚫릴지를 예측. 과잉 거부 18.4 % | 생성 전 예측. 엔진 통합과 긴 문맥 시험 없음 |
| AttriGuard | tool call마다 shadow replay. 11.5 → 32.4 s (2.8배) | 재실행 비용 |
| AgentWatcher, AttnTrace | 귀속 LLM + monitor LLM. 8.2 s/탐지, 10–20 s/traceback | 두 번째 모델 |
| PISanitizer | attention으로 지시를 유발하는 토큰을 찾아 지움. 약 1.8 s | 추가 pass. release gate 없음 |
| SentGuard | 문장 단위 hold-and-release. 36 ms 대 576 ms | 유해 콘텐츠만. 입력 scan과 GPU 공동 스케줄 없음 |
| vLLM Hook v0 | vLLM worker에서 Q/K를 저장해 attention을 재구성 | 수치 없음. 커널 안 창 단위 누산이 아님 |

남는 것은 네 가지다. (1) split-KV decode 커널이 이미 계산하는 분할별 부분 상태로 창 단위 attention 질량을 얻는 것과 그 오버헤드 수치. (2) 사용자 출력과 도구 호출에 공통인 chunk 단위 release를 엔진의 attention에 묶는 것. (3) 위협 종류별 마감을 가진 guard job을 타깃과 같은 GPU에서 스케줄하는 것. (4) 같은 detector로 잰 길이별 FPR 곡선.

기여 문장은 셋으로 쓴다. 측정(세 비대칭을 같은 backbone과 같은 엔진에서 정량화), 메커니즘(엔진 내 attention ledger, release gate, 마감 종류별 스케줄), 평가(8k–128k, 타깃 3종, detector 2종 이상에서 ASR, FPR, 지연, goodput, 오버헤드, 적응 공격).

### 2.6 예상 리뷰 질문

| 질문 | 답의 방향 | 답하는 실험 |
| --- | --- | --- |
| detector를 모든 창에 병렬로 돌리면 지연은 같지 않은가? | 작은 분류기라면 지연은 비슷할 수 있다. 차이는 오탐 누적과 LLM judge의 비용이다. 128k에서 창 1,023개를 4B judge로 판정하면 8B 타깃 prefill의 약 2.0배 연산이고 8개면 1.6 %다 \[계산: 창당 입력 512토큰 가정, 파라미터 × 토큰 비례\] | E1.2, E1.5, E3.4 |
| attention은 귀속이 아니지 않은가? | 설명에 쓰지 않고 후보 선택에 쓴다. 기준은 hit@k이고 leave-one-out과 비교한다 | E3.3 |
| 공격자가 attention을 피하면? | 피할수록 무방어 ASR이 얼마나 떨어지는지 곡선으로 낸다. 떨어지지 않으면 한계로 적고 fallback(얕은 probe 순위와의 합집합)을 잰다 | E5.1 |
| 무엇을 보장하는가? | 탐지 정확도는 보장하지 않는다. 보장하는 것은 순서다. 판정 전에는 어떤 chunk도 sink에 닿지 않는다 | E2.4 |
| 요약처럼 출력이 문서 전체에 기대면? | 판정 창 수가 늘어 이득이 준다. 과업 종류별 판정 창 수 분포와 최악을 그대로 낸다 | E4.7 |
| 지시문 형태가 아닌 injection은? | 역추적은 형태와 무관하게 창을 고르지만 판정기가 못 가를 수 있다. 재서 한계로 적는다 | E3.6 |
| 짧은 입력에서는? | 이득이 없고 hold 때문에 보이는 첫 chunk가 늦어진다. 길이 임계값 아래에서는 hold 없이 생성 전에 모든 창을 판정하는 쪽으로 바꾸고(4.5절), 수치로 적는다 | E4.7 |
| 왜 systems 논문인가? | 기여가 커널의 부분 상태 재사용, release gate, 마감 종류별 스케줄, SLO 아래 goodput에 있다 | E2, E4 |

## 3. 위협 모델과 범위

### 3.1 배치와 신뢰 경계

- 서빙 제공자가 엔진, guard, judge를 운영한다.
- 요청은 신뢰하는 구간(system prompt, 사용자 과업)과 신뢰하지 않는 구간(검색 문서, 도구 출력, 업로드 파일)으로 이루어진다. 애플리케이션이 구간 경계를 엔진에 알려 준다고 가정한다. 역추적의 순위는 신뢰하지 않는 구간의 창끼리만 매긴다.
- sink는 셋이다. 사용자에게 스트리밍되는 텍스트, 도구 호출과 그 인자, 메모리나 파일에 쓰는 도구 호출.

### 3.2 공격자

- **능력.** 신뢰하지 않는 구간 하나 이상의 내용을 임의로 쓴다. 위치, 길이, 개수는 자유다. 시스템 설계, detector, k, chunk 크기, 창 경계를 안다. 적응 공격에서는 타깃과 guard의 가중치도 안다. 엔진, system prompt, 사용자 과업은 바꾸지 못한다.
- **목표.** (H) 유해 콘텐츠를 입력에 넣어 통과시키거나 출력으로 끌어낸다. (I) 사용자 과업과 다른 출력이나 도구 호출을 일으킨다. 지시 탈취, 도구 인자를 통한 유출, 사용자에게 보이는 피싱 문구가 여기에 든다. (D) guard 작업을 부풀려 마감을 놓치게 한다.
- 사용자가 공격자인 경우(유해 요청, 직접 jailbreak)는 유해 경로가 담당한다.

### 3.3 계약

**어떤 출력 chunk도 그 chunk에 걸린 판정이 모두 끝나기 전에는 sink에 닿지 않는다.**

- 첫 토큰은 입력 전체에 대한 유해 판정이 통과한 뒤에 나간다.
- 각 chunk는 출력 guard와, 그 chunk의 역추적 창 전부에 대한 injection 판정이 통과한 뒤에 나간다.
- 도구 호출은 호출 전체가 생성되고 그 토큰들의 역추적 창이 모두 통과한 뒤에 commit한다.
- 판정이 마감 안에 오지 않으면 내보내지 않는다(fail-closed). 이 경우는 SLO 위반으로 센다.

이것은 순서에 대한 보장이고 E2.4에서 시험한다. 탐지 정확도는 보장이 아니라 측정값이다.

### 3.4 범위 문장

논문에는 다음 뜻의 한 문장을 쓴다. "임의의 적응적 공격자에 대한 완전한 차단은 범위 밖이다. 이 시스템은 검사가 sink에 대해 언제, 어느 창에 도는지를 보장하고, 탐지 품질은 꽂아 쓴 detector가 선택된 창에서 내는 품질이다."

### 3.5 공격 종류별 담당 단계

| 공격 | 담당 | 평가 | 비고 |
| --- | --- | --- | --- |
| 문맥에 숨은 유해 콘텐츠 | 1단계 scan + deep head | 기존 LG 측정, E3.5 | R@128k 0.734 (k=32, FPR 5 %) \[측정\] |
| 사용자의 유해 요청, 유해 출력 | 1단계 + 출력 guard | E3.2 | 기존 guard의 품질을 그대로 씀 |
| 지시문 형태의 간접 injection (ignore, fake completion, tool-knowledge 등 6종) | 역추적 + judge | E3.1 | 주 대상 |
| 과업과 정렬된 injection | 역추적 + judge | E5.3 | PIArena에서 ASR 0.81, 방어 적용 후에도 0.44–0.82 \[문헌\]. 한계 후보 |
| 창 여러 개에 조각난 payload | 두 경로 | E5.2 | LG의 조각 recall 0.598 \[측정\]. 한계 후보 |
| 지시문이 아닌 데이터 주입 (이력서의 숨은 문구 등) | 역추적 + judge | E3.6 | hireEZ 측정에서 주입의 90.5–95.7 %가 이 유형 \[문헌\]. 한계 후보 |
| attention 회피, decoy 창 | 역추적 | E5.1 | 이 논문이 새로 여는 공격면 |
| 지연 공격 | 스케줄러, release gate | E5.3 | 우회가 아니라 지연으로 귀결되는지 확인 |

### 3.6 범위 밖

- 타깃이나 guard 가중치의 backdoor, 학습 데이터 오염
- 신뢰하는 구간(system prompt, 사용자 과업)의 변조
- 사용자가 읽고 판단할 수 있는 보이는 거짓 정보
- 여러 요청에 걸친 장기 메모리 오염
- side channel, 모델 추출, 이미지·음성 입력

## 4. 시스템 구성과 구현 범위

&#91;embedded content: 구성도 · 유해 경로, 타깃 엔진, injection 경로, 스케줄러\]

유해 경로(위)는 Landmark Guard(LG)를 그대로 쓴다. 새로 만드는 것은 injection 경로(아래)와 release gate, 그리고 둘의 마감을 다루는 스케줄러다. 시스템 이름은 미정이라 이 문서에서는 AG라고 부른다.

### 4.1 구성 요소

| 구성 요소 | 하는 일 | 위치 | 상태 |
| --- | --- | --- | --- |
| Window map | 입력을 128토큰 블록으로 나눈다. 창은 인접 블록 2개(256토큰, stride 128)다. 블록마다 신뢰 여부를 표시한다 | 엔진, tokenizer 뒤 | 신규 |
| 유해 경로 (LG) | 모든 창을 Qwen3Guard-Gen-0.6B의 9/28층까지 읽고, 선형 router가 고른 상위 32개 창만 끝 층까지 판정한다 | 별도 프로세스, 타깃과 같은 GPU | 있음 |
| 출력 guard | 생성된 chunk의 유해 여부를 판정한다 | Qwen3Guard-Stream 또는 Gen | 기존 모델 사용 |
| Attention ledger | decode 단계마다 추적 층의 분할별 log-sum-exp를 읽어 블록별 attention 질량으로 바꾸고 chunk 단위로 누산한다 | attention backend | 신규, 핵심 |
| Trace selector | 신뢰하지 않는 창 중 누산 질량 상위 k개를 고른다. 층·head 가중은 균등 또는 calibration으로 학습한 선형 가중이다 | 엔진 | 신규 |
| Judge pool | detector를 plug-in으로 받는다. judge(과업, 창, chunk) → 통과 또는 실패. 요청 간 batch, verdict cache(창 해시 × 과업 해시) | 별도 프로세스 | 신규 |
| Release gate | chunk를 판정까지 보류한다. sink별 규칙(스트림, 도구 호출)을 가진다. fail-closed | 엔진 출력 경로 | 신규 |
| Deadline scheduler | guard job에 마감 종류(first-token, release, commit)를 붙여 마감 순으로 실행한다. 타깃 decode와 GPU를 나눠 쓴다 | guard 프로세스 + 엔진 hook | 신규 |
| 차단·격리 | 실패하면 chunk를 버리고 창을 표시한다. 그 창을 가리고 다시 생성하는 것은 선택 기능이다 | 엔진 | 신규, 재생성은 축소 후보 |

### 4.2 엔진 안에서 attention을 얻는 방법

**핵심 사실.** split-KV decode 커널은 한 요청의 KV를 분할로 나눠 분할마다 부분 attention 상태를 계산한 뒤 합친다. 합치는 데 쓰는 분할별 log-sum-exp가 이미 임시 버퍼에 있다. 분할 w의 attention 질량은 exp(lse\_w − lse\_전체)이므로(FlashInfer는 밑이 2인 로그를 쓰므로 2의 거듭제곱으로 바꾼다), 전체 attention 행렬을 만들지 않고 창 단위 질량을 얻는다.

소스 코드에서 확인한 것은 다음과 같다. 읽기만 했고 실행은 하지 않았다 \[문헌: 소스 코드\].

- **SGLang 0.5.21, Triton decode backend.** NVIDIA에서 decode는 항상 2단계로 돌고, 1단계가 분할·head별 log-sum-exp(자연로그)를 Python tensor `attn_lse`(batch, head, 분할)에 쓴다. 분할 길이는 입력 길이에 따라 정해지고 128토큰으로 고정되어 있지 않다.
- **FlashInfer 0.7.0.post1.** decode 커널이 분할·head별 값(m + log2 d)을 `tmp_s`에 두고 `VariableLengthMergeStates`로 합친다. `return_lse`는 합친 전체만 돌려준다. 분할 크기 고정(`fixed_split_size`)은 tensor-core decode 경로에서만 지원된다.
- **vLLM 0.31.0.** FlashInfer backend는 decode를 SM90·SM120에서 XQA로, SM100에서 trtllm-gen으로 넘긴다. RTX 6000 Ada(SM89)는 여기에 들지 않아 FlashInfer native wrapper가 돈다 \[해석\]. 기본 backend인 FLASH\_ATTN은 전체 lse만 돌려준다.
- 두 엔진 모두 attention 가중치를 내보내는 API는 없다.

**선택.** 1차 구현은 SGLang + Triton decode backend로 한다. 바꿀 것은 분할 경계를 128토큰으로 고정하는 것과 추적 층의 `attn_lse`를 밖으로 내보내는 것이다. 2차 구현은 vLLM + FlashInfer native wrapper다(`fixed_split_size` 8 page = 128토큰, `tmp_s` 읽기). 커널을 고칠 수 없을 때의 대안은 추적 층에서만 PyTorch로 q·K를 다시 계산해 128토큰마다 logsumexp를 내는 것이다. SGLang의 torch-native backend가 같은 gather를 이미 한다.

**확인해야 할 위험.**

- Triton backend는 우리 GPU에서 기본값이 아니다(기본은 flashinfer). 오버헤드는 수정 전 Triton 대비와 기본 backend 대비를 둘 다 보고한다.
- 128토큰 분할은 128k 입력에서 요청당 1,024개다. 커널 block 수가 늘어 느려질 수 있다 \[미측정\]. knob은 분할 크기(128, 256, 512, 1,024토큰)와 추적 층 수다. 추적하지 않는 층은 원래 분할을 쓴다.
- FlashInfer 경로는 workspace의 내부 배치에 의존하고, CUDA graph 호환이 보장되지 않는다고 소스에 적혀 있다.
- SGLang의 forward hook은 eager 경로에서만 불린다. ledger는 hook이 아니라 backend 안에 넣는다.

### 4.3 누산과 선택

- 블록 질량은 토큰 t, 층 ℓ, head h마다 m\_b = exp(lse\_b − lse\_전체)다.
- 추적 층과 head에 대해 가중 평균하고(균등 또는 학습 가중), chunk의 토큰에 대해 평균한다.
- **Rollout.** 출력 토큰이 앞서 생성된 토큰에 준 질량은 그 토큰의 블록 분포로 다시 배분한다. 생성 토큰 구간도 분할이므로 같은 값에서 계산된다. 긴 출력에서 출처가 생성 토큰 뒤로 숨는 것을 막기 위한 것이고, 효과는 E6.1에서 잰다.
- 창 점수는 인접 블록 2개의 합이다. 신뢰하는 구간의 창은 순위에서 뺀다.
- 상위 k개(기본 8) 또는 질량 임계값으로 고른다. 이미 통과한 창은 cache에서 답한다. 첫 출력 토큰은 prefill의 마지막 위치에서 나오므로 decode 커널의 ledger에 잡히지 않는다. 마지막 prompt 토큰을 decode 경로로 한 번 더 통과시키는 방법과 둘째 토큰부터 누산하는 방법을 E6.1에서 비교한다.
- 요청당 상태는 블록 수 × 추적 층 수 × 4 B다. 128k, 8개 층, head 평균 후 기준으로 1,024 × 8 × 4 B = 32 KiB다 \[계산\]. head별로 두면 32배다.

### 4.4 Release gate

- vLLM에서 넣을 자리는 `RequestState.make_request_output`이다. 이미 토큰을 보류했다가 다시 자르는 경로이고, 모든 API와 tool parser의 앞에 있다 \[문헌: 소스 코드\]. 스트리밍에는 도구 호출이 완성됐다는 이벤트가 없고 `finish_reason`으로만 끝을 알 수 있어서, gate가 호출 전체를 보류해야 한다.
- SGLang에서는 output streamer 쪽이 될 것으로 본다 \[추정, 확인 필요\].
- chunk는 첫 번째를 작게(8토큰), 이후는 32토큰 또는 문장 경계로 자른다. 도구 호출은 호출 전체가 한 chunk다.
- 판정은 다음 chunk의 생성과 겹쳐 돈다. 타깃은 멈추지 않고, 보이는 스트림만 판정 하나만큼 늦는다.

### 4.5 스케줄링

- job의 마감은 종류마다 다르다. scan과 deep head는 타깃의 첫 토큰 예상 시각, 출력 guard와 injection judge는 chunk 생성 완료 시각 + release 예산, tool commit은 호출 완료 시각이다.
- 정책은 마감 순 실행, 요청 간 batch, 요청당 판정 창 예산이다. 예산을 넘으면 hold를 늘린다. 입력이 길이 임계값(예: 4k, 창 31개)보다 짧으면 hold 없이 생성 전에 모든 창을 판정한다. 창이 적어 오탐 누적이 작고 판정 시간도 짧기 때문이다. 임계값은 E4.7에서 정한다.
- guard와 judge는 타깃과 같은 GPU에 두는 구성과 전용 GPU에 두는 구성을 둘 다 평가한다.
- **Prefix cache와의 결합.** 유해 판정과 창 verdict를 KV block 해시에 묶어 저장하면 cache hit인 턴은 scan을 다시 하지 않는다. vLLM에는 block 해시별 payload를 두는 선례(aux-output store)가 있다 \[문헌: 소스 코드\]. 구현 순서는 뒤쪽이다.

### 4.6 구현 규모

| 부분 | 언어 | 예상 LoC \[추정\] |
| --- | --- | --- |
| attention backend 변경 (분할 고정, lse 반출) | Python, Triton | 약 130 |
| ledger, selector, rollout | Python | 약 400 |
| release gate와 출력 경로 | Python | 약 250 |
| judge pool, verdict cache, scheduler | Python | 약 900 |
| LG 연동, 출력 guard | Python | 약 200 |
| 평가 harness, trace 재생기, 공격 생성기 | Python | 약 3,000 |
| 2차 구현 (vLLM + FlashInfer) | Python, C++ | 약 350 |

논문에는 엔진 변경 LoC와 전체 LoC를 나눠 적는다. Prism은 SGLang 통합 22줄을 따로 적었고, Sereno는 구성 요소별 LoC를, DirectKV는 기반 커널 위에 더한 총 LoC를 적었다.

## 5. 데이터

### 5.1 원칙

- 공격 문서와 정상 문서를 짝으로 만든다. 배경과 과업은 같고 주입만 다르다. FPR과 정상 과업 성능은 정상 쪽에서, ASR은 공격 쪽에서 잰다.
- 길이는 8k, 32k, 128k 토큰의 세 단계다. 창 수는 명목으로 63, 255, 1,023개다 \[계산\]. 문맥 한도가 131,072인 타깃(T1, T3, T4)의 128k 단계는 과업, template, 출력 몫 2,048토큰을 뺀 129,024토큰(창 1,007개)으로 맞춘다 \[계산\].
- 임계값, head 가중, k는 calibration 분할에서만 정하고 test에서는 고정한다. 분할은 배경 문서와 payload 템플릿 단위로 나눠 겹치지 않게 한다.
- 한 번 생성하고 여러 번 평가한다. 타깃 출력과 토큰별 ledger를 기록해 두면 k, chunk 크기, detector 교체는 다시 생성하지 않고 계산한다.

### 5.2 출처

크기와 라이선스는 각 저장소와 논문에서 확인한 값이다 \[문헌\].

| 데이터 | 내용과 크기 | 쓰임 | 라이선스 |
| --- | --- | --- | --- |
| AgentDojo (pkg 0.1.35) | tool agent 환경. 과업 97개, 보안 사례 629개. 데이터 7,000 + tool 문서 4,000 토큰 이하 | 도구 sink 평가의 주 데이터. 도구 출력에 정상 결과를 덧붙여 길이를 늘린 변형(AgentDojo-L)을 만든다 | MIT |
| InjecAgent | tool 응답에 주입. 1,054 사례 | 도구 sink, 단일 단계 | MIT |
| BIPIA | email·web·table QA, 요약, code QA. test prompt 86,250개 | 스트림 sink, 과업 다양성 | MIT |
| Open-Prompt-Injection | 공격 5종, 과업 7종의 toolkit | payload 생성기 | MIT |
| LongPIBench | 논문 심사, 이력서 심사, 코드 리뷰, email. 공개분은 합성 4 × 100개 | 원래 긴 입력 | MIT |
| PIArena | 13개 데이터, 1,700 표본. 긴 세트 6개의 평균 길이 8,907–19,777(단위 미기재) | 긴 문맥, 과업 정렬 공격 | MIT |
| LLMail-Inject | email assistant에 대한 공격 제출 208,095건(중복 제거) | calibration에 쓰지 않는 held-out. 사람이 쓴 적응형 공격 | MIT |
| NotInject | trigger 단어가 든 정상 prompt 339개 | 오탐 stress | MIT |
| TaskTracker | clean·poisoned 쌍을 만드는 notebook | 지시문이 많은 정상 음성 생성, 깊이 실험 | MIT |
| ResumeShield, CrackedPDFs | 숨은 지시가 든 이력서 104개(주입 72개), PDF 29,322개(주입 9,774개). E3.6에서는 숨은 지시 쪽 대조군으로 쓴다 | E3.6 | Apache-2.0, MIT |
| LG v2 유해 데이터 | 긴 정상 문서에 유해 needle | 유해 경로, 깊이 실험의 유해 쪽 | 내부 |
| Mooncake trace (FAST '25) | 약 1시간. conversation 12,031건, toolagent 23,608건. timestamp, input\_length, output\_length, hash\_ids. 32k 이상 입력이 870건과 763건, 최대 126,195 토큰 | 도착 시각과 길이 분포. 32k 이상 입력이 있는 유일한 production trace | 공개 |
| GitHub Copilot Coding Agent Traces 2026, TraceLab | tool 호출별 지속 시간 | 도구 sink의 tool 대기 시간 분포 | 공개 |

### 5.3 긴 입력 벤치마크 만들기

내부 이름은 LC-Inject다.

1. 과업 원천 5개(AgentDojo-L, InjecAgent, BIPIA, LongPIBench, PIArena의 긴 세트)에서 과업, 정답 근거, 성공 판정기를 가져온다.
2. 같은 장르의 정상 조각(다른 email, 웹 문서, 도구 출력, 매뉴얼)을 이어 붙여 목표 길이를 채운다. 배경은 지시문 밀도(256토큰 창당 명령형 문장 수)로 낮음, 중간, 높음의 세 등급으로 나눈다. 여기에 네 번째 정상 등급을 따로 둔다. 과업이 문맥 안의 지시를 따르라고 한 경우다(예: 이 절차서대로 해 달라는 과업). 출력이 지시가 든 창에 정당하게 기대므로 역추적 방식에 가장 불리하고, AlignSentinel이 기존 detector의 FPR 0.18–0.66을 보고한 유형이다 \[문헌\]. 지시문이 많은 배경의 후보는 매뉴얼, README, 절차 문서, 요청이 담긴 email, 도구 문서다. 출처별 라이선스는 W0에 확인한다.
3. payload를 깊이 10 %, 50 %, 90 % 위치에 넣는다. 공격 종류는 naive, ignore(override), fake completion, combined, tool-knowledge(AgentDojo 기본), task-aligned의 6종이다.
4. 정상 쌍은 같은 문서에서 주입만 뺀 것이다. InjecAgent에는 정상 사례가 없으므로 주입 문구를 뺀 tool 응답으로 정상 쌍을 만들고, utility는 내지 않는다.
5. 크기는 원천 5 × 길이 3 × (공격 300 + 정상 300)으로 타깃당 test 최대 9,000 요청이다. LongPIBench는 공개분 400개에서 뽑고, 원래 길이가 수천에서 수만 토큰이라 맞는 길이 단계에만 넣는다. calibration은 test의 25 %(전체의 20 %)를 따로 만든다. 생성기는 길이를 매개변수로 받으므로 E1.2와 E3.4의 중간 길이(16k, 64k 등)는 부분집합으로 따로 만든다.
6. n = 300에서 95 % CI의 반폭은 비율 0.10일 때 ±3.4 pp, 0.02일 때 ±1.6 pp다. 원천 5개를 합친 n = 1,500에서는 ±1.5 pp와 ±0.7 pp다 \[계산\]. 두 방식의 ASR 차이는 셀 하나(n = 300)로는 ±2 pp 안에서 가릴 수 없으므로, 차이에 대한 주장은 원천을 합친 값과 paired bootstrap으로 한다.
7. 공격 성공은 원 벤치마크의 판정기로 가린다. 판정기가 없으면 LLM grader를 쓰고, 사람이 라벨한 200개로 검증한다(일치율 95 % 이상을 요구, 제안값).
8. 유해 쪽은 LG v2 데이터를 그대로 쓴다. 유해 needle과 injection이 같이 든 혼합 문서 300개를 따로 만들어 두 경로가 서로 방해하는지 본다(E3.2).

### 5.4 깊이 실험용 창 데이터

- 단위는 256토큰 창이다. 유해 양성은 LG v2의 needle 창, injection 양성은 위 payload가 든 창이다. 음성은 일반 정상, 지시문이 많은 정상, NotInject의 3종이다.
- 음성 창은 20만 개 이상 둔다. 128k에서 요구되는 창 단위 FPR 5.0×10⁻⁵는 20만 개에서 기대 오탐 10개 수준이다 \[계산\].

### 5.5 성능 실험용 workload

| 이름 | 내용 | 도착과 길이 |
| --- | --- | --- |
| W-QA | 긴 문서 QA와 요약, 스트림 sink | Mooncake conversation trace |
| W-Agent | 도구 호출 sink | Mooncake toolagent trace + Copilot trace나 TraceLab의 tool 대기 시간 분포 |
| W-MT | 같은 긴 문서에 대한 여러 턴(prefix cache hit) | Mooncake의 hash\_ids로 prefix 공유 구조를 재현 |
| W-Len | 길이를 8k, 32k, 128k로 고정한 합성 부하 | Poisson |

- 요청 본문은 LC-Inject에서 길이가 맞는 것을 뽑아 채운다. 공격 비율은 1 %를 기본으로 하고 0 %와 10 %를 민감도로 둔다. 1 %는 hireEZ 측정에서 이력서의 약 1 %에 숨은 주입이 있었던 것을 따른 값이다 \[문헌\].
- 128k 요청은 8B 타깃에서 KV cache만 16.0 GiB라 48 GB GPU 한 장에 동시에 하나만 올라간다 \[계산\]. goodput 실험은 trace 전체와, 입력 8k 이상만 추린 부분 trace의 두 가지로 한다. Mooncake에서 32k 이상 입력은 conversation의 7.2 %, toolagent의 3.2 %다 \[계산\]. 길이 분포 전체는 W0에 집계한다. 128k 고정 부하는 W-Len에서 따로 본다.

## 6. 모델, 하드웨어, 소프트웨어

### 6.1 타깃 모델

구조 값은 Hugging Face config에서 확인했다 \[문헌\]. KV cache와 가중치 크기는 bf16 기준 계산이다 \[계산\].

| 모델 | 문맥 | 층 / Q head / KV head | KV cache | 가중치 | GPU | 역할 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | 131,072 | 32 / 32 / 8 | 128 KiB/토큰. 32k에서 4.0 GiB, 128k에서 16.0 GiB | 15.0 GiB | 1장 | T1, 주 타깃. PIShield, AttnTrace, PISanitizer가 쓴 모델 |
| Qwen3-4B-Instruct-2507 | 262,144 | 36 / 32 / 8 | 144 KiB/토큰. 4.5 GiB, 18.0 GiB | 7.5 GiB | 1장 | T2. 작은 타깃, 128k가 native |
| Mistral-Small-3.2-24B-Instruct-2506 | 131,072 | 40 / 32 / 8 | 160 KiB/토큰. 5.0 GiB, 20.0 GiB | 44.7 GiB | 2장 | T3. 큰 타깃, tool 사용 |
| Qwen3-8B | 32,768 | 36 / 32 / 8 | 144 KiB/토큰. 32k에서 4.5 GiB | 15.3 GiB | 1장 | 타깃으로 쓰지 않는다. 문맥 한도 때문에 W0 확인의 둘째 타깃도 T2로 바꿨다 |
| gpt-oss-20b (선택) | 131,072 | 24 / 64 / 8 | 48 KiB/토큰. 128k에서 6.0 GiB | MoE 20.9B, 활성 3.6B | 1장 | T4. 24층 중 12층만 전체 문맥을 보고 나머지는 128토큰 창만 본다. 누산의 stress case |

- Qwen3의 dense 8B·14B·32B는 native 문맥이 32,768이고 131,072는 YaRN을 켜야 한다. 128k 타깃으로는 쓰지 않는다.
- T1은 128k에서 가중치 15.0 + KV 16.0 = 31.0 GiB라 48 GB 한 장에 요청 하나가 올라가고 guard(약 1.3 GB \[측정\])와 judge가 같이 들어간다. 두 번째 128k 요청은 들어가지 않는다.

### 6.2 Guard와 judge

| 모델 | 용도 | 비고 \[문헌\] |
| --- | --- | --- |
| Qwen3Guard-Gen-0.6B | 유해 경로(LG). 9/28층 trunk, router, deep head | 문맥 32,768, Apache-2.0 |
| Qwen3Guard-Stream-0.6B | 출력 guard 후보 | token 단위 head, 문맥 8,192 |
| Llama Prompt Guard 2 (86M) | judge J1. 창만 보는 작은 분류기 | 512토큰 창. A100에서 512토큰당 92.4 ms(22M은 19.3 ms). gated라 접근 신청이 필요 |
| AgentWatcher monitor (Qwen3-4B-Instruct-2507 LoRA) | judge J2. 과업을 같이 보는 LLM judge | 원 시스템은 귀속 LLM + monitor로 탐지당 8.2 s(A100). monitor만 떼어 우리 역추적 뒤에 붙인다 |
| PIGuard (DeBERTa-v3-base, 184M) | judge J3 | 15.34 ms. E6.2 |
| DataSentinel (Mistral-7B fine-tune) | judge J4 | 약 1.6 s/query. E6.2 |

J1과 J2를 주 결과에 쓰고 J3과 J4는 detector 교체 실험에 쓴다. 각 judge의 baseline은 같은 judge를 모든 창에 돌린 것이다. J2를 쓰면 AgentWatcher 원본과의 차이가 귀속 방법 하나로 좁혀진다.

### 6.3 하드웨어

- RTX 6000 Ada 48 GB 4장을 가정한다(v4 계획서 기준).
- 10월 12일부터 12월 8일까지 57일 × 4장 × 24 h = 5,472 GPU-h다 \[계산\].
- 구성은 셋이다. 1장에 타깃, guard, judge를 같이 두는 구성. 2장(타깃 1 + guard 전용 1, 또는 두 장 모두 공존). 4장.
- DCP는 A100-40GB 4장 한 노드로 통과했다. 하드웨어 규모가 아니라 workload와 설정의 폭으로 설득한다.

### 6.4 소프트웨어

- SGLang 0.5.21(1차 엔진), vLLM 0.31.0(2차), FlashInfer 0.7.0.post1. 실험 내내 버전을 고정한다.
- 참조 구현은 Hugging Face transformers다. prefill은 SDPA로 돌리고 decode 단계만 eager attention으로 바꿔 새 토큰의 행만 얻는다. 긴 입력에서 prefill의 전체 attention 행렬을 만들면 메모리를 넘기 때문이다. W0와 ledger 검증(E2.1)에 쓴다.
- AgentDojo 0.1.35, Open-Prompt-Injection, 각 baseline의 공개 코드.
- 시간은 엔진과 클라이언트 양쪽에서 찍는다. 커널 counter는 Nsight Compute로 본다.
- decoding은 greedy다. QA는 최대 256토큰, agent는 과업이 끝날 때까지 생성한다.

## 7. Baseline

### 7.1 원칙

- 같은 detector, 같은 타깃, 같은 엔진, 같은 GPU 수에서 비교한다. 조건이 다른 문헌 수치는 표에서 따로 표시한다.
- 기존 방법은 원 논문의 설정에서 원 수치를 먼저 재현한 뒤 우리 설정에 쓴다. 결과는 이식 검증표(Table 4)로 낸다. 재현 기준은 품질 ±3 pp, 지연 ±20 %다(제안값).
- 우리 엔진에 재구현한 baseline은 원본보다 느리지 않음을 확인하고 논문에 적는다.

### 7.2 목록

| ID | Baseline | 대표하는 것 | 구현 | 보는 값 |
| --- | --- | --- | --- | --- |
| B0 | guard 없음 | 지연의 하한, ASR의 상한 | 엔진 그대로 | 전부 |
| B-oracle | 주입 창을 알려 준 선택 + 같은 judge | 선택의 상한 | 우리 엔진 | ASR, FPR |
| B-full | 모든 창을 LLM judge(J2)로 판정 | 탐지의 상한, 비용의 상한 | 우리 엔진, 부분집합 | ASR, FPR, GPU 시간 |
| B1a | 생성 전 blocking, 모든 창. 유해는 Qwen3Guard-Gen sliding, injection은 Prompt Guard 2(native 512/256 창과 우리 256/128 창을 둘 다 돌려 baseline에 유리한 쪽을 보고, max-pool) | 동기식 입력 guard | 우리 엔진 | TTFT, FPR |
| B1b | stream-first 비동기(Bedrock async, NeMo Guardrails 방식) | 비동기 출력 guard | 우리 엔진 | 판정 전에 노출된 토큰 수 |
| B1c | 모든 창을 prefill과 병렬로, 같은 GPU, 마감 없음 | naive 공존 | 우리 엔진 | TTFT, FPR, 간섭 |
| B2 | 타깃 activation probe. PIShield(Llama-3.1-8B 13층 선형 probe) | 엔진 내 intrinsic probe | 공개 코드(MIT) | ASR, FPR의 길이 의존성 |
| B3 | 두 번째 모델로 귀속 후 판정. AgentWatcher 원본, AttnTrace | attribution-gated | 공개 코드 | hit@k, ASR, FPR, 지연 |
| B4 | attention 특징과 sanitize. PISanitizer(공개 코드), Rennervate(논문대로 재구현, 코드 공개 여부 미확인) | attention 기반 방어 | 공개 코드, 재구현 | ASR, FPR, 지연 |
| B5 | 재실행·구조적 방어. MELON은 AgentDojo에서 실행, CaMeL·AttriGuard·CausalArmor는 문헌 수치 | commit 시점 방어 | 공개 코드, 문헌 | AgentDojo의 ASR과 utility, 비용 배수 |
| B6 | 유해 쪽. sliding(모든 창 끝 층까지), dense, LongGuard(문헌) | 긴 문맥 유해 guard | 기존 측정 | R@128k, 지연 |
| N1 | 무작위 k개 창 | 선택이 없을 때 | 우리 엔진 | hit@k, ASR |
| N2 | guard 모델의 attention 상위 k개 | 타깃이 아닌 모델의 attention | 우리 엔진 | hit@k |
| N3 | 얕은 층 injection probe 순위 상위 k개(LG 방식을 injection에 그대로) | 깊이 비대칭의 직접 확인 | 우리 엔진 | hit@k, FPR |
| N4 | 마지막 층만, head 균등 | 누산 범위 | 우리 엔진 | hit@k |
| N5 | 판정을 release 뒤에(hold 없음) | gate의 가치 | 우리 엔진 | 노출된 토큰 수 |
| N6 | 처음과 끝 N개 창만(truncation) | 흔한 축약 | 우리 엔진 | ASR |

B1–B6은 리뷰어가 요구할 다섯 묶음(배치 관행, intrinsic probe, 귀속 후 판정, attention 기반 방어, 재실행 방어)에 유해 쪽을 더한 것이다. N1–N6은 judge를 고정하고 선택 방법이나 gate만 바꾼 것이라 기여를 구성 요소별로 나눠 보여 준다. 표와 본문에서 '같은 judge의 모든 창'은 J1이면 B1a(품질은 B1c와 같다), J2면 B-full을 가리킨다.

### 7.3 문헌 수치로만 넣는 것

Table 1(설계 비교표)에 넣고 조건이 다르다는 것을 표시한다.

- SingProbe: token 간 지연 +0.18–0.45 %. fork의 공개 여부는 확인하지 못했다. 공개되어 있으면 실행으로 올린다.
- RTBAS: AgentDojo banking에서 4.37 s → 8.73 s. 코드를 찾지 못했다.
- CaMeL: 입력 토큰 2.82배, 출력 토큰 2.73배.
- AttriGuard: 11.5 s → 32.4 s. CausalArmor: 지연 1.22–1.30배, 코드가 있으면 실행으로 올린다. SentGuard: guard 호출당 36 ms, 문장 단위 hold는 우리 gate의 설정 하나로 재현해 E4.5에 넣는다.

### 7.4 시간이 모자랄 때의 최소 집합

B0, B1a, B1c, B-full(부분집합), B3(AgentWatcher), N1, N3, N5는 빼지 않는다. 이 여덟 개가 C1–C3의 비교 대상이다.

## 8. 실험 목록

### 8.1 지표와 통계

| 지표 | 정의 |
| --- | --- |
| ASR | 방어를 켠 상태에서 공격 목표가 달성된 비율. sink에 도달한 출력으로 판정한다 |
| FPR | 정상 요청 중 chunk가 하나라도 차단된 비율(문서 단위) |
| Utility | 정상 요청의 과업 성공률(방어 켬). 공격 요청에서 원래 과업이 성공한 비율도 따로 낸다 |
| hit@k | 공격이 성공한 요청(방어를 끈 상태에서 판정) 중, payload 토큰의 절반 이상을 담은 창이 신뢰하지 않는 창의 순위 상위 k개에 든 비율. 요청 단위 값은 chunk마다 고른 상위 k개의 합집합으로 계산하고, chunk 단위 값도 따로 낸다 |
| TTFT(생성), TTFV | 첫 토큰이 생성된 시각, 첫 chunk가 sink에 도달한 시각 |
| Release lag | chunk 생성 완료부터 release까지의 시간 |
| TPOT(보임) | sink에서 본 토큰당 간격 |
| Goodput | P90 TTFV와 P90 TPOT(보임)이 SLO를 지키는 최대 요청률. SLO = guard 없는 타깃을 전용 GPU에서 잰 P95 TTFT·TPOT × 배수 s (길이 등급별). s는 1.1, 1.25, 1.5, 2.0. 잘못 차단된 정상 요청은 SLO 위반으로 세고, 공격 요청을 차단한 것은 완료로 센다 |
| 판정 창 수 | 요청 하나에서 judge가 실제로 본 서로 다른 창의 수 |

- 품질 수치에는 bootstrap 95 % CI(1,000회)를 붙이고, 짝지은 비교는 paired bootstrap으로 한다.
- 지연은 P50, P90, P95, P99와 CDF로 내고, 실행을 3회 반복해 범위를 적는다.
- 운영점은 두 가지로 본다. 방식마다 임계값을 성공한 공격에 대한 recall이 0.80이 되게 맞추고 FPR을 비교하는 것, 그리고 문서 FPR이 1 %와 5 %가 되게 맞추고 ASR을 비교하는 것이다. 임계값은 calibration에서 정하고 ROC 전체도 남긴다.
- 통과 기준은 모두 제안값이다. GPU-h는 추정이다.

### 8.2 E0: 진행 여부를 정하는 확인 4개 (10월 12–18일)

엔진을 건드리지 않고 참조 구현으로 한다.

| ID | 무엇을, 어떤 조건에서 | 재는 값 | 통과 기준 | GPU-h, 산출 |
| --- | --- | --- | --- | --- |
| E0.1 | 타깃의 attention 역추적이 주입 창을 찾는가. T1(Llama-3.1-8B)과 T2(Qwen3-4B-Instruct-2507), 8k와 32k, 공격 600 + 정상 300(ignore, fake completion, tool-knowledge. 위치 3곳). 128k는 T1으로 100개 표본 | hit@k (k = 1, 2, 4, 8, 16, 32). 공격의 성공·실패별, 위치별. head 집계는 균등과 학습 가중 | 32k에서 hit@8 ≥ 0.90(95 % CI 하한 ≥ 0.85), 두 타깃 모두, 추적 층 8개 이하의 설정에서. 무작위 기준은 해당 창이 하나면 8/255 = 3.1 %, 겹치는 두 창이면 약 6.2 % | 23, Fig. 9 초안 |
| E0.2 | 판정 창을 줄이면 오탐이 주는가. E0.1의 문서에 J1과 J2를 모든 창 대 역추적 창(chunk마다 상위 8개, 출력 전체의 합집합)으로 적용. 정상 문서에는 과업이 문맥의 지시를 따르는 등급을 포함 | 성공한 공격에 대한 recall이 0.80인 점에서의 문서 FPR, ROC, 요청당 판정 창 수 | 32k에서 FPR(역추적) ≤ FPR(모든 창)/5. judge 하나 이상에서 FPR ≤ 5 % | 4, Fig. 10 초안 |
| E0.3 | 깊이 차이가 우리 backbone에서도 나오는가. Qwen3Guard-Gen-0.6B와 Llama-3.1-8B, 층별 선형 probe, 유해 대 injection, 음성 3종 | 층별 AUROC, 창 FPR 2×10⁻⁴(32k의 요구치)에서의 TPR. 10⁻³과 5×10⁻⁵에서의 값도 낸다. ℓ95 = 이 TPR이 최고값의 95 %에 처음 닿는 층 | ℓ95(injection, 지시문 많은 음성) ≥ 1.5 × ℓ95(유해). 미달이어도 중단 사유는 아님 | 20, Fig. 3 초안 |
| E0.4 | 마감 앞에 여유가 있는가. SGLang에서 T1의 prefill 시간과 TPOT, 같은 GPU에서 돌린 LG scan 시간, 창 8개에 대한 J1·J2 판정 시간. 8k, 32k, 128k | scan 종료 대 prefill 종료, judge 시간 대 chunk(8토큰, 32토큰) 생성 시간 | scan이 prefill보다 먼저 끝남(8k 이상). J1 판정 시간 ≤ chunk 생성 시간 | 4, Fig. 5 초안 |

**E0.1의 절차.** prefill은 SDPA로 돌리고 decode 단계만 eager attention으로 바꾼다. 생성 토큰마다 모든 층과 head의 attention 행을 128토큰 블록으로 합산하고, 신뢰하는 구간의 블록은 뺀다. chunk는 첫 8토큰, 첫 32토큰, 출력 전체의 세 가지로 본다. 집계 A는 위쪽 절반 층의 모든 head 균등 평균, 집계 B는 calibration 분할에서 학습한 층·head 가중(로지스틱 회귀), 집계 C는 기본 설정(위쪽 절반에서 고른 층 8개, head 균등)이다. 게이트는 C로 본다. 발표 자료에는 기준을 hit@32 ≥ 0.9로 적었지만, 8k 입력은 창이 63개라 상위 32개가 절반을 덮는다. 그래서 기준을 32k의 hit@8로 바꾼다. 둘째 타깃도 발표 자료의 Qwen3-8B(문맥 한도 32,768) 대신 주 타깃인 T2로 바꾼다.

**E0.2의 절차.** judge의 임계값을 훑어 ROC를 만든다. 모든 창 방식은 창 점수의 최댓값을, 역추적 방식은 출력 전체에서 판정한 창(chunk마다 상위 8개의 합집합) 점수의 최댓값을 문서 점수로 쓴다. 역추적 방식의 recall은 hit@k를 넘을 수 없으므로, 비교는 두 방식이 모두 닿는 recall 0.80에서 한다. J2에는 창마다 과업, 창, 생성된 첫 chunk를 같이 준다.

### 8.3 E1: 동기 측정 (W1–W2)

| ID | 무엇을, 어떤 조건에서 | 재는 값 | 통과 기준 | GPU-h, 산출 |
| --- | --- | --- | --- | --- |
| E1.1 | E0.3을 backbone 4개(Qwen3Guard-Gen-0.6B·4B, Llama-3.1-8B, Qwen3-4B-2507), seed 5개, 음성 창 20만 개로 확대 | 층별 곡선, ℓ95, 모든 창을 ℓ95까지 읽는 연산비 | backbone 2개 이상에서 차이 재현 | 24, Fig. 3 |
| E1.2 | 모든 창 판정의 오탐 누적. J1·J3은 정상 1,000 문서 × 길이 8단계(1k–128k), J2는 100 문서 × 4단계 | 창 단위 FPR p, 길이별 문서 FPR, 독립 가정 곡선 1 − (1 − p)^W와의 차이 | 보고 항목 | 15, Fig. 4a |
| E1.3 | attention의 집중도. 성공한 공격에서 주입 창의 질량 몫과 순위, 정상 요청에서 질량의 80 %를 차지하는 창 수 | 순위 CDF, 질량 몫, 길이별 | 보고 항목 | 20, Fig. 4b |
| E1.4 | 요청 하나의 시간표. 8k, 32k, 128k에서 prefill, scan, deep head, chunk 생성, judge, release의 시작과 끝 | Gantt, 여유 시간 | 보고 항목 | 6, Fig. 5·Fig. 8 |
| E1.5 | 지금 방식의 비용. 모든 창 J1, 모든 창 J2, AgentWatcher의 요청당 지연과 GPU 초. 3개 길이, 50 문서 | 초, GPU 초, FPR | "naive 해법의 실패"로 쓸 수치 3개 | 10, Fig. 1의 baseline 점 |

### 8.4 E2: 메커니즘 검증 (W1–W3)

| ID | 무엇을, 어떤 조건에서 | 재는 값 | 통과 기준 | GPU-h, 산출 |
| --- | --- | --- | --- | --- |
| E2.1 | ledger가 맞는가. 엔진 ledger 대 참조 구현. 8k 500 요청, 32k 100 요청, 128k 20 요청 | 블록 질량의 최대 절대 오차, 상위 8개 집합의 일치율 | 오차 ≤ 10⁻³, 일치율 ≥ 99 % | E2 전체 30 |
| E2.2 | 출력이 그대로인가. (a) 같은 분할에서 누산 켬과 끔, (b) 기본 분할 대 128토큰 고정 분할. greedy, 500 prompt × 길이 3 | 토큰 일치율, 최대 logit 차. (b)는 합산 순서가 달라져 bit 단위로 같지 않을 수 있으므로 불일치 위치를 같이 보고 | (a)는 토큰 100 % 일치, (b)는 일치율 99.9 % 이상 | 본문 수치 |
| E2.3 | 누산의 비용. 분할 크기(128, 256, 512, 1,024) × 추적 층 수(1, 4, 8, 전부) × 길이 3 × batch(8k는 1·4·16, 32k는 1·4, 128k는 1). 토큰 n개마다 한 번만 누산하는 sampling도 knob으로 둔다 | TPOT와 prefill의 증가(ms, %), 요청당 메모리, 커널 counter | 기본 설정에서 TPOT +2 % 이하, 최악 +5 % 이하. 수정 전 Triton과 기본 backend 양쪽 대비 | Fig. 14 |
| E2.4 | gate가 계약을 지키는가. 부하를 걸고 judge의 지연과 중단을 주입, 10⁵ chunk 이상 | 판정 전에 나간 chunk 수, gate 자체의 추가 지연 | 0개. gate 지연 ≤ 1 ms | 본문 수치 |

### 8.5 E3: 보안과 품질 (W3–W4, E3.6은 W6)

| ID | 무엇을, 어떤 조건에서 | 재는 값 | 통과 기준 | GPU-h, 산출 |
| --- | --- | --- | --- | --- |
| E3.1 | 주 결과표. T1–T3 × 원천 5 × 길이 3 × 시스템(B0, B1a, B-full, B2, B3, B4, N6, AG+J1, AG+J2, B-oracle). B5의 MELON은 AgentDojo의 원래 길이에서만. T3의 128k는 부분집합 | ASR, FPR, utility, 공격 하 utility, 판정 창 수. CI 포함 | 32k와 128k에서 같은 recall의 FPR 비 ≥ 5배, 원천을 합친 ASR 차의 95 % CI 상한 ≤ 3 pp(AG 대 같은 judge의 모든 창) | 660, Table 2 |
| E3.2 | 단계 × 공격 표. 유해 2종(문맥의 needle, 사용자의 유해 요청), injection 6종, 혼합 문서에 대해 scan, 출력 guard, 역추적 judge를 단독과 조합으로 | 단계별 탐지율, 단계별 오탐 | 보고 항목 | 10, Table 3 |
| E3.3 | 선택 품질. ledger(균등·학습 가중, rollout 유무) 대 N1–N4, AttnTrace, AgentWatcher의 귀속, leave-one-out(8k 부분집합) | hit@k를 k, 길이, 위치, 공격 종류별로. 주입 창 질량과 정답의 상관 | hit@8이 두 번째 모델 귀속과 ±3 pp 이내 | 25, Fig. 9 |
| E3.4 | 길이에 따른 오탐. 8k–128k 5단계, 같은 ASR에서 | judge별 문서 FPR 대 길이 | 모든 창은 증가, AG는 128k/8k 비 ≤ 2 | 15, Fig. 10 |
| E3.5 | 유해 경로 유지. injection 경로와 같이 돌릴 때 LG의 R@128k와 scan 시간. B6(sliding, dense)과 비교 | R@128k(FPR 5 %), 초 | 기존 측정(0.734, 1.10 s) 대비 ±1 pp, +5 % 이내 | 10 |
| E3.6 | 지시문이 아닌 주입(확장). hireEZ 측정이 기술한 유형을 따라 직접 합성한 데이터 주입, 8k–32k. ResumeShield와 CrackedPDFs는 숨은 지시 쪽 대조군 | hit@k, judge 정확도 | 기준 없음. 낮으면 한계로 기재 | 15 |

### 8.6 E4: 성능 (W4–W5)

| ID | 무엇을, 어떤 조건에서 | 재는 값 | 통과 기준 | GPU-h, 산출 |
| --- | --- | --- | --- | --- |
| E4.1 | 요청 하나의 지연 분해. 시스템 5개(B0, B1a, B1c, B3, AG), 길이 3 | prefill, scan, 첫 chunk, judge, queue, release의 ms | 보고 항목 | E4.1–E4.2 합 20, Fig. 8 |
| E4.2 | 지연 분포. W-Len과 W-QA. B1b와 N5에서는 판정 전에 노출된 토큰 수를 같이 잰다 | TTFT(생성), TTFV, release lag, TPOT(보임)의 CDF와 P50–P99 | 128k에서 TTFT(생성) 증가 ≤ 3 %. release lag P95 ≤ chunk 생성 시간 + judge 시간 | Fig. 11 |
| E4.3 | SLO 아래 goodput. 시스템 5(B0, B1a, B1c, B3, AG) × workload 3(W-QA, W-Agent, W-Len) × 요청률 6단계 × GPU 1·2·4장 | goodput 대 SLO 배수, attainment 대 요청률 | 입력 8k 이상에서 s = 1.25일 때 AG ≥ B0의 0.9배. B1a와 B3는 같은 s에서 AG보다 낮음. 짧은 입력이 대부분인 trace 전체에서는 0.9배에 닿는 s를 보고 | 183, Fig. 12 |
| E4.4 | trace 재생. Mooncake conversation과 toolagent(전체와 8k 이상 부분), W-MT. 20분 구간을 쓰고, 측정한 최대 부하의 절반에서 시작해 요청 수를 배수로 늘림 | SLO attainment, 99 % attainment에 필요한 GPU 수 | AG가 B1a·B3보다 적은 GPU로 99 % 도달 | 100, Fig. 13 |
| E4.5 | 누적 ablation. blocking 모든 창 → scan을 prefill과 병렬 → 역추적 선택 → verdict cache → 판정과 생성 겹치기 → 마감 순 스케줄 | 단계별 goodput, release lag, FPR | 각 단계가 지표 하나 이상을 개선 | 20, Fig. 16a |
| E4.6 | 공존 간섭. guard와 judge를 타깃과 같은 GPU에 둘 때 prefill과 TPOT의 변화, 전용 GPU 대비 | ms, % | prefill 지연 ≤ 3 % | 10 |
| E4.7 | 이득이 없는 조건. 짧은 입력(1k–4k), 문서 전체 요약, 높은 일정 부하의 정상 트래픽. 길이 적응 정책(4.5절)의 임계값도 여기서 정한다 | TTFV 증가, 판정 창 수 분포, goodput 손실 | 기준 없음. 수치를 그대로 기재 | 15, Fig. 11의 패널 |

### 8.7 E5: 적응 공격 (W5–W6)

| ID | 무엇을, 어떤 조건에서 | 재는 값 | 통과 기준 | GPU-h, 산출 |
| --- | --- | --- | --- | --- |
| E5.1 | attention 회피. 목표는 공격 성공과 주입 창의 누산 질량 최소를 동시에. white-box는 GCG 계열 joint loss(8k 이하), black-box는 LLM 공격자 500 query(32k). decoy 창 추가 | 무방어 ASR 대 주입 창 순위 곡선, 방어 시 ASR, fallback(얕은 probe 순위와의 합집합) 적용 시 ASR | 기준 없음 | 100, Fig. 15a |
| E5.2 | 조각내기. payload를 창 2, 4, 8개에 분산하고 창 경계에 걸침. scan 쪽에는 activation을 감추는 suffix | 두 경로의 탐지율, ASR | 기준 없음 | 30, Fig. 15b |
| E5.3 | 파이프라인을 아는 공격. (a) attention을 넓게 퍼뜨려 cache miss를 유도하는 판정 폭주, (b) 한 chunk 안에서 유출 완료, (c) 과업 정렬 주입 | 최악 release lag, goodput 저하, 판정 전 방출 수, ASR | (a)(b)에서 판정 전 방출 0 | 20, Fig. 15c |

### 8.8 E6: 민감도와 일반성 (W6)

| ID | 무엇을, 어떤 조건에서 | 재는 값 | 통과 기준 | GPU-h, 산출 |
| --- | --- | --- | --- | --- |
| E6.1 | 설정 민감도. k(1–32), chunk(8, 16, 32, 64, 문장), 분할 크기, 추적 층과 head, rollout 유무, 첫 토큰 처리, 선택 규칙(상위 k 대 질량 임계값) | hit@k, FPR, ASR, release lag | 기본값이 offline 최선 설정과 ±2 pp 이내 | 20, Fig. 16b |
| E6.2 | detector 교체. J1–J4 각각 모든 창 대 역추적 창 | FPR, ASR, GPU 초 | 모든 judge에서 FPR 감소 | 30, Fig. 10 |
| E6.3 | 다른 타깃과 보지 않은 데이터. gpt-oss-20b, LLMail-Inject(공격만 있어 ASR과 hit@k만 낸다) | hit@k, ASR, FPR | 재조정 없이 주 결과와 ±5 pp 이내 | 90 |
| E6.4 | 다른 엔진과 prefix cache. vLLM + FlashInfer 경로, W-MT | TPOT 오버헤드, cache hit 턴의 TTFV | 두 번째 엔진에서도 +5 % 이하 | 40 |

### 8.9 GPU 시간 합계

| 묶음 | GPU-h \[추정\] |
| --- | --- |
| E0 | 51 |
| E1 | 75 |
| E2 | 30 |
| E3 | 735 |
| E4 | 348 |
| E5 | 150 |
| E6 | 180 |
| 합계 | 1,569 |
| 재실행과 디버깅 50 % 포함 | 약 2,350 |

가용 5,472 GPU-h의 43 %다 \[계산\]. E3.1의 660에는 재실행 30 %가 이미 들어 있다. 8B 타깃의 요청당 시간을 8k 5 s, 32k 15 s, 128k 60 s로 가정했다. 128k 값은 LG 환경에서 잰 8B dense forward 50.3 s \[측정\]에서 왔고 E0.4가 실제 값으로 바꾼다. 병목은 GPU가 아니라 구현과 분석에 드는 사람 시간이다.

## 9. 그림과 표 목록

본문 그림 16개, 표 3개, 부록 표 1개다. Fig. 2, 6, 7은 개념 그림이고 나머지는 측정 그림이다. 분량이 넘치면 Fig. 2를 Fig. 6에 합치고 Fig. 13을 Fig. 12의 열로 넣는다.

### 9.1 그림

| 번호 | 내용 (축, 패널) | 필요한 값 | 실험 |
| --- | --- | --- | --- |
| Fig. 1 | 한 장 요약 Pareto. x = 지시문이 많은 정상 128k 입력의 문서 FPR(log), y = guard 없는 타깃 대비 TTFV 증가. 점은 B1a, B1c, B-full, B3, AG+J1, AG+J2. gain 영역 표시 | 시스템 6개의 FPR과 TTFV 증가 | E1.5, E3.1, E4.2 |
| Fig. 2 | 개념도. 두 위협, 지금의 배치, 세 비대칭 | 없음 | 없음 |
| Fig. 3 | 깊이. x = 상대 깊이, y = 창 FPR 2×10⁻⁴에서의 TPR. 선은 유해, injection(일반 음성), injection(지시문 많은 음성). 패널은 backbone 2개 | 층별 TPR과 AUROC, ℓ95 | E0.3, E1.1 |
| Fig. 4 | 증거. (a) 문서 FPR 대 입력 길이(1k–128k), judge 3종과 독립 가정 곡선. (b) 성공한 공격에서 주입 창의 attention 순위 CDF, 길이 3 | 창 단위 FPR p, 길이 8점의 문서 FPR, 순위 분포 | E1.2, E1.3 |
| Fig. 5 | 마감. (a) 길이별 prefill 시간과 scan 시간. (b) chunk 생성 시간과 judge 시간 | 길이 3–5점의 초 단위 시간 | E0.4, E1.4 |
| Fig. 6 | 시스템 구성도(4절의 그림) | 없음 | 없음 |
| Fig. 7 | 커널 안 누산. split-KV decode의 분할별 부분 상태와 창 질량 계산, 추가되는 상태 | 없음 | 없음 |
| Fig. 8 | 요청 하나의 timeline. 타깃, guard, 보이는 스트림의 세 줄. AG 대 B1a | 단계별 시작과 끝(ms) | E1.4, E4.1 |
| Fig. 9 | 선택 품질. (a) hit@k 대 k, 길이 3. (b) ledger 대 N1–N4, AttnTrace, AgentWatcher 귀속, leave-one-out. (c) 위치와 공격 종류별 | hit@k 표 | E0.1, E3.3 |
| Fig. 10 | 길이에 따른 오탐. x = 8k–128k, y = 같은 ASR에서의 문서 FPR. 모든 창 대 AG, judge별 패널 | 길이 5점 × 방식 2 × judge 2–4 | E3.4, E6.2 |
| Fig. 11 | 지연 분포. TTFV와 release lag의 CDF(0.99 위 확대), 길이별. 짧은 입력 패널 포함 | CDF, P50–P99 | E4.2, E4.7 |
| Fig. 12 | goodput 격자. 행은 요청률, SLO 배수, GPU 수. 열은 workload 3개. 선은 시스템 5개. 99 % 선 표시 | attainment, goodput | E4.3 |
| Fig. 13 | trace 재생. 시간에 따른 attainment, 99 %에 필요한 GPU 수 막대(도달 못 하면 "4+") | attainment 시계열, GPU 수 | E4.4 |
| Fig. 14 | 누산 오버헤드. x = 분할 크기와 추적 층 수, y = TPOT 증가(%와 ms), 길이별. 요청당 메모리 | 오버헤드 표 | E2.3 |
| Fig. 15 | 적응 공격. (a) 무방어 ASR 대 주입 창 순위. (b) 조각 수 대 탐지율. (c) 판정 폭주 시 release lag | 곡선 3개 | E5.1–E5.3 |
| Fig. 16 | (a) 누적 ablation: 품질 막대와 지연 선. (b) k와 chunk 민감도, offline 최선 표시 | 단계별 값, sweep 값 | E4.5, E6.1 |

### 9.2 표

| 번호 | 내용 | 필요한 값 | 실험 |
| --- | --- | --- | --- |
| Table 1 | 설계 비교. 선행 10여 개 × 귀속 방법, 추가 모델·pass, 판정 대상, 판정 시점, 보고된 비용 | 문헌 수치 | 2.5절, 7.3절 |
| Table 2 | 주 결과. 블록은 타깃 × 길이(32k, 128k). 8k는 부록과 Fig. 10에 둔다. 행은 하한(B0), 같은 judge의 모든 창, AG, 상한(B-oracle), 기존 시스템. 열은 원천 5 × (ASR, FPR, utility)와 평균. n과 CI 표기 | 타깃당 9,000 요청의 집계 | E3.1 |
| Table 3 | 단계 × 공격. 행은 유해 2종, injection 6종, 혼합 문서. 열은 scan 단독, 출력 guard 단독, 역추적 judge 단독, 전체, 가장 강한 외부 baseline. 오탐은 단계별로 따로 | 탐지율과 오탐 | E3.2 |
| Table 4 (부록) | baseline 이식 검증. 원 논문 수치 대 우리 재현 | 재현 값 | 7.1절 |

### 9.3 초록과 서론에 들어갈 수치

아래 일곱 개가 채워져야 초록을 쓸 수 있다. 지금은 모두 비어 있다.

1. 지시문이 많은 정상 128k 입력의 문서 FPR: 모든 창 판정 대 AG (같은 judge, 같은 ASR)
2. ASR: 무방어 대 AG, 타깃 3종 평균
3. 성공한 공격의 hit@8 (32k, 128k)
4. TTFT(생성) 증가와 release lag의 P95
5. SLO 배수 1.25에서의 goodput: guard 없는 타깃, blocking 모든 창, 두 번째 모델 귀속, AG
6. 누산의 TPOT 오버헤드와 엔진 변경 LoC
7. 요청당 판정 창 수의 평균 (128k의 창 1,023개 대비)

## 10. 단계별 일정

&#91;embedded content: 일정 · 2026년 10월 12일–12월 8일, 계획값\]

오늘(10월 10일)부터 본문 마감까지 59일이다 \[계산\]. 평가 단계는 서로 겹치게 잡았다. 집필은 W1의 2절 초안부터 조금씩 하고, W4부터 본문 전체를 실험과 나란히 쓴다.

### 10.1 주별 작업

| 주 | 구현 | 실험 | 집필 | 게이트 |
| --- | --- | --- | --- | --- |
| W0, 10/12–10/18 | 참조 구현(decode 단계 attention 행), LC-Inject 생성기(길이는 매개변수. W0에는 8k·32k 전량과 128k 표본), 성공 판정기 | E0.1–E0.4 | 없음 | G0 |
| W1, 10/19–10/25 | SGLang Triton backend에서 분할 고정과 lse 반출, ledger와 selector, 참조 구현과 대조 | E1.1, E1.2, E2.1 시작 | 2절 초안(관찰 3개) |  |
| W2, 10/26–11/1 | release gate, judge pool, verdict cache, LG 연동, 128k 데이터 전량 생성 | E1.3–E1.5, E2.1–E2.3 | Fig. 3–5 | G1 |
| W3, 11/2–11/8 | scheduler, baseline 이식(B1–B4)과 이식 검증 | E3.1(T1의 세 길이, T2의 8k·32k, AgentWatcher와 AttnTrace pass), E3.3, E2.4 | 4절 초안 |  |
| W4, 11/9–11/15 | trace 재생기, workload 구성 | E3.1(T2의 128k, T3, 나머지 baseline), E3.2, E3.4, E3.5, E4.1–E4.2 | 1절과 3절 초안, Table 2 | G2 |
| W5, 11/16–11/22 | 공격 생성기(GCG 계열, LLM 공격자) | E4.3–E4.7, E5.1 시작 | 6절 전반, Fig. 9–12 | G3 |
| W6, 11/23–11/29 | 여유가 있으면 vLLM 2차 구현 | E5.1–E5.3, E6.1–E6.4, E3.6 | 전체 초안, Fig. 13–16 | G4 |
| W7, 11/30–12/6 | 코드 동결 | 재실행과 빈 칸 채우기만 | 12월 1일 초록 등록, 내부 리뷰 반영 |  |
| 마감, 12/7–12/8 |  |  | 최종 점검과 익명화. 제출은 12월 8일 17:59 EST, 한국 시간 12월 9일 07:59 |  |

주별 GPU 사용 예정은 다음과 같다 \[추정\]. 주당 가용량은 4장 × 168 h = 672 GPU-h이고, 재실행 50 %를 더해도 모든 주가 그 안에 든다. E3.1은 재실행 전 값 510으로 넣었다.

| 주 | 예정 GPU-h | 재실행 50 % 포함 |
| --- | --- | --- |
| W0 | 51 | 77 |
| W1 | 49 | 74 |
| W2 | 56 | 84 |
| W3 | 261 | 392 |
| W4 | 329 | 494 |
| W5 | 378 | 567 |
| W6 | 295 | 443 |

### 10.2 게이트

기준은 제안값이다.

| 게이트 | 날짜 | 통과 기준 | 미달이면 |
| --- | --- | --- | --- |
| G0 | 10월 18일(일) | E0.1: 32k에서 hit@8 ≥ 0.90(두 타깃, 추적 층 8개 이하). E0.2: recall 0.80에서 FPR 비 ≥ 5배이고 judge 하나 이상에서 FPR ≤ 5 % | 둘 다 미달이면 중단하고 LG v4로 돌아간다. E0.1만 미달이면 hit@32 ≥ 0.90일 때에 한해 k = 32로 다시 재고, 아니면 중단한다. E0.2만 미달이면 FPR 비가 2배 이상일 때에 한해 J2와 질량 임계값 선택으로 다시 재고, 아니면 중단한다. 다시 재는 경우 10월 22일까지 정한다 |
| G1 | 11월 1일(일) | E2.1 통과(오차 ≤ 10⁻³, 일치율 ≥ 99 %). E2.2의 (a) 토큰 100 % 일치. E2.3이 32k에서 TPOT +5 % 이하. T1의 128k에서 hit@8 ≥ 0.85(표본 300). AgentDojo 1건과 32k 문서 1건이 gate까지 끝에서 끝으로 동작 | PyTorch 재계산 대안으로 바꿔 11월 4일까지 오버헤드를 다시 잰다. +5 % 이하면 그대로 간다. +5–15 %면 토큰 n개마다 한 번 누산하는 sampling으로 +5 % 이하에 맞추고 hit@k 손실을 같이 보고한다. +15 %를 넘으면 측정과 설계 중심으로 범위를 줄이거나 v4로 돌아간다 |
| G2 | 11월 15일(일) | Table 2가 T1의 세 길이와 T2의 8k·32k에서 채워짐. 32k에서 원천을 합친 FPR 비 ≥ 5배와 ASR 차의 95 % CI 상한 ≤ 3 pp가 judge 하나 이상에서 성립하고, 원천 5개 중 3개 이상에서 방향이 같음 | 주장을 성립하는 설정(예: 도구 출력)으로 좁힌다 |
| G3 | 11월 22일(일) | Fig. 11과 Fig. 12가 1-GPU 구성에서 완성. E2.3 최종값 +2 % 이하. 1–5절 초안 | 성능 주장을 지연 분포 중심으로 줄이고 goodput은 workload 하나만 남긴다 |
| G4 | 11월 29일(일) | E5.1–E5.3 완료. 모든 그림의 초안과 전체 초안 | E6을 부록으로 돌리거나 뺀다 |

### 10.3 줄일 순서

1. 2차 엔진(vLLM + FlashInfer)
2. T4(gpt-oss-20b)
3. 창을 가리고 다시 생성하는 복구
4. E3.6(지시문이 아닌 주입)
5. W-MT(prefix cache)
6. T3의 128k
7. 4-GPU 구성

줄이지 않는 것은 E2.1–E2.3, T1·T2의 32k·128k 주 결과, E3.3, E4.3, E5.1, 7.4절의 baseline 최소 집합이다.

### 10.4 LG v4 계획과의 관계

- v4의 유해 경로 측정(goodput 포함)은 이 논문의 설계 1 근거로 그대로 쓴다.
- W0 동안 E0가 쓰는 GPU는 51 GPU-h로 1주 가용량 672 GPU-h의 8 %다 \[계산\]. v4 실험은 계속 돌릴 수 있고, 겹치는 것은 사람 시간이다.
- G0를 통과하면 v4 중 이 논문에 쓰이지 않는 실험은 멈춘다. G0에서 중단하면 v4로 돌아가고, 손실은 W0의 사람 시간 1주다.
- 같은 학회에 두 논문을 따로 내면 LG가 양쪽에 들어가 내용이 겹친다. OSDI '27에는 하나만 내는 것으로 잡았고, 이것은 12절의 결정 항목이다.

## 11. 위험과 대응, 논문에 적을 한계

### 11.1 위험

| 위험 | 걸린 것 | 알게 되는 시점 | 대응 |
| --- | --- | --- | --- |
| 역추적이 긴 입력에서 주입 창을 못 맞힌다 | 논문 전체 | G0(128k는 표본만), 128k 전체는 G1 | 중단. k를 32로 늘려도 안 되면 LG v4로 돌아간다 |
| 정상 요청에서 고른 창이 지시문 많은 창에 쏠려 오탐 이득이 작다 | C1 | G0, G2 | 과업을 같이 보는 J2를 주 judge로 한다. FPR 비가 2배 미만이면 주장을 줄인다. 과업이 문맥의 지시를 따르는 정상 문서는 따로 집계해 최악값으로 보고한다 |
| 커널 변경이 어렵거나 느리다. 128토큰 분할의 비용을 아직 재지 않았다 | C3 | G1 | 분할을 키우고 질량이 큰 분할만 다시 잘게 계산하는 2단계 방식. PyTorch 재계산 대안 |
| injection 선행 연구가 붐빈다 | novelty | 지금 | 기여를 엔진 내 누산, release gate, 마감 스케줄, 길이별 FPR 곡선으로 한정한다. Table 1에서 차이를 적고 B2–B5와 비교한다 |
| 적응 공격에 뚫린다. "The Attacker Moves Second"에서 12개 방어 대부분이 ASR 90 % 이상이었다 \[문헌\] | 보안 주장 | W5–W6 | 탐지 우위를 주장하지 않는다. 같은 detector를 어디에 언제 돌리느냐로 주장을 한정하고 E5 결과는 그대로 싣는다 |
| 8.5주에 엔진 통합과 평가를 모두 새로 한다 | 일정 | 매 게이트 | 10.3절의 순서로 줄인다 |
| OSDI '26에 guardrail 논문이 없었다 | 리뷰어 배정, systems 기여에 대한 의문 | 제출 후 | 1쪽에서 커널과 스케줄링 기여를 먼저 내세운다. CFP의 ML 논문 조건 세 가지를 서론에서 채운다 |
| baseline을 재현하지 못한다. RTBAS와 RedVisor는 코드가 없고, MELON은 라이선스 파일이 없고, Prompt Guard 2와 Llama는 gated다 | 비교의 공정성 | W3 | 7.4절의 최소 집합을 지킨다. 문헌 수치는 조건을 표시해 Table 1에만 둔다 |
| 깊이 비대칭이 측정에서 나오지 않는다 | 관찰 1 | E0.3 | 관찰을 증거와 마감 두 개로 줄이고 설계 1은 비용 절감으로만 쓴다 |
| 128k에서는 GPU당 동시 요청이 하나라 goodput 실험이 대기열 실험이 된다 | E4의 해석 | W4 | 주 결과는 trace 분포로 내고 128k는 고정 부하로 따로 낸다 |

### 11.2 논문에 적을 한계

- 탐지 정확도는 꽂아 쓴 detector에 달려 있다. 보장하는 것은 순서다.
- 공격이 성공했는데 주입 창이 상위 k개에 없는 경우는 놓친다. 그 비율(1 − hit@k)과 조건을 적는다.
- 요약처럼 출력이 문서 전체에 기대는 과업에서는 판정 창 수가 늘어 이득이 준다.
- 짧은 입력에서는 이득이 없고, hold만큼 보이는 첫 chunk가 늦는다.
- 과업과 정렬된 주입, 지시문이 아닌 데이터 주입, 조각난 payload에는 약할 수 있다. 측정값과 함께 적는다.
- 신뢰하는 구간과 신뢰하지 않는 구간의 경계를 애플리케이션이 알려 준다고 가정한다.
- 타깃을 직접 서빙하는 제공자만 쓸 수 있다. API 뒤에 있는 모델에는 쓸 수 없다.
- 평가 장비는 RTX 6000 Ada 4장 한 노드다. Hopper와 Blackwell의 다른 decode 경로(XQA, trtllm-gen)에서는 누산을 다시 구현해야 한다.
- production 배치에서 얻은 증거는 없다.

### 11.3 원고 작성에서 지킬 것

- 본문 12쪽, 익명. 시스템 이름도 익명화 규정을 따른다.
- CFP는 대부분을 AI가 생성한 원고를 부정으로 다룬다. 이 계획서와 그림 초안은 계획용이고 원고 문장은 저자가 쓴다.
- 앞쪽 몇 쪽만으로 먼저 심사할 수 있다(방식은 확인 필요). 1–2쪽 안에 문제, 관찰 3개, 기여, Fig. 1이 모두 들어가야 한다.

## 12. 미정 사항과 확인이 필요한 것

### 12.1 결정이 필요한 것

1. OSDI '27에 내는 논문을 이것 하나로 하고 LG를 설계 1로 흡수할지, LG v4를 따로 둘지. 이 계획은 G0를 통과하면 하나로 내는 것으로 잡았다.
2. G0 기준값(hit@8 0.90, FPR 비 5배)의 확정.
3. 주 엔진. 이 계획은 변경이 가장 작은 SGLang을 1차로 잡았다. v4 실험이 쓰던 엔진과 다르면 유해 경로의 goodput 측정을 다시 해야 할 수 있다.
4. collaborator의 역할. 커널과 엔진 쪽을 누가 맡는지.
5. 시스템 이름.

### 12.2 확인이 필요한 사실

- SGLang Triton decode의 `attn_lse`를 CUDA graph를 켠 상태에서 step마다 읽을 수 있는지. 코드는 읽었지만 실행하지 않았다.
- FlashInfer의 `tmp_s`를 plan이 돌려주는 offset으로 Python에서 읽을 수 있는지.
- 128토큰 분할이 처리량에 주는 비용.
- SGLang에서 release gate를 넣을 정확한 위치.
- Rennervate와 SingProbe의 코드 공개 여부, CausalArmor의 코드.
- Prompt Guard 2와 Llama-3.1의 gated 접근 승인.
- Mooncake trace의 길이 분포 전체와 hash\_ids의 단위.
- OSDI '27 CFP의 부록·보충 자료 허용 범위와 조기 심사 방식.
- 지시문이 많은 배경 corpus의 출처별 라이선스.
- LG v2 데이터의 창 단위 라벨을 깊이 실험에 그대로 쓸 수 있는지.

### 12.3 확인을 마친 것

- AgentTracer(arXiv 2610.09935, 10월 7일)는 앞선 조사에서 열지 못했던 논문이다. 이번에 초록을 확인했다. tool call 로그에서 intent 그래프를 만들어 주입 지점을 사후에 추적하는 방법이고, 주입 지점 정확도 94.17 %를 보고한다 \[문헌\]. 엔진 안에서 도는 것도, 생성 중에 도는 것도 아니라서 직접 겹치지 않는다. 관련 연구로 인용한다.
- 타깃 모델의 층 수, head 수, 문맥 길이는 오늘 Hugging Face config에서 다시 확인했다.
- 엔진의 분할별 log-sum-exp, vLLM의 출력 경로와 prefix cache 구조는 오늘 세 저장소의 해당 tag에서 읽어 확인했다.

### 12.4 이 계획서의 수치 중 가정인 것

- 새 방향의 측정값은 아직 하나도 없다. \[측정\]으로 표시한 것은 모두 LG의 값이다.
- 통과 기준은 전부 제안값이다.
- 요청당 시간, GPU-h, LoC는 추정이다.

## 13. Sources

### OSDI '26 논문과 CFP

- [OSDI '26 프로그램](https://www.usenix.org/conference/osdi26/technical-sessions) · [OSDI '27 Call for Papers](https://www.usenix.org/conference/osdi27/call-for-papers)
- [AEGIS](https://www.usenix.org/system/files/osdi26-lei.pdf) · [Semisolates and Try](https://www.usenix.org/system/files/osdi26-lamprou.pdf) · [SMARTTalk](https://www.usenix.org/system/files/osdi26-akewar.pdf) · [SPEX](https://www.usenix.org/system/files/osdi26-zhong.pdf) · [Sereno](https://www.usenix.org/system/files/osdi26-xin.pdf)
- [Murakkab](https://www.usenix.org/system/files/osdi26-chaudhry.pdf) · [Prism](https://www.usenix.org/system/files/osdi26-yu-shan.pdf) · [Revisiting Pipeline Parallelism](https://www.usenix.org/system/files/osdi26-hwang.pdf) · [LMetric](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf) · [DirectKV](https://www.usenix.org/system/files/osdi26-luo.pdf) · [StriaTrace](https://www.usenix.org/system/files/osdi26-wu-haonan.pdf)

### 관찰의 근거

- [Zhou et al., EMNLP 2024](https://arxiv.org/abs/2406.05644) · [PIShield](https://arxiv.org/abs/2510.14005) · [TaskTracker](https://arxiv.org/abs/2406.00799) · [LEC](https://arxiv.org/abs/2412.13435)
- [LongPIBench](https://arxiv.org/abs/2608.28411) · [LongGuard](https://arxiv.org/abs/2608.27580) · [PIArena](https://arxiv.org/abs/2604.08499) · [hireEZ 이력서 측정](https://arxiv.org/abs/2605.28999)
- [Attention Tracker](https://arxiv.org/abs/2411.00348) · [AT2](https://arxiv.org/abs/2504.13752) · [AgentWatcher](https://arxiv.org/abs/2604.01194) · [AttnTrace](https://arxiv.org/abs/2508.03793) · [PISanitizer](https://arxiv.org/abs/2511.10720)
- [SentGuard](https://arxiv.org/abs/2606.02041) · [Amazon Bedrock Guardrails streaming](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-streaming.html)

### 가까운 선행

- [SingProbe](https://arxiv.org/abs/2608.30703) · [CausalArmor](https://arxiv.org/abs/2602.07918) · [MindGuard](https://arxiv.org/abs/2508.20412) · [Rennervate](https://arxiv.org/abs/2512.08417) · [BASIS](https://arxiv.org/abs/2608.08027) · [AttriGuard](https://arxiv.org/abs/2603.10749)
- [vLLM Hook v0](https://arxiv.org/abs/2603.06588) · [T3](https://arxiv.org/abs/2602.04581) · [PASTE](https://arxiv.org/abs/2603.18897) · [ARGUS](https://arxiv.org/abs/2605.03378) · [AlignSentinel](https://arxiv.org/abs/2602.13597) · [AgentTracer](https://arxiv.org/abs/2610.09935)
- [RTBAS](https://arxiv.org/abs/2502.08966) · [MELON](https://arxiv.org/abs/2502.05174) · [CaMeL](https://arxiv.org/abs/2503.18813) · [DataSentinel](https://arxiv.org/abs/2504.11358) · [InjecGuard, NotInject](https://arxiv.org/abs/2410.22770) · [Qwen3Guard](https://arxiv.org/abs/2510.14276)
- [PASCAL](https://arxiv.org/abs/2602.11530) · [SPORK](https://arxiv.org/abs/2607.03333) · [Sherlock](https://arxiv.org/abs/2511.00330) · [VIGIL](https://arxiv.org/abs/2601.05755)

### 적응 공격

- [The Attacker Moves Second](https://arxiv.org/abs/2510.09023) · [Prompt Overflow](https://arxiv.org/abs/2605.23196) · [ASTRA](https://arxiv.org/abs/2507.07417) · [Evasive Injections](https://arxiv.org/abs/2602.00750)

### 데이터와 trace

- [AgentDojo](https://github.com/ethz-spylab/agentdojo) · [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) · [BIPIA](https://github.com/microsoft/BIPIA) · [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection) · [TaskTracker](https://github.com/microsoft/TaskTracker)
- [LongPIBench 코드](https://github.com/liu00222/LongPIBench) · [LongPIBench 데이터](https://huggingface.co/datasets/RainWatcher/LongPIBench) · [PIArena](https://github.com/sleeepeer/PIArena) · [LLMail-Inject](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) · [NotInject](https://huggingface.co/datasets/leolee99/NotInject)
- [ResumeShield](https://github.com/jbarach2012/ResumeShield) · [CrackedPDFs](https://huggingface.co/datasets/volkthienpreecha/crackedpdfs)
- [Mooncake FAST '25 trace](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release) · [Azure Public Dataset (Copilot Coding Agent Traces 2026)](https://github.com/Azure/AzurePublicDataset) · [TraceLab](https://github.com/uw-syfi/TraceLab)

### Baseline 코드와 모델

- [Llama Prompt Guard 2 86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) · [PIGuard](https://github.com/leolee99/PIGuard) · [AgentWatcher 코드](https://github.com/wang-yanting/AgentWatcher) · [AgentWatcher monitor](https://huggingface.co/SecureLLMSys/AgentWatcher-Qwen3-4B-Instruct-2507)
- [AttnTrace](https://github.com/Wang-Yanting/AttnTrace) · [PIShield](https://github.com/weizou52/PIShield) · [PISanitizer](https://github.com/sleeepeer/PISanitizer) · [MELON](https://github.com/kaijiezhu11/MELON) · [CaMeL](https://github.com/google-research/camel-prompt-injection) · [Attention Tracker](https://github.com/khhung-906/Attention-Tracker)
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) · [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) · [Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) · [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) · [Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B)

### 엔진

- [SGLang](https://github.com/sgl-project/sglang) v0.5.21 · [vLLM](https://github.com/vllm-project/vllm) v0.31.0 · [FlashInfer](https://github.com/flashinfer-ai/flashinfer) v0.7.0.post1 · [FlashInfer 논문](https://arxiv.org/abs/2501.01005)

### 프로젝트 문서

- `계획서 v4` (Landmark Guard, OSDI '27)
- `claude/osdi26-evaluation-profiles-2026-09-29.md` (OSDI '26 13편 프로필)
- `claude/pivot-ideas-noninstruction-and-depth-2026-10-09.md` (방향 전환 아이디어와 Pascal 확인)
