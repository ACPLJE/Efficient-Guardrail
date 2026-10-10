# DECISIONS — 계획서와 다르게 하는 것, 정한 것, 확인한 사실

기준 계획서 `docs/asymmetric-guard-osdi27-plan-2026-10-10.md`와 다르게 구현해야 할 때, 코드보다 먼저 여기에 적고 강지의에게 확인을 받는다(CLAUDE.md 3.6). 계획서 12.2절의 "확인이 필요한 사실"을 확인하면 그 결과도 여기에 적는다.

**상태 값**

| 상태 | 뜻 |
| --- | --- |
| 결정 | 강지의가 명시적으로 정함 |
| 채택 | 기본안을 제시했고 이의가 없어 채택함 |
| 기본안 | 기한까지 이의가 없으면 채택. 기한 전에는 코드가 이 안을 따르되 바꾸기 쉽게 둔다 |
| 열림 | 측정이나 확인 결과를 보고 정한다 |
| 확인됨 / 반증됨 | 사실 항목의 결과 |

항목 형식: 번호 · 날짜 · 상태 · 관련 계획서 절 · 원래 내용 · 결정 또는 제안 · 이유 · 영향 받는 실험·그림.

---

## 1. 결정

### D-001 Python 환경: 새 3.11 venv 하나로 통일

- 날짜 2026-10-10 · 상태 **결정** · 관련: CLAUDE.md 4절, 계획서 6.4절, 구축 계획 S0
- 원래: 클러스터의 Python 3.9 venv(`/home3/212aig08/servsec/exp/secu`).
- 결정: AG의 모든 코드(하네스, 엔진 SGLang, shim, judge, AgentDojo)를 **Python 3.11 venv 하나**에 설치한다. 위치는 `/home/212aig08/ag_venv/ag311`을 제안한다(생성 전, 경로는 바꿔도 된다). 기존 3.9 venv는 LG 기존 코드 재현용으로 그대로 둔다. AttnTrace는 Python 3.10.15·torch 2.5.1을 고정하므로 **별도 venv**에 두고 파일·subprocess 경계로만 부른다(SPEC 13절).
- 이유: SGLang(2026-10-10 main)은 `requires-python >= 3.10`, AgentDojo 0.1.35도 `>= 3.10`이다 \[문헌: 각 저장소 pyproject.toml\]. 3.9로는 엔진을 실행할 수 없다. ledger shim은 SGLang 프로세스 안에서 `ag` 코드를 import하므로 하네스를 다른 venv에 두면 설치와 commit 일치 검사가 두 벌이 된다. vLLM main은 `>= 3.11`이라 선택 단계 S8g까지 같은 venv로 덮는다 \[문헌\]. LG 코드의 requirements는 Python 3.11에서 시험됐다고 적혀 있다 \[문헌: requirements.txt 머리말\].
- 영향: CLAUDE.md 4절의 Python 행, 구축 계획의 "2-venv 경계"와 "Python 3.9 문법" 규칙(폐기), CI matrix(3.11 단일). 확인할 사실은 F-01, F-02.

### D-002 collaborator는 연구 구현에 관여하지 않는다

- 날짜 2026-10-10 · 상태 **결정** · 관련: 계획서 12.1절 4번, 구축 계획 4절
- 원래: 구축 계획이 엔진 트랙(S1E, S3E, S8g, gate hook, E2.3-final)을 collaborator 몫으로 잡음.
- 결정: 모든 단계의 담당은 강지의다. 코드는 Claude가 쓰고, 클러스터 실행·결과 확인·사람 라벨은 강지의가 한다.
- 영향: 엔진 트랙과 하네스 트랙이 병렬이 아니라 한 사람의 직렬 작업이 된다. 병목은 코드 작성이 아니라 클러스터에서 job을 돌리고 결과를 확인하는 사람 시간이다 \[해석\]. 구축 계획의 순서를 조정했다(구축 계획 "개정 이력" 참조). G1 전 2주(W1–W2)에 엔진 ledger와 런타임을 한 사람이 모두 맞춰야 하므로 G1 위험이 커진다. G1 미달 시 계획서 10.2절의 11/4 재측정 규칙을 그대로 따르고 기준은 바꾸지 않는다.

---

## 2. 채택 (2026-10-10 제시, 이의 없음)

### D-003 Record v1 스키마

- 상태 **채택** · 관련: 계획서 5.1절(한 번 생성하고 여러 번 평가), 4.3절 · SPEC 5절
- 내용: 토큰별·층별 head 평균 블록 질량(fp16), chunk별·층별·head별 블록 질량(fp16, chunk 정의 3종), rollout용 생성 토큰 간 질량, 신뢰 블록 표시를 한 파일에 둔다. E2.1 차등 테스트 부분집합만 fp32로 저장한다. `schema_version = 1`, 바꾸면 올린다.
- 이유: k, chunk 크기, 집계 방식, detector 교체, B-oracle, N1·N4·N6을 재생성 없이 계산해야 한다. head별 토큰 단위 저장은 128k에서 요청당 약 0.5 GB라 쓰지 않는다 \[계산\].
- 영향: E0.1, E0.2, E1.3, E2.1, E3.3, E3.4, E6.1, E6.2.

### D-004 TTFV 측정 지점

- 상태 **채택(수정)** · 관련: 계획서 8.1절 · SPEC 6.2절
- 2026-10-10 제시안: 엔진 gate 직후와 클라이언트 수신을 둘 다 기록하고, 논문은 gate 직후 값(B0만 클라이언트 값).
- 수정: 둘 다 기록하는 것은 같다. **논문의 TTFV는 모든 시스템에서 클라이언트 수신 시각**으로 쓴다. gate 직후 시각은 release lag와 지연 분해(Fig. 8)에만 쓴다.
- 이유: 시스템마다 측정 지점이 다르면 B0 대 AG 비교에 측정 지점 차이가 섞인다. 계획서 8.1절의 정의("첫 chunk가 sink에 도달한 시각")도 sink, 즉 클라이언트 쪽이다. 같은 노드에서 돌므로 두 지점의 차이는 작지만, 같은 지점으로 재는 편이 반박할 여지가 없다 \[해석\].
- 영향: E4.1, E4.2, E4.3, E4.4, E4.7, Fig. 1·8·11·12·13, 초록 4·5. TTFT(생성)의 비교 지점은 D-033.

### D-005 공격 성공 판정용 LLM grader = Qwen3-8B

- 상태 **채택** · 관련: 계획서 5.3절 7항
- 내용: 원 벤치마크 판정기가 없는 원천은 Qwen3-8B를 grader로 쓴다. 타깃(T1 Llama-3.1-8B, T2 Qwen3-4B-Instruct-2507, T3 Mistral-Small-24B)과 다른 모델이라 자기 출력 채점을 피한다. 사람 라벨 200개와 일치율 ≥ 0.95 \[제안값\]로 검증한다. 라벨용 출력은 T2 8k 초안 데이터 200건의 B0 생성(약 2 GPU-h \[추정\])으로 만든다.
- 영향: W0-grader, E0.1(성공한 공격 집합), E3.1 이후 모든 ASR.

### D-006 G0 fallback

- 상태 **채택** · 관련: 계획서 10.2절 G0, 11.1절
- 내용: (a) AgentDojo-L이 10/15까지 준비되지 않으면 나머지 원천 4개로 G0를 판정하고 10/22에 보충한다. (b) Llama-3.1 또는 Prompt Guard 2 접근 승인이 10/14까지 나오지 않으면 T2 타깃·J2 judge만으로 E0.1·E0.2를 시작하고, "두 타깃 모두" 조건은 승인 뒤 보충한다. G0 sheet에 어느 fallback을 썼는지 열로 남긴다.
- 영향: E0.1, E0.2, G0 판정.

### D-007 계획서 10.1절 일정 이탈

- 상태 **채택** · 관련: 계획서 10.1절
- 내용: E0.4는 무수정 엔진 기동이 필요해 W0 안에 끝나지 않으므로 10/25로 민다(G0 중단 기준이 아니므로 G0 판정에는 영향 없음, G0 sheet에 pending). E1.4는 엔진 ledger·gate 뒤 11/5. E1.1·E1.2·E1.5는 11/8까지. E4.1·E4.2는 W5.
- 영향: Fig. 3·4·5·8, G0 sheet.

### D-008 E2.4를 CPU 계약 테스트와 GPU 실행으로 나눔

- 상태 **채택** · 관련: 계획서 8.4절 E2.4
- 내용: S3에서 mock judge로 10⁵ chunk 이상 계약 테스트(가상 시계)를 CPU에서 돌리고, 실제 부하 아래 GPU 실행(`run_id E2.4-gpu`)은 S4에서 11/8까지 한다. 계획서의 E2.4 결과는 GPU 실행 값이다.
- 영향: E2.4, E3.5(선행).

### D-009 E3.1을 기록에서 다시 계산할 수 있는 행부터 시작

- 상태 **채택** · 관련: 계획서 8.5절 E3.1, 10.1절
- 내용: scheduler·reset이 완성되기 전인 11/1부터, 기록된 출력에서 offline으로 계산할 수 있는 행(B0, B-oracle, N1, N6, B1a·B-full의 품질 열)을 먼저 돌린다. B1a 셀은 Table 4 행 통과 뒤(`afterok`). live 행은 scheduler·reset 완성(≤ 11/5) 뒤.
- 이유: 660 GPU-h \[추정\]를 W3–W4 두 주에 소화하려면 W3 초에 시작해야 한다.
- 영향: E3.1, Table 2.

### D-010 E4.3 2-GPU 행은 필수

- 상태 **채택** · 관련: 계획서 10.3절 7번, 6.3절
- 내용: 10.3절의 축소 후보 "4-GPU 구성"은 4-GPU 행만 가리킨다. 2-GPU는 6.3절의 기본 구성(타깃 1 + guard 전용 1, 또는 두 장 공존)이므로 E4.3에서 뺄 수 없다.
- 영향: E4.3, E4.6, Fig. 12.

---

## 3. 기본안 (기한까지 이의 없으면 채택)

| 번호 | 내용 | 기본안 | 기한 | 영향 |
| --- | --- | --- | --- | --- |
| D-011 | 품질 실험의 verdict cache | 요청마다 reset. 성능 실험은 global. 정책은 config_hash에 포함 | 10/16 | E0.2, E3.1, E3.4, judged windows |
| D-012 | E4.*의 기본 judge (계획서 미명시) | J1. Fig. 1의 AG+J2 TTFV는 E4.2에서 J1·J2 모두 돌려 채운다. E4.5 workload는 W-QA | 11/10 | E4.1–E4.7, Fig. 1·12 |
| D-013 | Fig. 5와 Fig. 8의 출처 (계획서 2.3절 C4와 9.1절 불일치) | Fig. 5 ← E0.4 + E3.5 scan 점, Fig. 8 ← E4.1. E1.4는 Fig. 8 초안 | 10/12 | Fig. 5·8 |
| D-014 | 128k 창 수 분모 (9.3절 7번) | T1·T3·T4는 129,024토큰 → 창 1,007개, T2는 131,072토큰 → 1,023개 \[계산\]. 초록 수치는 타깃별로 적는다 | 10/28 | 초록 7, E0.1-128k |
| D-015 | Table 4 지연 재현 기준 ±20 % | 원 논문과 GPU가 다르면 절대값 대신 B0 대비 배수로 비교하고 표에 GPU를 적는다 | 10/25 | Table 4 |
| D-016 | E2.3 최종값(G3)의 측정 조건 | E4에서 쓰는 ledger export 모드, scheduler·gate·judge pool 켠 상태, `run_id E2.3-final`. export off 값은 Fig. 14의 하한선 | 11/10 | E2.3, Fig. 14, 초록 6 |
| D-017 | 엔진 변경 LoC의 범위 (4.6절) | patch 파일만 "엔진 변경", shim(ledger·gate·scheduler hook)은 "AG 런타임"으로 따로 센다 | 11/20 | 초록 6, 구현 절 |
| D-018 | 첫 출력 토큰의 ledger (4.3절) | 참조 기록은 첫 토큰 행을 포함하고 `first_token_policy`로 표시. 엔진 기본값은 둘째 토큰부터. G0·E2.1 비교는 같은 정책끼리 | 10/12 | E0.1, E2.1, E6.1 |
| D-019 | 창 자격(eligibility) | 128토큰 블록에 신뢰하지 않는 토큰이 하나라도 있으면 그 블록은 신뢰하지 않음. 창은 두 블록 중 하나라도 신뢰하지 않으면 순위 대상 | 10/12 | 모든 hit@k, 판정 창 수 |
| D-020 | SLO 길이 등급 경계 (8.1절 "길이 등급별") | 입력 길이 [0, 4k) short, [4k, 16k) 8k, [16k, 64k) 32k, [64k, ∞) 128k. 혼합 workload의 goodput은 B0 대비 정규화 비율로 계산(SPEC 6.3절) | 11/10 | E4.3, E4.4, Fig. 12·13 |
| D-026 | payload 창 규칙 (8.1절은 "절반 이상을 담은 창"만 정함) | 절반 조건을 만족하는 창 1–3개. 만족하는 창이 없을 때만(연속 payload p > 512) 가장 많이 담은 창 하나. 여러 곳에 흩어진 payload는 조각마다 정하고 hit_any·hit_all을 따로 낸다 | 10/12 | hit@k, E5.2 |
| D-027 | AttnTrace(B3)의 100단어 segment 점수를 창 점수로 바꾸는 방법 | 데이터 생성 때 segment → 128토큰 블록 사상표 저장, 창 점수 = 겹치는 segment 점수의 최댓값 | 10/25 | E3.3, Fig. 9b |
| D-028 | 다중 GPU(E4.3 2·4장, E4.4)에서 replica 간 요청 라우팅 (계획서 미명시) | 하네스의 처리 중 요청 최소 정책, 결과에 표기 | 11/15 | E4.3, E4.4, Fig. 12·13 |
| D-029 | G0·초록 3의 hit@8 정의 (계획서 10.2는 chunk 3종 중 어느 것인지 미명시) | 서빙 chunk(첫 8토큰, 이후 32토큰)마다 고른 상위 8개의 합집합 기준 hit@8[union], 집계 C. 표준 정의 3종(first8, first32, all)의 hit@8도 함께 보고 | 10/14 | E0.1, G0, Fig. 9, 초록 3 |
| D-030 | "같은 ASR"에서의 문서 FPR (계획서 C1·E3.4·Fig. 10·초록 1; 운영점 미정의) | 모든 창 방식의 R80 τ에서 나온 calibration ASR에 각 방식의 calibration ASR을 맞추는 운영점(A-match). ASR은 offline gate 재생(SPEC 6.1.2)으로 계산. R80 값도 항상 함께 보고. 대안: R80으로 대신하고 그림 축 이름을 "같은 recall"로 바꿈 | 11/8 | E3.4, Fig. 10, 초록 1 |
| D-031 | 집계 C의 "위쪽 절반에서 고른 8개 층" (계획서 8.2) | 위쪽 절반 층 U에서 균등 간격 8개 `U[round(i·(|U|−1)/7)]`. 타깃별 층 목록을 `configs/targets/*.yaml`에 고정 | 10/14 | E0.1, G0, 엔진 추적 층 |
| D-032 | SLO 기준 run의 부하 조건 (계획서 8.1은 "전용 GPU의 P95"만 정함) | B0, 전용 GPU 1장, closed loop 동시 실행 1, 길이 등급별 LC-Inject, 출력 길이 고정, 3회 반복 P95의 중앙값. short 등급은 1k–4k 부분집합. tool sink 요청은 별도 등급으로, 같은 run에서 잰 출력 전체 수신 시각의 P95를 기준으로 함 | 11/10 | E4.3, E4.4, Fig. 12·13, 초록 5 |
| D-033 | TTFT(생성) 증가율의 비교 지점 (B0는 클라이언트 시각, AG는 엔진 시각) | AG 쪽을 엔진 시각 + 같은 요청의 엔진→클라이언트 전달 지연으로 보정해 클라이언트 기준으로 비교. 대안: B0의 서버 쪽 첫 토큰 시각을 얻을 수 있으면 그것을 씀. TPOT(생성) 비교(E2.3, E4.6)는 양쪽 모두 클라이언트 수신 시각을 기본으로 함 | 11/10 | C2, E4.2, 초록 4 |
| D-034 | E4.3의 2·4-GPU 구성 (계획서 6.3은 "타깃 1 + guard 전용 1, 또는 두 장 공존") | n = 2: B0·B1a·B1c·B3는 타깃 replica 2(guard·judge 공존). AG는 "replica 2 공존"과 "타깃 1 + guard·judge 전용 1"을 둘 다 돌려 보고. judge pool과 유해 guard는 replica마다 하나. n = 4: B0 계열은 replica 4, AG는 "replica 4 공존"과 "replica 2 + 전용 2" | 11/15 | E4.3, E4.4, Fig. 12·13 |
| D-035 | trace workload의 요청 본문 (계획서 5.5 "길이가 맞는 것을 뽑아") | trace의 input_length ±10 % 안으로 LC-Inject 생성기가 만든 요청. 실제/trace 토큰 비를 summary에 기록. "8k 이상 부분 trace"는 min_input_tokens = 8,192 | 11/8 | E4.2–E4.5 |
| D-036 | 운영점 임계값의 calibration 단위와 live 기본 운영점 | τ는 (target, judge, length_class)마다 원천을 합친 calibration에서 정함. live 품질 run의 기본 운영점은 F5(문서 FPR 5 %) | 10/16 | E0.2, E3.1, E3.4 |
| D-037 | T4(gpt-oss-20b, 선택)의 추적 층 | 전체 문맥을 보는 12개 층에서만 upper_half·tracked8을 고르고, sliding 층은 제외. Record에 attention sink 질량을 따로 기록 | T4 실험 전 | E6.3 |
| D-038 | `lgs1/lineage.py` 재사용 범위 (CLAUDE.md 3.3 "가능하면 그 모듈을 확장") | `canonical_hash`, `dir_fingerprint`, `code_commit`, `iter_rows`만 재사용. `RowsWriter`와 `assert_resume_compatible`은 LG의 6-키 Lineage를 전제하므로 쓰지 않고 `ag/harness/{lineage,writer}.py`에 AG 필드 전체를 비교하는 구현을 둔다 | 10/12 | 모든 run의 lineage·resume |

---

## 4. 열림 (측정 후 결정)

| 번호 | 내용 | 정하는 근거 | 시점 |
| --- | --- | --- | --- |
| D-021 | T3(Mistral-Small-24B, GPU 2장)에서 TP-2 ledger가 안 되면 | `test_tp2_ledger_one_request` 결과. 안 되면 T3는 8k·32k만 또는 참조 경로로 | 10/27 |
| D-022 | Mooncake 재생 배율 `arrival_scale`과 20분 구간 추출 규칙 | 실제 trace는 3.4–6.7 req/s, 41–57k input tok/s \[측정, clone한 trace\]로 1장 + 8B 타깃으로는 원 속도 재생 불가. W0-mooncake 집계 후 | 11/8 |
| D-023 | 기존 LG 테스트가 3.11 venv에서 green인지, 아니면 CPU gate 명령을 manifest로 한정할지 | S0 첫 작업으로 측정. 7개 모듈이 `env/s1.env`·`LG_OUT`·절대 경로를 참조 \[측정: grep\] | 10/12 |
| D-024 | "판정과 생성 겹치기"를 끈 상태(E4.5 ablation 단계) 구현 | SGLang에 요청 단위 pause가 없으면 chunk 단위 이어 생성(prefix cache 재사용)으로 흉내낸다(SPEC 12절 A-09) | 11/1 |
| D-025 | 요청당 실제 시간과 E0.2·E1.2 GPU-h 재추정 | S2의 1요청 실측 2건(T2 8k, T1 128k). E1.2는 J1만으로 약 26 h \[계산\] | 10/15 |

---

## 5. 확인할 사실

| 번호 | 사실 | 출처 | 상태 |
| --- | --- | --- | --- |
| F-01 | 로그인 노드에 Python 3.11 인터프리터가 있는가(없으면 conda 또는 uv로 받음) | D-001 | 열림 |
| F-02 | `/home/212aig08`에 venv 2개(3.11 주 venv, AttnTrace용 3.10)를 둘 공간이 있는가. venv 하나 약 10 GB \[추정\]. Record 저장: 타깃당 약 59–66 GB(8k·32k, layers = all, calibration 포함) + T1 128k 300건 약 8 GB \[계산, SPEC 5.2\] | D-001 | 열림 |
| F-03 | SGLang 0.5.21의 `requires-python`과 torch 버전, Python 3.11 wheel 설치 가능 여부 | D-001, 계획서 6.4 | 열림 |
| F-04 | 무수정 SGLang Triton decode backend가 RTX 6000 Ada(SM89)에서 기동되는가 | 계획서 4.2 | 열림 |
| F-05 | SGLang Triton decode의 `attn_lse`를 CUDA graph를 켠 상태에서 step마다 읽을 수 있는가 | 계획서 12.2 | 열림 |
| F-06 | 128토큰 분할이 처리량에 주는 비용 | 계획서 12.2 | 열림 |
| F-07 | SGLang에서 release gate를 넣을 정확한 위치(output streamer로 추정) | 계획서 12.2 | 열림 |
| F-08 | FlashInfer `tmp_s`를 plan이 돌려주는 offset으로 Python에서 읽을 수 있는가(선택 단계 S8g만) | 계획서 12.2 | 열림 |
| F-09 | Rennervate·SingProbe의 코드 공개 여부, CausalArmor 코드 | 계획서 12.2 | 열림 |
| F-10 | Llama-3.1-8B-Instruct, Prompt Guard 2 접근 승인 | 계획서 12.2 | 열림(신청 필요) |
| F-11 | Mooncake trace의 길이 분포 전체와 `hash_ids` 단위. FAST25-release README는 512토큰 블록, prefix 포함 해시라고 적음 \[문헌\] | 계획서 12.2 | 일부 확인(분포는 W0-mooncake) |
| F-12 | OSDI '27 CFP의 부록·보충 자료 허용 범위와 조기 심사 방식 | 계획서 12.2 | 열림 |
| F-13 | 지시문이 많은 배경 corpus의 출처별 라이선스 | 계획서 12.2 | 열림 |
| F-14 | LG v2 데이터의 창 단위 라벨을 깊이 실험에 그대로 쓸 수 있는가 | 계획서 12.2 | 열림 |
| F-15 | AgentDojo 0.1.35가 3.11 venv에서 SGLang과 의존성 충돌 없이 설치되는가 | D-001 | 열림 |
| F-16 | AgentDojo `requires-python >= 3.10`, AttnTrace Python 3.10.15·torch 2.5.1 고정, SGLang main `>= 3.10`, vLLM main `>= 3.11` | 저장소 pyproject·environment.yml | **확인됨** (2026-10-10, clone 기준) |
