# 비대칭 가드 구축 계획 — 시스템 논문 공개 코드 조사와 단계별 구축 순서

2026-10-10 · 강지의 (Jieui) · 기준 계획서: `docs/asymmetric-guard-osdi27-plan-2026-10-10.md`(비대칭 가드, OSDI '27). 이전 계획서 `docs/plan_osdi_2026-09-29.md`(LG v2)는 참고만.

값의 출처 표기: \[측정\] 우리가 잰 값(이 문서에서는 2026-10-10에 clone한 저장소에서 센 값 포함), \[계산\] 산술로 얻은 값, \[문헌\] 논문·문서·소스 코드에서 확인한 값, \[추정\] 근거가 약한 어림값, \[해석\] 사실에서 끌어낸 판단. 통과 기준은 모두 제안값이다. 이 문서는 코드를 담지 않는다.

## 개정 이력

**2026-10-10 개정 1** — 결정 D-001(Python 3.11 venv 하나), D-002(collaborator 없음, 모든 단계 담당 강지의)를 반영했다. 아래 본문의 해당 문장은 고쳤고, 고친 내용의 근거는 `docs/DECISIONS.md`에 있다. 인터페이스·지표·결과 스키마는 `docs/SPEC.md`가 기준이다.

- **Python**: "2-venv 경계"와 "Python 3.9 문법·numpy-only 핵심 모듈" 규칙을 폐기했다. AG 코드는 3.11 venv 하나에 설치하고, AttnTrace만 별도 venv(파일·subprocess 경계)다. CI는 3.11 단일이다. SGLang·AgentDojo가 `requires-python >= 3.10`인 것은 pyproject로 확인했다 \[문헌\].
- **1인 수행의 영향**: 엔진 트랙(S1E, S3E, S8g)과 하네스 트랙(S2, S3)이 병렬이 아니라 한 사람의 직렬 작업이 된다. 코드는 Claude가 이 저장소 세션에서 쓰므로 병목은 클러스터에서 job을 제출·확인하는 사람 시간이다 \[해석\]. 순서를 다음처럼 조정한다.
  - **W0 (G0까지)**: S0 → S1 → S2 순서가 우선이다. S1E 엔진 spike는 스크립트 하나와 GPU job 하나로 줄여 S2의 1요청 실측(③)과 같은 날 제출한다. 무수정 엔진 B0 플러그인과 patch v0는 G0 뒤(10/19–21)로 민다. E0.4(10/25)는 그대로다.
  - **W1–W2 (G1까지)**: G1 기준 세 개(E2.1, E2.2(a), E2.3 32k 1점)를 내는 S3E를 먼저 하고, S3 런타임(judge pool, gate, AG 플러그인, smoke)을 그다음에 한다. S3b 동기 그림은 GPU job만 던져 두고 분석은 G1 뒤로 미룬다. S3c Table 4는 B1a(J1)·B3 두 행만 11/1까지, 나머지는 11/8.
  - **G1 위험**: 한 사람이 2주에 엔진 ledger와 런타임을 모두 맞춰야 하므로 G1 항목 중 end-to-end smoke가 가장 늦을 수 있다. 미달이면 계획서 10.2절의 11/4 재측정 규칙을 그대로 따르고 기준은 바꾸지 않는다(D-002).
  - **선택 단계**: S8g(vLLM)는 원래 축소 1순위(cut-1)였고, 1인 수행에서는 사실상 하지 않는 것으로 본다.
- **SPEC 우선**: 인터페이스·이벤트·지표·결과 파일의 정의는 `docs/SPEC.md` v1.1이 이 문서보다 우선한다. 특히 이 문서 2.4절의 goodput 서술(attainment 90 % 교차점), 4.1절의 "이벤트 12종", 4.5절의 TTFV 측정 지점(B0만 클라이언트), 2.4절의 TPOT(보임) 분모는 SPEC 6절·3절의 정의(위반 비율 기준 goodput, 이벤트 15종, 모든 시스템 클라이언트 수신 TTFV, `N_vis − n_0` 분모)로 대체한다.

## 0. 한눈에 보기

**무엇을 알아봤나.** OSDI '24·'25·'26, SOSP '24, FAST '25, MLSys '24의 LLM serving 논문 artifact 12개와 우리가 직접 쓰는 baseline·벤치마크 저장소 2개를 clone해 읽었다(1절). 결론은 세 가지다. (1) 심사자가 보는 "functional"의 실제 수준은 "명령 하나로 결과 파일이 나오고 그림이 재생된다"이고, 그 수준에 못 미친 artifact가 절반이다(README가 가리키는 디렉터리 부재, 그림 스크립트 0개, 테스트 0개). (2) 상위 artifact가 공통으로 하는 것은 지표를 기호와 식으로 문서화하고, 측정을 시스템 밖(typed event 또는 순수 HTTP 클라이언트)에 두고, 참조 구현 대비 차등 테스트를 CI gate로 쓰는 것이다. (3) 14개 중 실행 lineage(commit·config·env·GPU·model sha)를 한 파일에 모두 적는 곳, mock 모델로 데이터→그림까지 CPU에서 통과하는 테스트, 초록 수치 자동 생성은 0개다 \[측정\]. 이 셋이 우리 차별점이다.

**코드를 어떻게 쓸 것인가.** `ag/` 패키지(config·harness·systems·judges·data·metrics·reference·sweep) + `engine_integration/sglang/`(patch 파일 1개 + shim) + `plots/`(그림당 스크립트 1개 + `numbers.json`) + `tests/manifests/{cpu,gpu,integration}`의 배치로 간다(2절). 깊이 목표는 하네스 4–6k LoC, 엔진 patch ≤ 300줄, CPU 테스트 ≥ 40 모듈, 그림 스크립트 16 + 표 3 + numbers, AE 문서에 claims → Fig → E → 스크립트 → 기대값 표다(2.3절).

**무엇을 돌려야 하나.** 그림 16개·표 4개·초록 수치 7개를 만드는 실험은 계획서의 33개에 실험 번호가 없던 작업 5개(grader 검증, Mooncake 집계, 128k 300건, G1 smoke, Table 4 이식 검증)를 더한 38개다(3절). 합계 1,569 GPU-h \[계산, 계획서 값\]이고 W3–W4의 E3.1(660)이 병목이다.

**어떤 순서로 만드나.** 달력이 아니라 의존 관계로 자른 단계 S0–S7 + 선택 S8a–g다(4절). S0 계약·자산(10/12) → S1 걷는 뼈대: mock 요청 하나가 config→events→지표→results/lineage→Fig. 9→numbers까지 CPU에서 60초(10/14) → S2 E0 임계 경로(참조 구현만, G0 10/18)와 S1E 엔진 spike(1요청 `attn_lse` 반출로 G1 위험을 G0 시점에 퇴출)를 병행 → S3E 엔진 ledger·차등 테스트(G1)와 S3 런타임 조립(G1)을 병행 → S4 Table 2(G2) → S5 goodput(G3) → S6 공격·민감도(G4) → S7 집필·동결. go/no-go를 바꾸지 않는 실험은 부속 단계(S2b·S3b·S4b·S5b)로 뒤에 둔다. 모든 주가 672 GPU-h 안에 든다 \[계산\].

**브랜치.** main 직접 push 금지, 작업마다 `<주제>/<설명>` 브랜치 + PR, main 불변식은 CPU 테스트 두 명령 green. 엔진 변경 `engine/*`는 patch 1개 + shim으로만 들어오고 접점 6개(Record v1, LedgerSink, ReleaseGate hook, info(), E2.3 계약, E0.4 합성)에서만 하네스와 만난다. 게이트마다 태그(`g0-2026-10-18` …), `freeze-2026-11-30` 뒤에는 그림·문서 렌더링만(5절).

**지금 결정할 것.** Python은 3.11 venv 하나로 결정됐다(D-001). 남은 것은 Record 스키마, TTFV의 sink 점, grader LLM, AgentDojo-L과 gated 모델의 G0 fallback, 계획서 10.1 일정 이탈 7건 등 20개(6.1절). 첫 주 행동 순서는 6.2절.

## 1. 시스템 논문 공개 코드 14개 조사

### 1.1 대상과 방법

OSDI '24·'25·'26, SOSP '24, FAST '25, MLSys '24의 LLM serving 논문 artifact와, 우리가 하네스·baseline으로 직접 쓰는 저장소를 골랐다. 전부 shallow clone을 받아 읽기만 했고(실행 안 함), LoC는 `find -name '*.py' | xargs wc -l`로 셌다 [측정, 2026-10-10 clone 기준]. OSDI '26 논문 중 공개 코드를 찾은 것은 Prism(kvcached)과 GraCE뿐이고 ECHO·SPEX·LMetric·Kairox의 공식 artifact는 검색으로 찾지 못했다 [문헌: 웹 검색].

| 저장소 | 학회 | 왜 골랐나 | 총 py LoC | 하네스·평가 LoC | 엔진 변경 |
| --- | --- | --- | --- | --- | --- |
| microsoft/sarathi-serve | OSDI '24 | 벤치마크 하네스·지표 정의·capacity search | 19,046 | 3,789 | 분리 불가(vLLM fork 전체) |
| microsoft/vidur | MLSys '24 | config → runner → metrics store → plot 구조의 본보기 | 13,067 | 5,367 | 0(시뮬레이터) |
| AlibabaPAI/llumnix | OSDI '24 | 엔진 무패치 통합, 차등 테스트 | 32,802 | 933 | 0 patch, wrapper 2,460 |
| ServerlessLLM | OSDI '24 | 다중 구성 요소 + patch 파일 + 차등 테스트 CI | 14,537 | 945 | **331줄 patch 파일** |
| snu-comparch/InfiniGen | OSDI '24 | 그림당 디렉터리 AE 구조 | 23,450 | 1,756 | **258줄**(변형 디렉터리 diff) |
| microsoft/ParrotServe | OSDI '24 | fake engine으로 CPU 통합 테스트 | 19,455 | 1,823 | 자체 엔진 |
| efeslab/Nanoflow | OSDI '25 | 커널 프로파일 저장·Gantt | 18,753 | 3,788 | 자체 엔진 |
| LoongServe | SOSP '24 | 긴 입력 serving, 860줄 AE README | 14,236 | 4,932 | LightLLM fork 전체 |
| LLMServe/DistServe | OSDI '24 | 측정→fit→시뮬레이터→실측 순서, goodput 정의 | 10,446 | 2,541 | 외부 branch(셀 수 없음) |
| kvcache-ai/Mooncake | FAST '25 | 우리가 재생할 trace의 스키마 | 99,447 | 3,211 | 0 |
| vllm `benchmarks/` + sglang `bench_serving` | 도구 | 사실상 표준 serving 벤치마크·지표 식 | 25,020 | ~17,700 | 0 |
| ethz-spylab/agentdojo | NeurIPS '24 D&B | LC-Inject의 바탕, attack/defense 플러그인 | 19,835 | 2,360 | 0 |
| ovg-project/kvcached (Prism) + GraCE | OSDI '26 | 최신 AE 관례, 버전별 monkeypatch, 테스트 manifest | 52,723 / 46,500 | 5,041+1,387 / 2,694 | patch 990줄 + monkeypatch 9,558 |
| AttnTrace + Open-Prompt-Injection | baseline B3 / payload 생성기 | 이식 난이도와 Table 4 재현 조건 | 3,432 / 5,206 | 1,046 / 1,237 | 0 |

### 1.2 관례 비교

| 저장소 | driver | config | 시스템·하네스 분리 | baseline | 지표 정의 | 결과·lineage | 그림 스크립트 | 테스트 | AE 문서 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sarathi-serve | `python -m sarathi.benchmark.main` + capacity_search(Ray, QPS 이진 탐색) | dataclass→flat argparse, 실행별 `config.yaml` dump | **반대**: MetricsStore singleton이 엔진 안 | 같은 엔진 안 scheduler 플러그인 | `metrics/README.md`에 기호·식(최상급) | csv, commit·GPU 없음 | **없음**(README가 가리키는 디렉터리 부재) | 없음, CI는 lint | README만 |
| vidur | `python -m vidur.main`; YAML→product→Ray | dataclass 트리 + `_type` 선택자 | **깨끗함**: typed event 7개가 유일한 다리 | scheduler 플러그인 registry | `docs/metrics.md` 23개 식 | csv, lineage 없음 | 1개 | 없음 | README |
| llumnix | 서버 + 클라이언트 수동 2프로세스 | yacs YAML / argparse | HTTP 경계, **시계가 시스템 쪽** | dispatch policy registry | 코드 안 16점 breakdown | json append-in-place | 없음 | unit 6.7k + e2e, **차등 테스트** | 없음 |
| serverlessllm | 셸 하나, sweep은 for | 셸 플래그/env | 하네스가 함수 호출을 time으로 감쌈 | 5곳 hard-code | avg/min/max만 | 조건별 JSON 커밋 | 막대 1개 | CPU unit + **GPU 차등 테스트가 CI gate** | README 3단 |
| infinigen | `scripts/figureNN/run.sh` 그림당 디렉터리 | bash 상수 | 없음, **symlink 교체로 시스템 선택** | 변형 디렉터리 | stdout | 파일 없음 | **없음** | 없음 | 그림별 시간·HW 표(최상급) |
| parrotserve | 없음(artifact 삭제됨) | JSON | HTTP 경계, 측정은 DEBUG 로그 | vendored fork `HACK` 표시 | prose | 로그만 | 없음 | pytest 2.6k, **fake engine(CPU)** | 환경 pin 이유 |
| nanoflow | 소스 주석 토글 | JSON 복사본 | 없음 | 다른 스크립트로 vLLM 측정 | 커널만, e2e 0 | sqlite, 전부 gitignore | 손으로 숫자 적음 | 없음 | README |
| loongserve | 스크립트 번호 순서, sqlite 캐시 | Python literal, `eval()` | 2단: offline SUT ABC / online HTTP | 외부 fork 4개 + adapter registry | 평균만, 길이 의존 SLO | `.exp`(repr→eval), 파일명이 메타데이터 | 2개(선언형) | 없음 | **AE README 860줄** |
| distserve | 2터미널 수동 | argparse + eval | HTTP 경계, 서버가 per-token 시각 반환 | 외부 branch | `RequestResult` 한 곳, goodput=90 % 교차점 | `.exp`, lineage 없음 | `_plot.py` Fig 8/9/10 | 1개(assert 없음) | README-AE(시간 예산) |
| mooncake | `benchmark.py --replay-scales` | argparse | 저장 벤치만 | 없음 | 논문 §5.1 | stdout | 없음 | 124 test, replay는 0 | trace README(최상급 데이터 문서) |
| vllm/sglang bench | `vllm bench serve/sweep` | argparse, sweep JSON | **순수 HTTP 클라이언트**, 엔진 import 0 | 백엔드당 함수 1개 | `cli.md` 측정점·식, goodput 다중 SLO | run당 JSON, lineage 없음 | group-by plotter | **CPU scripted-clock 테스트** | 사용자 문서 |
| agentdojo | click CLI, pair JSON, skip-if-exists | pydantic + YAML | `BasePipelineElement.query` 5-tuple | attack registry / defense는 if-elif | boolean, 집계는 노트북 | 36,679개 JSON 커밋 | 노트북 1개 | unit 1.2k, CI | docs 사이트 |
| kvcached / GraCE | env var 스위치 / `run_case` 루프 + `RUN_ID` | YAML + env / 텍스트 workload 목록 | **런타임 monkeypatch 패키지**(version range) | 같은 바이너리 on/off, ABBA | vLLM 지표 그대로 | **`setup.json`(sha·GPU·버전·명령줄)** / `results/raw/<run_id>/` | **그림당 wrapper 1개, `--run-id latest`** | **cpu/gpu/integration manifest 강제** | README 20 KB(30분 검증, claims) |
| attntrace / OPI | `os.system(nohup)` / matrix→subprocess, `--dry-run` | JSON | HF 내부 접근(엔진 불가) / `Model.query→str` | in-process registry | 분모 불일치 | 파일명 / **fingerprint + CacheMismatchError** | 없음 | 없음 | 명령만 |

### 1.3 모두가 하는 것, 상위만 하는 것, 아무도 안 하는 것

**사실상 모든 artifact가 하는 것** (못 하면 표준 미달):
- 시스템 선택을 CLI 플래그·enum 하나로 하고 baseline을 같은 루프에 넣는다.
- 출력 길이를 고정한다(`ignore_eos` + `max_tokens`).
- warm-up 뒤 측정하고 요청 단위 원시 값을 남긴다.
- 결과 파일 존재로 resume한다.

**상위 3–4개만 하는 것** (여기까지 하면 AE 심사에서 두드러진다):
- 지표를 기호와 식으로 문서화한다(sarathi `metrics/README.md`, vidur `docs/metrics.md`, vllm `cli.md`).
- 측정과 시스템을 완전히 분리한다: typed event가 유일한 다리(vidur), 또는 엔진 import 0인 HTTP 클라이언트(vllm/sglang bench).
- 시계 식별식을 CPU에서 테스트한다(vllm: `(latency − ttft) == sum(itl)`).
- 실행 lineage를 남긴다(kvcached `setup.json`), 지문 불일치 시 거부한다(OPI `CacheMismatchError`).
- 그림당 스크립트 하나, `results/raw/<run_id>` → `processed` → `figures/figNN.pdf`(GraCE), 그림별 소요 시간·HW 표(infinigen, loongserve).
- 테스트 경계 manifest(cpu/gpu/integration)를 도구로 강제한다(kvcached).
- 참조 구현 대비 차등 테스트가 CI gate다(serverlessllm, llumnix).

**아무도 안 하는 것** (우리 차별점): 14개 중 commit hash·config hash·env·GPU·model sha·data manifest를 한 파일에 모두 적는 곳은 0개, mock 모델로 CPU에서 데이터→시스템→지표→그림까지 통과하는 테스트도 0개, 초록 수치 자동 생성(`numbers.json`)도 0개다 [측정].

### 1.4 실제 깊이

- **엔진 변경 LoC를 셀 수 있는 곳**은 serverlessllm(patch 331줄), infinigen(변형 디렉터리 258줄), kvcached(patch 143–249줄 + monkeypatch 패키지) 셋뿐이다. 나머지는 fork 전체거나 외부 branch라 심사자가 세지 못한다 [측정]. 하네스는 보통 1–5k LoC다. 계획서 4.6절의 하네스 3,000 LoC [추정]는 이 범위의 하한이고, 엔진 변경 약 130 LoC [추정]는 kvcached·infinigen과 같은 자릿수라 "Prism은 SGLang 통합 22줄"처럼 따로 적을 가치가 있다.
- **테스트 0개**: sarathi, vidur, infinigen, nanoflow, loongserve, attntrace, OPI, GraCE. OSDI '24 artifact 6개 중 4개가 테스트 0이다 [측정]. 테스트가 0인 artifact는 전부 살아 있는 버그를 가지고 있었다(sarathi 미정의 변수 3곳, distserve import 깨진 스크립트 2개, nanoflow merge marker 6파일). 1.5명이 8.5주에 그림 16개를 돌리는 우리에게 테스트는 장식이 아니라 재실행 비용을 줄이는 도구다 [해석].
- **실제 CPU CI**: vllm bench(4.7k LoC, scripted clock), sglang, kvcached(62 cpu 모듈), serverlessllm, agentdojo. **mock 모델**: parrotserve `fake_engine_server.py`, serverlessllm `dummy_backend.py`, loongserve `dummy`, kvcached sys.modules fake. 단 데이터→그림까지 끝까지 도는 mock 테스트는 없다.
- **심사자가 보는 "functional" 수준**은 "명령 하나로 결과 파일이 나오고 그림이 재생된다"이지 테스트 커버리지가 아니다 [해석]. 그러나 그 수준에 못 미친 artifact가 절반이다: sarathi `osdi-experiments/` 부재, parrotserve artifact 삭제, nanoflow README가 없는 스크립트 광고.

### 1.5 피해야 할 것 (증거와 함께)

1. **측정 코드를 시스템 안에 두기** — sarathi MetricsStore singleton, llumnix `/generate_benchmark`가 서버 안에서 시각을 찍음. baseline을 다른 엔진으로 돌리면 시계가 달라져 비교 불가. → 플러그인은 이벤트만 내고, 식은 `ag/metrics/`에만 둔다.
2. **시스템 선택을 설치 상태 변경으로 하기** — infinigen `rm` + `ln -s`, nanoflow 소스 주석 토글. 병렬 실행 불가, 중단 시 환경 파손. → registry + YAML `type:`.
3. **sweep 격자·SLO 상수를 여러 곳에 복제** — distserve는 세 벌이 이미 어긋남(OPT-66B SLO (0.4, 0.1) vs (2.5, 0.15)). → `configs/slo.yaml` 하나를 runner와 plots가 같이 읽고 hash.
4. **lineage 없음, 파일명이 메타데이터** — loongserve `.exp`를 `eval()`로 읽고 조건은 파일명, llumnix json append-in-place. → `lineage.json` + 행마다 `run_id`, 불일치 resume 거부.
5. **그림에 숫자를 손으로 적기** — nanoflow `durations=[70,63,71,70]`, agentdojo 노트북이 loader 복제. → `plots/figNN.py`가 `results/`만 읽고 `numbers.json`도 같은 loader.
6. **테스트 0 + 살아 있는 버그** — 위 1.4절. → CPU manifest가 PR gate, CI에서 `<<<<<<<`와 절대 경로 grep.
7. **Poisson을 자르기** — sarathi/vidur가 간격을 `3/qps`로 cap, sglang은 yield 후 sleep(정규화 없음). → vllm 방식(사전 계산 Gamma, 마지막 도착 = N/rate로 정규화, seed 기록).
8. **워밍업·캐시 상태를 결과에 섞기** — 같은 서버 재실행 시 prefix cache로 throughput 부풀음, loongserve 계측 모드가 CUDA sync 강제. → 조건마다 `reset()`, warm-up 기록·제외, 계측 모드는 별도 config_hash.
9. **분모가 다른 지표** — attntrace precision과 ASV의 분모가 다름, serverlessllm throughput이 prompt 토큰을 생성 토큰으로 셈. → 모든 비율에 분모 열, 식별식 CPU 테스트.
10. **하드코딩된 클러스터 경로** — loongserve `/mnt/petrelfs/...` 16곳, kvcached `/home/xingqi/`. → 경로는 `env/*.env`와 YAML만.
11. **외부 서비스가 측정 경로에 묶임** — sarathi `wandb.Api()` 무조건 호출, llumnix unit test가 Ray head 요구. 우리 계산 노드는 네트워크가 없다. → 측정 경로에 네트워크 의존 0, Ray 대신 Slurm array.
12. **baseline에 우리 오버헤드를 넣기** — llumnix 'round-robin baseline'이 Llumnix manager hop을 포함, nanoflow는 vLLM을 다른 workload로 측정. → B0는 patch 미적용 로그로 확인한 엔진 그대로.
13. **vendored fork·외부 branch에 숨은 변경** — parrotserve 3rdparty에 `ignore_eos` 강제·max len hard return. → 엔진 변경은 patch 파일 1개, hunk마다 `# AG:` 표시.
14. **readiness를 `sleep`으로** — parrotserve `sleep 15`, distserve "줄이 뜰 때까지 사람이 기다림". → `/health` 폴링 + timeout + 프로세스 사망 감지.
15. **결과를 통째로 gitignore하면서 샘플도 없음** — nanoflow. → `tests/fixtures/results_mock/`에 작은 결과 트리를 커밋해 그림 스크립트가 GPU 없이 돈다.

### 1.6 저장소별 한 줄 교훈

| 저장소 | 우리가 가져올 것 (경로) | 우리가 피할 것 |
| --- | --- | --- |
| sarathi-serve | 지표를 기호(a_r, s_r, f_r, c_r)와 식으로 적는 `metrics/README.md`; 요청 타임스탬프 상태 기계 `sequence_state.py`; capacity search 골격(Ray → Slurm array로) | 측정이 엔진 안, README가 약속한 `osdi-experiments/` 부재 |
| vidur | typed event ↔ MetricsStore 분리(`events/*`, `metrics_store.py`); 지표 이름 enum 한 곳(`metrics/constants.py`); YAML axes → product → job | lineage 없음, 그림-데이터 매핑 없음 |
| llumnix | `backend_interface.py` ABC + docstring; `test_correctness.py`(raw == wrapped) = E2.2의 본; 실행 후 불변식 fixture(`check_log_exception`) | 시계가 서버 안, append-in-place json |
| serverlessllm | `vllm_patch/*.patch` + apply/check 스크립트; `store_test.py` 차등 테스트가 CI fail; 조건마다 subprocess 격리 + GPU 메모리 출력 | baseline이 5곳 hard-code, percentile 없음 |
| infinigen | 그림당 디렉터리 = claim→script 매핑; 그림별 소요 시간·HW 표; 변형 디렉터리로 엔진 LoC 세기 | symlink로 시스템 선택, plot 코드 0 |
| parrotserve | `testing/fake_engine_server.py`(per-token sleep, seeded 길이)와 `localhost_server_daemon.py`(contextmanager 프로세스) | 측정이 DEBUG 로그, artifact가 hotfix로 삭제됨 |
| nanoflow | 커널 microbench 레시피(warm-up 10, CUDA graph ×100, CUDA event); 스케줄을 JSON으로 dump해 Gantt | 소스 편집으로 실험 선택, 그림에 숫자 손으로 |
| loongserve | AE README 골격(그림별 human/machine 시간, 기대 콘솔 줄, 사전 계산 결과, subset 허용); 길이 의존 SLO; 지배당한 run 제거 규칙 | `eval()`, 파일명 메타데이터, 절대 경로 16곳 |
| distserve | 측정→fit→시뮬레이터→실측 순서; `structs.RequestResult` 한 곳; goodput = attainment 90 % 교차점(`find_intersection`); SLO 배수 sweep은 같은 원시 파일 재평가 | 2터미널 수동, 시계 두 종류 섞음, 격자 세 벌 |
| mooncake | 4필드 trace 스키마와 README(layout 표, 통계표, 샘플, 의미); `write_json_atomic` + `schema_version` | replay 코드 없음, stdout만 |
| vllm/sglang bench | `get_request`(Gamma 사전 계산·정규화·self_timed replay); `calculate_metrics`(goodput 다중 SLO); scripted-clock 식별식 테스트; `p{P}_{metric}_ms` 키 체계 | lineage 없음, 엔진 측 이벤트 없음 |
| agentdojo | `attack_registry.py`(decorator registry); canary로 주입 가시성 확인(`get_injection_candidates`); `check_suites.py` 식 데이터 gate; `tool_output_formatter` hook(AgentDojo-L) | defense는 if-elif, 오류를 ASR에 합침, 노트북 1개 |
| kvcached / GraCE | `integration/patch_base.py`(PatchManager, 버전 범위, 적용 로그 한 줄); `tests/manifests/{cpu,gpu,integration}.txt` + 분류 검사 도구; `setup.json`; `results/raw/<E>/<run_id>/` + `latest.txt`; 그림당 wrapper; 30분 검증 README | 사용자 경로 하드코딩, AE 층이 없는 kvcached |
| attntrace / OPI | `attribute(question,[context],answer)` 그대로 import(재작성 금지); OPI `artifacts.py`(fingerprint·atomic save·CacheMismatchError); `run.py`의 matrix→Experiment→`--dry-run`; 공격 payload 템플릿 | 전역 `set_default_tensor_type`, `os.system(nohup)`, 하이퍼파라미터 파일명 |

## 2. 우리 코드의 방식, 내용, 깊이

### 2.1 방식: 원칙을 디렉터리로

CLAUDE.md의 원칙 8개를 디렉터리로 옮긴다. 기존 `lgs1`–`lgs8`은 건드리지 않고 새 패키지 `ag/`가 import한다(규칙 3.7). 모든 항목에 출처 저장소를 적었다.

```
ag/                          # 새 패키지. 코드·주석 영어
  config/                    # YAML 로더 + typed dataclass + config_hash           (vidur, sarathi, OPI fingerprint)
  harness/                   # 측정만 한다. 시스템을 import하지 않는다             (vidur, vllm bench)
  systems/                   # 플러그인. System.run(request) -> Iterator[Event]    (llumnix ABC, agentdojo registry)
  judges/                    # judge 플러그인 J1–J4 + mock                         (agentdojo pi_detector, OPI defenses)
  data/                      # LC-Inject 생성기, trace 로더, manifest               (agentdojo, mooncake, vllm datasets)
  metrics/                   # 식 하나 = 함수 하나                                 (sarathi/vidur metrics, distserve structs)
  reference/                 # oracle: HF SDPA prefill + eager decode ledger        (attntrace attention_utils, serverlessllm store_test)
  sweep/                     # YAML axes -> jobs.jsonl -> Slurm array               (vidur config_explorer, sarathi capacity_search)
engine_integration/
  sglang/                    # patch 파일 1개 + apply/check/remove + 버전별 shim     (kvcached integration, serverlessllm vllm_patch)
  vllm/                      # 2차(축소 1순위)
plots/                       # fig01.py … fig16.py, table2–4.py, abstract_numbers.py, _common.py   (GraCE, vllm sweep/plot.py)
tests/                       # manifests/{cpu,gpu,integration}.txt + 분류 검사 도구     (kvcached)
  fixtures/results_mock/     # 그림 스크립트가 GPU 없이 도는 작은 결과 트리
slurm/                       # common.sh(기존) + 실험별 sbatch 템플릿
configs/                     # exp/eNN_x.yaml, systems/*.yaml, workloads/*.yaml, slo.yaml, table4/*.yaml
results/                     # gitignore. /home/212aig08 쪽. raw/<E>/<run_id>/..., processed/, latest.txt
docs/                        # SPEC.md, DECISIONS.md, metrics.md, AE.md
```

### 2.2 내용: 모듈 책임

**`ag/harness/`** — 시계는 여기에만 있다(`time.perf_counter_ns()` 한 종류).
- `events.py`: `EventType(IntEnum)` = ARRIVED, ADMITTED, PREFILL_START, PREFILL_DONE, TOKEN_GENERATED, CHUNK_GENERATED, JUDGE_SUBMITTED, VERDICT, CHUNK_RELEASED, CHUNK_BLOCKED, FINISHED, ERROR. `Event(run_id, request_id, type, t_ns, chunk_id?, window_id?, judge_id?, n_tokens?, payload)`. (vidur `event_type.py`, distserve `lifetime.py`, llumnix `timestamps.py`)
- `request_state.py`: 요청당 타임스탬프 상태 기계. 읽기 전 접근은 raise. (sarathi `sequence_state.py`)
- `runner.py`: 요청 생성기 → 도착 시각 gating → `system.run(request)` → 이벤트 수집 → JSONL. open-loop는 도착 시각을 미리 계산하고 하네스가 gate한다. (vllm `serve.py:get_request`)
- `writer.py`: `events.jsonl`(행마다 `run_id`, `schema_version`), `summary.json`을 tmp + `os.replace`로 atomic 저장. (mooncake `write_json_atomic`, 기존 `lgs1/lineage.py:RowsWriter`)
- `lineage.py`: 기존 `lgs1/lineage.py.Lineage`를 상속·확장(env_hash, gpu_name, gpu_count, hostname, slurm_job_id, started_at, finished_at, engine_version, server_cmd). (kvcached `write_setup`)
- `checks.py`: G0–G4와 실행 후 불변식을 `Check(name, ok, detail)` 목록으로 — 종료 시 보류 chunk 0, judge queue 비어 있음, 판정 전 방출 수(E2.4), 로그에 Traceback 0. (kvcached `tests/e2e/run.py`, llumnix `check_log_exception`)
- `reset.py`: 조건 사이에 prefix cache·verdict cache·judge cache를 비우는 hook. (vllm `reset_caches`, sglang `flush_cache`)

**`ag/systems/`** — 플러그인은 이벤트만 **낸다**. 계산하지 않는다.
- `base.py`: `System.setup(cfg, lineage) / run(request) -> Iterator[Event] / reset() / teardown() / info() -> dict`. (llumnix `backend_interface.py`)
- `registry.py`: `@register_system("B1a")`. (agentdojo `attack_registry.py`)
- `b0_engine.py`, `b1a_blocking.py`, `b1b_async.py`, `b1c_parallel.py`, `b2_pishield.py`, `b3_attntrace.py`, `b3_agentwatcher.py`, `b4_pisanitizer.py`, `ag.py`. N1–N6와 B1c는 `ag.py`의 config 오버레이(파일 복사 금지). (parrotserve `GlobalSchedulerConfig` 불리언, nanoflow의 반례)
- `mock_system.py`: 결정적 mock. per-token sleep 또는 가상 시계, 합성 attention(주입 창에 질량 집중), judge 지연·verdict를 요청 매개변수로. (serverlessllm `dummy_backend.py`, parrotserve `fake_engine_server.py`)
- `sglang_client.py`: 서버 launch / `/health` 폴링 / teardown / SSE 파싱 공통 어댑터. (vllm `sweep/server.py:ServerProcess`)

**`ag/judges/`**: `Judge.verdict(task, window, chunk) -> (bool, score, latency)`; `prompt_guard2.py`(J1), `agentwatcher_monitor.py`(J2), `piguard.py`(J3), `datasentinel.py`(J4, OPI `apps/DataSentinelDetector.py` 포팅), `mock_judge.py`.

**`ag/data/`**: `lc_inject/`(생성기 3단: 요청 jsonl → 시스템 출력 jsonl → 채점; payload 템플릿은 OPI `attackers/*`; canary로 주입 가시성 확인), `agentdojo_l.py`(`tool_output_formatter` hook으로 길이 확장; AgentDojo는 3.11 주 venv에 함께 설치하고 pair JSON을 읽음(D-001)), `traces/mooncake.py`(4필드 로더 + 결정적 hash_id→토큰 확장, explicit RNG), `arrivals.py`(poisson/gamma/trace, 사전 계산 + 정규화, seed 기록), `manifest.py`(길이 분위수·공격 비율·sha → `data_manifest_sha`). `tests/data/`에 각 trace 50줄 절단본 커밋.

**`ag/metrics/`**: `per_request.py`(TTFT_gen, TTFV, release_lag, TPOT_visible, exposed_tokens, judged_windows), `quality.py`(ASR, FPR, utility, hit@k; 기존 `lgs1/metrics.py:strict_matched_fpr` 재사용), `goodput.py`, `aggregate.py`(`results/raw/**` → run당 1행 parquet). `names.py`의 enum 하나에서 JSONL 키·`numbers.json`·그림 파일명·`docs/metrics.md` 표를 생성한다(vidur `constants.py`).

**`ag/reference/`**: `ledger_oracle.py`(HF SDPA prefill + eager decode, 128토큰 블록 질량; 수학은 attntrace `attention_utils.py`), `diff.py`(E2.1 오차·상위 8 일치, E2.2 토큰 일치 → `diff_report.json`, 실패 시 CI fail). oracle 출력은 (model_sha, data_manifest_sha, config_hash)로 content-addressed 캐시.

**`ag/sweep/`**: `expand.py`(YAML axes → product → `JobConfig.key()` → `jobs.jsonl`), `submit.py`(index → `--array=0-N` sbatch, `%N` 금지, 동시성은 array 크기·`--dependency`), resume는 결과 파일 존재로 skip, `capacity.py`(SLO 아래 최대 부하 이진 탐색, 선택).

**`engine_integration/sglang/`**: `ag-sglang-0.5.21.patch`(Triton decode 분할 고정 + `attn_lse` 반출), `patch.sh / check_patch.sh / remove_patch.sh`(dry-run 멱등), `shim/`(ledger·gate·scheduler hook, `PatchManager` + 버전 범위, 적용 시 한 줄 로그 → 하네스가 grep해서 `system_active` 체크; B0에서는 그 줄이 없어야 함), `README.md`(상류 commit, hunk 표시 규칙 `# AG:`).

**`plots/`**: `_common.py`(run 로더, `--run-id latest`, group-by/filter/error bar, 누락 (system, workload) 셀을 **실패로** 보고), `fig01.py`…`fig16.py`, `table2.py`…`table4.py`, `abstract_numbers.py` → `numbers.json`(9.3절 7개). 모두 `results/`만 읽는다.

**`tests/`**: `manifests/{cpu,gpu,integration}.txt` + `tools/check_test_classification.py`; `test_metrics_identity.py`(scripted clock: SPEC 6.2 식별식 1–3, `TTFV == TTFT_gen + (g0 − f) + (l0 − g0) + (d0 − l0)`(SPEC 6.2 식별식 1)), `test_mock_end_to_end.py`(데이터 → mock system → events → JSONL → `plots/fig11.py`), `test_lineage_mismatch.py`, `test_patch_applies.py`(dry-run), `test_trace_loader.py`, `test_plots_on_fixture.py`(16개 전부 mock fixture로).

**`results/` 규약**: `results/raw/<E>/<run_id>/{lineage.json, resolved.json, events.jsonl, summary.json, checks.json, server.log, cmd.txt}`, `results/raw/<E>/latest.txt`, `results/processed/figNN/<run_id>/*.csv`, `figures/figNN.pdf`. `run_id` = `<E>-<system>-<workload>-<config_hash[:8]>-<seed>-<ts>`.

### 2.3 깊이: 맞출 목표 (제안값)

| 항목 | 목표 | 근거 |
| --- | --- | --- |
| 하네스 + 플러그인 + 데이터 + 그림 LoC | 4–6k | vidur 5.4k, loongserve 4.9k, kvcached 6.4k [측정]. 계획서 3,000은 하한 |
| 엔진 patch | 단일 patch 파일 ≤ 300줄, `wc -l`로 보고 | serverlessllm 331, infinigen 258, kvcached 143–249 [측정] |
| CPU 테스트 | `tests/manifests/cpu.txt` ≥ 40 모듈, mock end-to-end 1개 이상, 지표 식별식 전부 | kvcached 62, vllm bench 4.7k LoC [측정] |
| GPU 테스트 | manifest로 분리, E2.1/E2.2 차등 테스트 포함 | serverlessllm `store_test.py` |
| 그림 스크립트 | 16개 + 표 3개 + `numbers.json`, mock fixture로 CI 실행 | GraCE 5개, loongserve 7개가 현재 최고 [측정] |
| AE 문서 | 그림별 (명령, 시간, 결과 경로, 기대 줄), claims C1–C5 → Fig → E → 스크립트 → 기대값 ± 허용 | loongserve + GraCE 합본 |
| lineage | 실행당 `lineage.json` + 행마다 `run_id`, 불일치 resume 거부 | 기존 `lgs1/lineage.py`(6 필드)에 env_hash, gpu_name, slurm_job_id, started_at/finished_at 추가 |

### 2.4 지표 정의와 타임스탬프 점 (SPEC 8.1절에 넣을 식)

타임스탬프 점(하네스가 찍는다, 단일 시계):

| 기호 | 이벤트 | 어디서 | 출처 |
| --- | --- | --- | --- |
| a_r | ARRIVED | 요청 생성기가 정한 도착 시각(사전 계산) | sarathi `arrived_at`, vllm |
| q_r | ADMITTED | 하네스가 `system.run`에 넘기는 순간 | vllm `client_queue_time` |
| s_r | PREFILL_START | 엔진이 첫 batch에 넣은 시각 | sarathi s_r |
| f_r | PREFILL_DONE | 첫 토큰 **생성**(sink 도달 아님) | sarathi `prompt_processing_completed_at` |
| g_{r,i} | TOKEN_GENERATED[i] | 토큰 i 생성 | distserve per-token |
| c^gen_{r,j} | CHUNK_GENERATED[j] | chunk j의 마지막 토큰 생성 | 신규 |
| v_{r,j,w} | VERDICT(j, w) | judge가 (과업, 창 w, chunk j) 판정 반환 | 신규 |
| c^rel_{r,j} | CHUNK_RELEASED[j] / CHUNK_BLOCKED[j] | gate가 chunk j를 sink로 내보낸(또는 버린) 시각 | 신규 |
| v_r | 첫 CHUNK_RELEASED | 첫 chunk가 sink 도달 | 신규(TTFV) |
| c_r | FINISHED | 마지막 chunk release 또는 차단 종료 | sarathi c_r |

규칙: 플러그인은 이벤트만 내고 계산하지 않는다. 시계는 한 종류만(distserve는 perf_counter와 time.time()을 섞어 그림마다 시계가 달랐다). 서버와 클라이언트가 다른 프로세스면 run 시작 시 offset을 한 번 재 `lineage.json`에 기록한다. `--max-concurrency`의 대기는 `q_r − a_r`로 따로 보고한다.

기존 정의 채택(vllm `serve.py:calculate_metrics` [문헌]): `ttft = t(첫 chunk) − st`, `latency = t(마지막 토큰) − st`, `tpot = (latency − ttft)/(output_len − 1)`, 식별식 `latency − ttft == Σ itl`, goodput = 모든 SLO를 만족한 요청 수 / duration. ITL은 SGLang 규칙(chunk 간격을 토큰 수로 균등 분배)을 쓴다 — AG가 다중 토큰 chunk를 내보내기 때문이다.

AG 확장 정의(요청 r, chunk j, 창 w, chunk 토큰 수 n_j, release된 chunk 집합 V_r):
- **TTFT(생성)** = f_r − a_r. B0와 AG의 차이 = ledger·공존 오버헤드만.
- **TTFV** = v_r − a_r. 식별식: TTFV = TTFT_gen + (c^gen_{r,0} − f_r) + lag_{r,0}.
- **Release lag** lag_{r,j} = c^rel_{r,j} − c^gen_{r,j}. N5(hold 없음)는 정의상 0.
- **TPOT(보임)** = (c^rel_{r,last} − v_r) / (Σ_{j∈V_r} n_j − 1). 식별식: c^rel_{r,last} − v_r == Σ ITL_visible.
- **판정 전 노출 토큰** = Σ_j n_j · 1[c^rel_{r,j} < max_w v_{r,j,w}]. B1b/N5의 핵심 값, AG/B1a는 0이어야 함(E2.4 계약).
- **판정 창 수** = |{w : ∃j, VERDICT(j, w)}|, cache hit 포함·제외 둘 다. 상한 = 신뢰하지 않는 창 수.
- **hit@k**: 분모를 항상 적는다(attntrace는 성공 공격만으로 조건부 계산해 분모가 지표마다 달랐다). 행에 `attack_succeeded_b0` 플래그.
- **ASR / FPR / utility**: AgentDojo 식 pair boolean. 엔진·API 오류는 별도 outcome 열(ASR에 합치지 않음). FPR 운영점은 `lgs1/metrics.py:strict_matched_fpr`, bootstrap 95 % CI 1,000회.
- **Goodput(AG)**: SLO_ttfv(L) = s · P95_TTFT(B0, 전용 GPU, 길이 등급 L), SLO_tpot(L) = s · P95_TPOT(B0, L), s ∈ {1.1, 1.25, 1.5, 2.0}. good ⇔ TTFV ≤ SLO_ttfv ∧ TPOT(보임) ≤ SLO_tpot ∧ (정상 요청이면 차단되지 않음). attainment(λ, s)는 요청 가중·토큰 가중 둘 다, goodput(s) = attainment가 90 %(또는 99 %) 선을 지나는 최대 λ(distserve `find_intersection` 보간). 해결된 절대 ms 임계값을 `summary.json`에 적는다. SLO 배수 sweep은 같은 원시 파일을 재평가한다(재실행 없음).
- 출력 길이·토큰 수: 모든 시스템에서 `ignore_eos=True, max_tokens=output_len`, greedy. 출력이 첫 chunk(8토큰)보다 짧은 요청(Mooncake toolagent의 24 %가 ≤ 3토큰 [측정])의 정의를 SPEC에 명시한다.

### 2.5 baseline 이식과 Table 4 프로토콜

조사한 14개 중 baseline의 원 수치를 재현하는 단계를 가진 곳은 0개였다 [측정]. 우리 절차:

1. **원 설정 고정** `configs/table4/<baseline>.yaml`: 원 코드·commit·모델 revision·데이터·하이퍼파라미터를 YAML로 적고 hash. 예 AttnTrace: Llama-3.1-8B-Instruct bf16, greedy, max_output_tokens=100, LongBench ≤ 32k, inject_times=5, seed=12, data_num=100, K=5, q=0.4, B=30 [문헌: `scripts/script_prompt_injection.py`].
2. **원 스크립트로 1회 실행** (GPU 1장, 데이터는 로그인 노드에서 선 캐시). 원 저장소를 수정하지 않는다.
3. **비교**: 원 논문 수치(또는 동봉 결과 파일 `attntrace/results/main/*.json`: n=76, 평균 21.5 s/sample [측정, 원 결과 파일]) 대 재현값. 품질 ±3 pp. 지연은 GPU가 다르면 절대값 대신 B0 대비 배수로 비교(DECISIONS 후보).
4. **플러그인 경로로 같은 설정 재실행**: 우리 `System` wrapper + 같은 YAML. 원 스크립트와 플러그인의 품질 차이 ≤ 1 pp. 이것이 llumnix `test_correctness.py` 식 차등 테스트이고 GPU manifest에 등록한다.
5. **기록**: `results/raw/table4/<baseline>/<run_id>/`에 원 로그, 플러그인 events.jsonl, lineage.json, `table4_row.json`. `plots/table4.py`가 모아 부록 표.

| ID | 원 코드 | wrapper 방식 |
| --- | --- | --- |
| B0 | SGLang 0.5.21 무수정 | 서버 launch(patch 미적용 로그 확인) → SSE chunk를 TOKEN_GENERATED/CHUNK_RELEASED 동시각으로 |
| B1a | Prompt Guard 2 / Qwen3Guard sliding | 플러그인 안에서 모든 창 judge → 전부 통과 시 B0 경로. JUDGE/VERDICT 이벤트를 prefill 전에 |
| B1b | Bedrock/NeMo 방식 | release를 생성 즉시, VERDICT는 뒤에 → exposed_tokens > 0 |
| B1c | 모든 창을 prefill과 병렬 | AG 플러그인의 scheduler=none, selector=all 오버레이 |
| B2 | PIShield(Llama-3.1-8B 13층 probe) | 1차: HF 사본에서 post-hoc activation, "second pass" 시간으로 보고 |
| B3-AttnTrace | `attntrace/src/attribution/attntrace.py`(121줄) | B0로 답 생성 → HF 사본에서 `attribute()` → 상위 K segment를 judge. 100단어 segment ↔ 128토큰 블록 매핑 표를 데이터에 저장. 그대로 import, 재작성 금지 |
| B3-AgentWatcher | 귀속 LLM + monitor | 원본 전체를 플러그인으로, monitor만 떼어 J2로도 등록 |
| B4 | PISanitizer(공개), Rennervate(재구현) | HF 경로에서 입력 변환 후 B0 생성 |
| B5 | MELON | AgentDojo pipeline element로 실행, pair JSON을 하네스가 읽어 ASR/utility 열만 |
| N1–N6 | 우리 코드 | `configs/systems/ag_n1.yaml` 오버레이 |

공통 규칙: 공정성 knob(`kv_capacity_tokens, ignore_eos, max_output_len, max_seq_len, chunked_prefill_size`)는 YAML 필드로 config_hash에 들어간다. 외부 모델을 쓰는 baseline(B2–B4)은 GPU 메모리와 추가 pass 시간을 이벤트 payload로 낸다. AttnTrace(Python 3.10.15·torch 2.5.1 고정)만 **분리 venv + 파일 경계**로 부른다. 나머지는 3.11 주 venv에 함께 둔다(D-001). 포팅 시 함정: attntrace 전역 `set_default_tensor_type`, `wait_for_available_gpu_memory` 300 s 폴링; OPI `CUDA_VISIBLE_DEVICES` 덮어쓰기, 동시성 제한 없는 Popen; agentdojo `LocalLLM`의 기록되지 않는 seed, 오류를 ASR에 합치는 `benchmark.py`.

## 3. 그림·표·실험 전체 목록과 정렬

계획서 8–9절에서 뽑아 두 번 교차 검증한 목록이다(검증기 2개가 80개 지적, 전부 반영 또는 계획서 기준으로 기각). 계획서에 실험 번호가 없지만 코드와 GPU 시간이 드는 작업 5개는 `(E-ID 없음)`으로 추가했다: **W0-grader**(5.3절 7항 LLM grader 대 사람 라벨 200개), **W0-mooncake**(5.5절 길이 분포 집계 + 12.2절 hash_ids 단위), **E0.1-128k**(10.2절 G1의 "T1 128k hit@8 ≥ 0.85, 표본 300"), **W2-e2e-smoke**(G1의 "AgentDojo 1건 + 32k 문서 1건이 gate까지 end-to-end"), **E3.0**(7.1절 Table 4 이식 검증 + 재구현 baseline 속도 확인). 이름은 이 문서가 붙인 것이고 계획서 번호가 아니다.

### 3.1 그림·표·초록 수치 → 원천 실험

`(추론)`이 붙은 것은 9.1절의 "실험" 열에 없고 다른 절에서 끌어온 매핑이다.

| 번호 | 내용 | 원천 실험 |
|---|---|---|
| Fig. 1 | 한 장 요약 Pareto. x=지시문 많은 정상 128k 입력의 문서 FPR(log), y=guard 없는 타깃 대비 TTFV 증가. 점 B1a, B1c, B-full, B3, AG+J1, AG+… | E1.5, E3.1, E4.2 |
| Fig. 2 | 개념도. 두 위협, 지금의 배치, 세 비대칭 | 개념 그림 |
| Fig. 3 | 깊이. x=상대 깊이, y=창 FPR 2×10⁻⁴에서의 TPR. 선: 유해, injection(일반 음성), injection(지시문 많은 음성). 패널 backbone 2개 | E0.3, E1.1 |
| Fig. 4 | 증거. (a) 문서 FPR 대 입력 길이(1k–128k), judge 3종과 독립 가정 곡선. (b) 성공한 공격에서 주입 창의 attention 순위 CDF, 길이 3 | E1.2, E1.3 |
| Fig. 5 | 마감. (a) 길이별 prefill 시간과 scan 시간. (b) chunk 생성 시간과 judge 시간 | E0.4, E1.4, E3.5, E4.6 |
| Fig. 6 | 시스템 구성도(4절). 유해 경로, 타깃 엔진, injection 경로, 스케줄러 | 개념 그림 |
| Fig. 7 | 커널 안 누산. split-KV decode의 분할별 부분 상태(lse)와 창 질량 계산, 추가되는 상태 | 개념 그림 |
| Fig. 8 | 요청 하나의 timeline. 타깃, guard, 보이는 스트림의 세 줄. AG 대 B1a. 2.3의 어느 주장에도 안 붙음(C2는 Fig. 5, 11–13만) → C2에 Fig. 8을 붙이도록 … | E1.4, E4.1 |
| Fig. 9 | 선택 품질. (a) hit@k 대 k, 길이 3. (b) ledger 대 N1–N4, AttnTrace, AgentWatcher 귀속, leave-one-out. (c) 위치와 공격 종류별 | E0.1, E3.3, E0.1-128k |
| Fig. 10 | 길이에 따른 오탐. x=8k–128k, y=같은 ASR에서의 문서 FPR. 모든 창 대 AG, judge별 패널. Table 2의 8k 결과도 여기에 | E3.4, E6.2, E0.2, E3.1 |
| Fig. 11 | 지연 분포. TTFV와 release lag의 CDF(0.99 위 확대), 길이별. 짧은 입력 패널 포함 | E4.2, E4.7 |
| Fig. 12 | goodput 격자. 행=요청률, SLO 배수, GPU 수. 열=workload 3개. 선=시스템 5개. 99 % 선 표시 | E4.3 |
| Fig. 13 | trace 재생. 시간에 따른 attainment, 99 %에 필요한 GPU 수 막대(도달 못 하면 '4+') | E4.4 |
| Fig. 14 | 누산 오버헤드. x=분할 크기와 추적 층 수, y=TPOT 증가(%와 ms), 길이별. 요청당 메모리 | E2.3 |
| Fig. 15 | 적응 공격. (a) 무방어 ASR 대 주입 창 순위. (b) 조각 수 대 탐지율. (c) 판정 폭주 시 release lag | E5.1, E5.2, E5.3 |
| Fig. 16 | (a) 누적 ablation: 품질 막대와 지연 선. (b) k와 chunk 민감도, offline 최선 표시 | E4.5, E6.1 |
| Table 1 | 설계 비교. 선행 10여 개 × 귀속 방법, 추가 모델·pass, 판정 대상, 판정 시점, 보고된 비용. 조건 다른 문헌 수치(SingProbe, RTBAS, CaMeL, AttriGuard, C… | 문헌(2.5·7.3절) |
| Table 2 | 주 결과. 블록=타깃 × 길이(32k, 128k); 8k는 부록과 Fig. 10. 행=하한(B0), 같은 judge의 모든 창, AG, 상한(B-oracle), 기존 시스템. 열=원천 5 × (A… | E3.1 |
| Table 3 | 단계 × 공격. 행=유해 2종, injection 6종, 혼합 문서. 열=scan 단독, 출력 guard 단독, 역추적 judge 단독, 전체, 가장 강한 외부 baseline. 오탐은 단계별 | E3.2 |
| Table 4 (부록) | baseline 이식 검증. 원 논문 수치 대 우리 재현(품질 ±3 pp, 지연 ±20 %). B2–B5 공개 코드 대상 | E3.0 |
| 초록 1 | 지시문이 많은 정상 128k 입력의 문서 FPR: 모든 창 판정 대 AG(같은 judge, 같은 ASR) | E3.1, E3.4, E6.2 |
| 초록 2 | ASR: 무방어 대 AG, 타깃 3종 평균 | E3.1 |
| 초록 3 | 성공한 공격의 hit@8(32k, 128k) | E0.1, E0.1-128k, E3.3 |
| 초록 4 | TTFT(생성) 증가와 release lag의 P95 | E4.2 |
| 초록 5 | SLO 배수 1.25에서의 goodput: guard 없는 타깃(B0), blocking 모든 창(B1a), 두 번째 모델 귀속(B3), AG | E4.3 |
| 초록 6 | 누산의 TPOT 오버헤드와 엔진 변경 LoC | E2.3 |
| 초록 7 | 요청당 판정 창 수의 평균(128k의 창 1,023개 대비) | E3.1, E0.2, E4.7 |

### 3.2 실험 38개 정렬표

정렬 기준은 계획서의 주(W0→W6)이고, 같은 주 안에서는 ID 순이다. "경로"는 참조 구현(HF transformers, 엔진 무수정)으로 되는지 엔진 patch가 필요한지, "기록 재사용"은 5.1절의 "한 번 생성하고 여러 번 평가"가 적용되어 타깃 재생성 없이 기록된 출력·ledger에서 계산되는지다. GPU-h는 계획서 값 [추정]이고 E2.x·E4.1–4.2는 계획서가 묶음 합계만 주므로 첫 항목에 몰아 적었다. 33개 계획 ID의 합은 1,569 GPU-h로 8.9절과 같다 [계산].

| ID | 내용 | 산출 | GPU-h | 주 | 게이트 | 경로 | 기록 재사용 | 축소 순위 | 선행 실험 |
|---|---|---|---|---|---|---|---|---|---|
| E0.1 | 역추적 hit@k 확인(참조 구현) | Fig. 9 초안 | 23 | W0 | G0 | 참조 |  | never-cut(G0) | W0-grader |
| E0.2 | 역추적 창 판정의 오탐 감소 확인 | Fig. 10 초안 | 4 | W0 | G0 | 참조 | 예 | never-cut(G0) | E0.1 |
| E0.3 | 깊이 비대칭 확인(층별 probe) | Fig. 3 초안 | 20 | W0 | G0 | 참조 |  | normal | — |
| E0.4 | 마감 여유 확인(prefill vs scan, chunk vs judge) | Fig. 5 초안 | 4 | W0 | G0 | 참조 |  | normal | — |
| W0-grader | (E-ID 없음) 공격 성공 판정기 검증: LLM grader 대 사람 라벨 200개 | text (판정기 검증 수치, 부록 후보) | 0 | W0 | G0 | 참조 |  | never-cut | — |
| W0-mooncake | (E-ID 없음) Mooncake trace 다운로드·길이 분포 집계·hash_ids 단위 확인 | text (5.5 분포 수치; E4 workload 설계 입력) | 0 | W0 | 없음 | 참조 |  | normal (12.2 확인 항목) | — |
| E1.1 | 깊이 측정 확대(backbone 4, seed 5) | Fig. 3 | 24 | W1 | G1 | 참조 |  | normal | E0.3 |
| E1.2 | 모든 창 판정의 오탐 누적(길이 8단계) | Fig. 4a | 15 | W1 (정상 문서 8단계 생성 조건부; 미충족 시 W2) | G1 | 참조 | 예 | normal | E0.1, E0.2 |
| E2.1 | ledger 정확성(엔진 vs 참조) | text | 30 | W1–W2 | G1 | 엔진 |  | never-cut | E0.1 |
| E0.1-128k | (E-ID 없음) E0.1 확장: T1 128k hit@8, 표본 300 (G1 기준) | Fig. 9 (128k 점), text (9.3 3번의 128k hit@8) | 0 | W2 | G1 | 참조 |  | never-cut(G1) | E0.1, E2.1 |
| E1.3 | attention 집중도 | Fig. 4b | 20 | W2 | G1 | 참조 | 예 | normal | E0.1, E0.1-128k |
| E1.4 | 요청 하나의 시간표(Gantt) | Fig. 5, Fig. 8 | 6 | W2 | G1 | 엔진 |  | normal | E0.4, E2.1, W2-e2e-smoke |
| E1.5 | 지금 방식의 비용(naive 실패 수치 3개) | Fig. 1 (baseline 점), text | 10 | W2 (모든 창 J1·J2), W3 (AgentWatcher 원본: 1… | G1 | 참조 |  | normal | E0.2, E3.0 |
| E2.2 | 출력 동일성 | text | 0 | W2 | G1 | 엔진 |  | never-cut | E2.1 |
| E2.3 | 누산 오버헤드 | Fig. 14 | 0 | W2 (G1: 32k TPOT ≤ +5 %), W5 재측정 (G3: 최… | G1 | 엔진 |  | never-cut | E2.1 |
| W2-e2e-smoke | (E-ID 없음) end-to-end 동작 확인: AgentDojo 1건 + 32k 문서 1건이 gate까지 | text (G1 체크) | 0 | W2 | G1 | 엔진 |  | never-cut(G1) | E2.1, E2.2 |
| E2.4 | release gate 계약 검증 | text | 0 | W3 | G2 | 엔진 |  | normal (10.3 미기재; 3.3 계약 검증이므로 never-cut 후보 [해석]) | W2-e2e-smoke |
| E3.0 | (E-ID 없음) baseline 이식 검증(Table 4) + 재구현 baseline 속도 확인 | Table 4 (부록), text (7.1 속도 확인) | 0 | W3 | G2 | 엔진 |  | never-cut (7.4 최소 집합 B3 AgentWatcher 부분; B2·B4·B5 부분은 normal) | E2.2 |
| E3.1 | 주 결과표 | Table 2, Table 2 부록 8k 블록 (9.2 '8k는 부록'), Fig. 1 (FPR 점), Fig. 10 (8k… | 660 | W3–W4 | G2 | 엔진 | 예 | never-cut | E0.1, E0.2, E2.1, E2.2, E2.3, E0.1-128k, E3.0 |
| E3.3 | 선택 품질 | Fig. 9 | 25 | W3 | G2 | 엔진 | 예 | never-cut | E0.1, E2.1, E0.1-128k, E3.0 |
| E3.2 | 단계 × 공격 표 | Table 3 | 10 | W4 | G2 | 엔진 | 예 | normal | E3.1 |
| E3.4 | 길이에 따른 오탐 | Fig. 10 | 15 | W4 | G2 | 엔진 | 예 | normal | E3.1 |
| E3.5 | 유해 경로 유지 | text (C4) | 10 | W4 | G2 | 엔진 |  | normal | E2.3, E2.4 |
| E4.1 | 요청 하나의 지연 분해 | Fig. 8 | 20 | W4 | G3 | 엔진 |  | normal | E2.3, E2.4, E1.4 |
| E4.2 | 지연 분포 | Fig. 11, Fig. 1 (TTFV 증가) | 0 | W4 | G3 | 엔진 |  | never-cut (N5·B1c는 7.4 최소 집합이고 N5는 여기서만 잼; Fig. 11은 G3 기준) [해석] | E4.1 |
| E4.3 | SLO 아래 goodput 격자 | Fig. 12 | 183 | W5 | G3 | 엔진 |  | never-cut | E4.2 |
| E4.4 | trace 재생 | Fig. 13 | 100 | W5 | G3 | 엔진 |  | normal | E4.3 |
| E4.5 | 누적 ablation | Fig. 16a | 20 | W5 | G3 | 엔진 |  | normal | E4.3 |
| E4.6 | 공존 간섭 | text (C4) | 10 | W5 | G3 | 엔진 |  | normal | E4.1 |
| E4.7 | 이득이 없는 조건 | Fig. 11 (패널), text | 15 | W5 | G3 | 엔진 |  | normal | E4.2 |
| E5.1 | attention 회피 공격 | Fig. 15a | 100 | W5–W6 | G4 | 엔진 |  | never-cut | E3.1, E3.3 |
| E3.6 | 지시문이 아닌 데이터 주입(확장) | text (한계) | 15 | W6 | G4 | 참조 |  | cut-4 | E3.1 |
| E5.2 | 조각내기 공격 | Fig. 15b | 30 | W6 | G4 | 엔진 |  | normal | E3.1 |
| E5.3 | 파이프라인을 아는 공격 | Fig. 15c | 20 | W6 | G4 | 엔진 |  | normal | E4.2, E4.5 |
| E6.1 | 설정 민감도 | Fig. 16b | 20 | W6 | G4 | 엔진 | 예 | normal | E3.1, E3.3, E2.3 |
| E6.2 | detector 교체 | Fig. 10 | 30 | W6 | G4 | 참조 | 예 | normal | E3.1, E3.4 |
| E6.3 | 다른 타깃과 보지 않은 데이터 | text | 90 | W6 | G4 | 엔진 |  | normal (gpt-oss-20b(T4) 부분만 cut-2; LLMail-Inject 부분은 normal) | E3.1 |
| E6.4 | 다른 엔진과 prefix cache | text | 40 | W6 | G4 | 엔진 |  | cut-1 | E2.3, E4.2 |

주별 GPU-h 합계 [계산, 계획서 8.x 값 기준]: W0 51, W1 69, W2 36, W3 685(E3.1 660 포함), W4 55, W5 428, W6 245. W3가 주당 가용 672 GPU-h를 넘으므로 E3.1은 W3–W4에 걸쳐 돌려야 하고(계획서 10.1도 그렇게 적음), 그 뒤 W5의 E4.3·E4.4·E5.1도 재실행 50 %를 더하면 642 GPU-h라 여유가 거의 없다.

### 3.3 계획서에서 비어 있거나 어긋나는 것 (DECISIONS.md 후보)

코드를 쓰기 전에 정해야 하는 것들이다. CLAUDE.md 3.6에 따라 `docs/DECISIONS.md`에 적고 확인을 받는다.

1. **Python 버전.** 결정됨(D-001): AG 코드는 Python 3.11 venv 하나, AttnTrace만 별도 venv. SGLang main과 AgentDojo 0.1.35는 `requires-python >= 3.10`, AttnTrace는 Python 3.10.15 고정이다 [문헌: pyproject·environment.yml].
2. **E4.3–E4.7, E4.4의 AG judge.** 계획서가 judge를 명시하지 않는다. 성능 실험의 기본 judge를 J1(Prompt Guard 2, 작은 분류기)로 통일하고 Fig. 1의 AG+J2 TTFV는 E4.2에서 양쪽 judge를 돌려 채우는 안. E4.5 ablation의 workload도 미명시(W-QA로 가정).
3. **Fig. 5와 Fig. 8의 출처.** 2.3절 C4는 Fig. 5를 E3.5·E4.6의 산출로 적지만 9.1절은 E0.4·E1.4만 든다. E1.4(W2)는 release 시각을 그리므로 엔진의 release gate에 의존해 CLAUDE.md 원칙 2(동기 그림은 엔진 없이)와 어긋난다. 권장: Fig. 5는 E0.4(무수정 엔진)로, Fig. 8은 E4.1(엔진)로 분리하고 E1.4는 Fig. 8의 초안으로 둔다.
4. **E0.1 표본 수와 집계.** "8k와 32k, 공격 600 + 정상 300"이 길이별인지 합계인지, G0의 hit@8이 chunk 3종(첫 8, 첫 32, 전체) 중 어느 것인지 불명(집계는 C로 지정). 128k 표본은 E0.1 100개, G1 300개.
5. **128k 창 수 분모.** 5.1절과 9.3절 7번은 명목 1,023개, T1·T3·T4는 129,024토큰으로 1,007개. `numbers.json`의 분모를 정해야 한다.
6. **TTFV의 sink 정의.** 클라이언트(SSE 수신)에서 찍을지 엔진 output streamer(gate 직후)에서 찍을지. 두 점을 모두 이벤트로 내고 논문에는 어느 것을 쓸지 SPEC에서 정한다(다른 프로세스면 clock offset 보정).
7. **Table 4 지연 재현 ±20 %.** 원 논문 GPU(A100/H100)와 RTX 6000 Ada가 다르면 성립하지 않는다. 절대 지연 대신 B0 대비 배수로 비교하는 규칙으로 바꿀지.
8. **E3.1 Table 2의 B5 MELON.** AgentDojo 원래 길이(7k 이하)에서만 돌아 32k·128k 블록을 채울 수 없다. 본문 수치 또는 부록으로.
9. **E3.3 leave-one-out 부분집합 크기.** 창 63개마다 재생성이 필요해 25 GPU-h 안에 예산이 보이지 않는다.
10. **E0.2·E1.2 GPU-h 재추정.** E1.2 15 GPU-h는 J1만으로도 1,000문서 × 128k ≈ 1.02 M 창 × 92.4 ms ≈ 26 h [계산]라 batch 없이는 부족하고, E0.2 4 GPU-h는 J2(4B LLM judge)를 모든 창에 돌리면 수십 GPU-h가 든다 [추정]. G0 전에 다시 잰다.
11. **gated 모델 fallback.** Llama-3.1(T1) 승인이 늦으면 E0.1은 T2만으로, Prompt Guard 2(J1) 승인이 늦으면 E0.2는 J2만으로 시작한다. G0의 "두 타깃 모두" 조건은 승인 뒤 보충한다.
12. **Mooncake trace 재생 비율.** 실제 trace는 3.4–6.7 req/s, 41–57k input tok/s [측정]로 RTX 6000 Ada 1장 + 8B 모델로는 재생 불가. `arrival_scale`과 20분 구간 추출 규칙을 고정하고 논문에 표기한다. 128k 길이 등급은 실제 trace에 요청이 0개 [측정]라 W-Len(Poisson 합성)만 가능하다.
13. **엔진 변경 LoC의 보고 범위.** patch 파일(Triton 분할 고정 + lse 반출)만 "엔진 변경"으로 세고 shim(ledger·gate·scheduler hook)은 "AG 런타임"으로 셀지. 4.6절 표의 행 정의와 맞춘다.
14. **첫 출력 토큰 처리.** 4.3절: 첫 토큰은 prefill 마지막 위치에서 나와 decode ledger에 잡히지 않는다. E2.1의 참조 대조 범위(첫 토큰 포함 여부)와 E0.1의 참조 기록을 맞춘다. 두 방법 비교는 E6.1.
15. **E2.3 재측정.** G3의 "E2.3 최종값 ≤ +2 %"는 scheduler·gate·judge pool이 모두 붙은 상태의 재측정을 뜻하므로 별도 실행이다. 분할 크기 sweep에서 hit@k를 같이 기록하면 E6.1의 분할 축 재실행을 줄인다.

## 4. 단계별 구축 계획

달력이 아니라 의존 관계로 자른 단계다. 단계마다 코드·데이터 산출물과 테스트로 확인할 수 있는 진입·종료 기준이 있고, 매 단계는 돌아가는 상태(CPU 테스트 green)로 끝난다. 계획서의 게이트(G0 10/18, G1 11/1, G2 11/15, G3 11/22, G4 11/29, 제출 12/8)는 단계의 닻(anchor)이다. 세 가지 관점(뼈대 우선, 위험·임계 경로 우선, 그림 가치 우선)으로 초안을 만들고 심사·합성·누락 검사 2라운드를 거쳤다. 척추는 **위험·임계 경로 우선**이다: 논문을 죽이는 두 위험(E0.1 hit@8 미달, 엔진 ledger 과속 오버헤드)을 가장 적은 코드로 가장 먼저 퇴출시키고, go/no-go를 바꾸지 않는 실험은 부속 단계(S2b·S3b·S4b·S5b)로 평행 배치한다. 뼈대 우선 초안에서는 S1의 모듈·테스트 목록과 접점 4종을, 그림 가치 우선 초안에서는 replay adapter(엔진이 늦어도 품질 열을 채움)와 G3의 "얇은 제출본" 검사를 가져왔다.

### 4.0 모양

```
S0 계약·자산(10/12) ─ S1 걷는 뼈대(10/14) ─┬─ S1E 엔진 spike ─────────┐
                                            ├─ S2  E0 임계 경로 (G0 10/18) ┤
                                            └─ S2b E0.3·E0.4 (부속)       ┘
        ┌─ S3E 엔진 ledger·E2.1–E2.3 (G1 11/1) ─────────┐
        ├─ S3  런타임 조립·smoke·128k·E1.4 (G1 11/1)      ┤
        ├─ S3b 동기 그림 E1.1–E1.3·E1.5 (부속, ≤11/8)     ┤
        └─ S3c baseline 이식·Table 4 (≤11/1)             ┘
        ┌─ S4  scheduler·E2.4·E3.1·E3.3 → Table 2 (G2 11/15) ─┐
        └─ S4b E3.2·E3.4·E3.5 → Table 3·Fig. 5·10 (부속)       ┘
        ┌─ S5  SLO·workload·E4.1–E4.3·E4.7·E2.3-final → Fig. 11·12·1 (G3 11/22) ─┐
        └─ S5b E4.4–E4.6·E4.3 2-GPU → Fig. 13·16a (부속)                        ┘
        ── S6  E5·E6 → Fig. 15·16·10 최종 (G4 11/29)
        ── S7  집필·동결 (freeze 11/30, 초록 12/1, 제출 12/8)
        ── S8a–S8g 선택(10.3절 축소 순서의 역순, 코드는 11/29 전까지만)
```

단계 × 주 GPU-h 행렬 [계산, 계획서 8.x 값 + 추가 작업 추정; 괄호는 재실행 50 % 포함]:

| 주 | 단계와 GPU-h | 합계 |
| --- | --- | --- |
| W0 (10/12–18) | S0 2 · S1E 2 · S2 31 · S2b 20 | 55 (83) |
| W1 (10/19–25) | S1E 2 · S2b 4 · S3E 10 · S3b 24 | 40 (60) |
| W2 (10/26–11/1) | S3E 20 · S3 10 · S3b 35 · S3c 10 | 75 (113) |
| W3 (11/2–8) | S3 6 · S3b 10 · S4 335 | 351 (527) |
| W4 (11/9–15) | S4 355 · S4b 43 | 398 (597) |
| W5 (11/16–22) | S5 208 · S5b 60 · S6 50 | 318 (477) |
| W6 (11/23–29) | S5b 100 · S6 245 (+ S8 선택 60) | 345 (518; 선택 포함 405/608) |
| W7 (11/30–12/6) | S7 재실행 35 | 35 (53) |

합계 1,677 GPU-h(선택 포함) [계산]. 모든 주가 672 GPU-h 이하다. W3–W4는 E3.1(660)이 걸려 있어 다른 GPU 작업을 넣을 여유가 거의 없다 [해석].

### 4.1 걷는 뼈대 (S1, 전부 CPU·pytest, 실제 모델·데이터·엔진 없음)

**정의.** mock 요청 1건이 config → registry → MockSystem → events → hit@k → `$AG_OUT/raw/mock/<run_id>/`(lineage.json 포함) → `plots/fig09.py` → `fig09.pdf` + `numbers.json`(stub)을 60초 안에 통과한다. Python 3.11로 쓴다(D-001). `ag` 패키지는 3.11 venv 하나에 `pip install -e .`로 설치한다.

**순서** (항목마다 PR 하나, 모듈당 100–200줄):

1. `docs/SPEC.md`(S0에서 작성)의 인터페이스를 코드로 옮긴다.
2. `ag/config/{schema.py, hashing.py, overlay.py}` + `configs/exp/skeleton.yaml`. hashing은 `lgs1.lineage.canonical_hash` 재사용.
3. `ag/data/{request.py, window_map.py, mock_lc_inject.py, manifest.py}`. window_map은 `lgs1/lgcore/windows.py`를 b=128로 래핑. 창 수 63 / 255 / 1,007(T1 129,024토큰) / 1,023(T2) [계산].
4. `ag/systems/{base.py, registry.py, mock_system.py}`, `ag/judges/{base.py, mock_judge.py}`. mock은 가상 시계, 합성 attention(주입 창에 질량 집중), pass-through gate, `producer='mock'`.
5. `ag/selector/select.py`: top-k·질량 임계값·집계 A/C·신뢰 창 제외·chunk 합집합. 기록 위에서도, 엔진 shim 안에서도 같은 함수. rollout은 S2.
6. `ag/reference/{record.py, ledger_oracle.py}`: Record v1(npz + index.jsonl, `producer ∈ {mock, reference, sglang-shim, replay}`), oracle은 `tests/conftest.py`의 `tiny`(= `lgs1/lgcore/adapter.tiny_qwen3`, 4층 hidden-64)로 SDPA prefill + eager decode. 실제 모델 경로는 S2.
7. `ag/harness/{events.py, request_state.py, runner.py, writer.py, lineage.py, checks.py, openai_front.py}`. lineage는 **자체 비교기**를 둔다: `lgs1/lineage.py:assert_resume_compatible`은 LINEAGE_KEYS 6개만 비교하므로 [측정: lgs1/lineage.py] AG 필드 전부(patch_sha, engine_version, producer, model_sha.*, env_hash.{harness,engine}, slurm_array_*, hostname)를 비교하고, lgs1에서는 `canonical_hash`·`dir_fingerprint`·`RowsWriter`만 가져온다. openai_front는 OpenAI 호환 `/v1/chat/completions` 골격(backend는 mock)으로 role 경계 → trusted_spans, 도구 호출 완료 경계 규칙을 여기서 정한다(AgentDojo·E0.4·AG gate가 공통으로 쓴다).
8. `ag/metrics/{names.py, quality.py(hit_at_k), per_request.py(TTFT_gen·TPOT·TTFV 식), aggregate.py}`.
9. 결과 디렉터리: `$AG_OUT/raw/mock/<run_id>/{lineage.json, resolved.json, events.jsonl, records/, summary.json, checks.json, cmd.txt, server.log}`. `run_id = <E>-<system>-<workload|data>-<config_hash[:8]>-<seed>-<ts>`; 하위 run은 `<E>-final`, `<E>-gpu`.
10. `plots/{_common.py, fig09.py, abstract_numbers.py}`. `_common`은 `--run-id latest`, mock run의 latest 거부, lineage 불일치 실패, `-dirty` 거부.
11. `tests/fixtures/results_mock/`(run 2개; 행 파일은 `events.jsonl.txt`로 두어 규칙 3.2의 `*.jsonl` 미추적과 충돌하지 않게), `tests/gpu/conftest.py`(skipif CUDA 없음; 기존 `tests/conftest.py` 미수정), `agctl.py` 최소(run / doctor / `--sbatch` 시 `sbatch --test-only` 자동 실행).

**첫 그림**: Fig. 9(a) hit@k 대 k. mock 집중 fixture는 hit@1 ≈ 1.0, noise fixture는 k에 단조 증가, N1 무작위는 8/255 = 3.1 % [계산]. mock으로 그린 그림은 "1차 산출"로 세지 않는다.

**테스트**(`tests/manifests/cpu.txt` ≥ 10개; 누적 목표 S2 ≥ 20, S3 ≥ 35, S5 ≥ 45): test_ag_config, test_events_schema, test_hit_at_k, test_window_map, test_record_roundtrip, test_oracle_tiny, test_selector_aggregations, test_lineage_ag(patch_sha만 다른 resume → exit 2), test_mock_end_to_end(60초), test_plots_on_fixture(fig09), test_latest_rejects_mock, test_openai_front_mock, test_agctl_test_only, test_per_request_formulas.

**stub으로 두는 것**: 실제 Target/Judge, sglang_client(S1E), gate·scheduler(MockSystem pass-through), EngineAdapter의 replay(S3)·sglang(S3E) 구현, 집계 B·rollout(S2), `engine_integration/`(README 한 장), LG 유해 경로 연결(S3).

### 4.2 단계

표기: **담당**은 모두 강지의다(D-002). 원래 collaborator 몫이던 단계는 "(원래 collab)"으로 표시했다. **GPU-h**는 계획서 값 [추정]. **브랜치**는 CLAUDE.md 3.2의 `<주제>/<설명>`.

#### S0 계약과 자산 — 10/12까지, 담당 both, 2 GPU-h

- **목표**: 코드 전에 `docs/SPEC.md`(플러그인 API, Request·Event 12종과 정렬 키, clock 규칙, Record v1 스키마와 E2.1 대조 입도, 지표 식, results·run_id 규약, YAML·overlay·공정성 knob, 가정 목록)와 `docs/DECISIONS.md`(6절의 결정 항목 + 12.2절 확인 사실)를 연다. gated 모델(Llama-3.1, Prompt Guard 2) 신청은 10/10 즉시. 로그인 노드에서 모델·데이터·baseline 코드·trace를 받는다.
- **만드는 것**: SPEC·DECISIONS·`docs/metrics.md`·`docs/results.md`; `pyproject.toml`(ag 패키지, `requires-python >= 3.11`) + `env/requirements-ag311.txt`; `slurm/ag_common.sh`(`env/ag.env`, offline 강제, 3.11 venv 활성화; 기존 `common.sh` 미수정); `ag/data/registry_ag.py` + `slurm/ag_download_login.sh`(`lgs1/download.py` 재사용) → `ag_assets_<MMDD>.json`; CI `.github/workflows/cpu.yml`(Python 3.11), `tests/manifests/{cpu,gpu,integration}.txt` + `tools/check_test_classification.py`, `tools/pr_check.sh`, `.gitignore`(results/, *.jsonl, *.npz, env/*.env, figures/*.pdf; 예외 tests/fixtures/**, tests/data/**); `ag/data/traces/mooncake.py` + `scripts/mooncake_stats.py` + `scripts/tool_wait_stats.py` + `tests/data/mooncake_*_50.jsonl.txt`.
- **첫 작업으로 잴 것**: 기존 LG 테스트 27개 모듈이 CI 조건(3.11, `env/s1.env` 없음)에서 green인지 [측정]. 7개 모듈이 `env/s1.env`·`LG_OUT`·절대 경로를 참조한다 [측정: grep]. 깨지면 CPU gate 명령을 manifest 기반으로 하고 CLAUDE.md 3.4 문구 조정을 결정 항목으로 올린다.
- **실험**: W0-mooncake(길이 분포·hash_ids 단위·tool 대기 분포 집계).
- **종료 기준**: 문서 5종 main; cpu.yml green; 비-optional 자산이 HF_HOME에; gated 상태 매일 기록(10/14까지 미승인이면 T2-only/J2-only로 E0.1 시작); `scripts/py311_bootstrap_login.sh`(3.11 venv 생성·SGLang 설치) + `engine_env_check.sbatch`로 무수정 SGLang 0.5.21이 RTX 6000 Ada + Triton backend에서 기동하는지 lineage.json에 [측정].
- **테스트**: CPU test_trace_loader, test_tool_wait_stats, test_registry_ag, test_no_secrets, test_ag_common_sh, test_check_manifest, test_gitignore_rules, test_pyproject_core_deps.
- **브랜치**: `spec/plugin-api`, `env/ag-assets`, `env/py311-bootstrap`, `env/engine-env-check`, `env/ci-manifests`, `env/pyproject`, `data/mooncake-stats`, `docs/claude-md-python`.

#### S1 걷는 뼈대 + openai_front 골격 — 10/14까지, 담당 me, 0 GPU-h

4.1절 그대로. **종료 기준**: 두 CPU 명령(`python -m pytest tests -x -q`, `pytest $(cat tests/manifests/cpu.txt)`) green(3.11); `python -m ag.harness.run --config skeleton.yaml` → `raw/mock/<run_id>/` 8파일; `plots/fig09.py --run-id <mock>` → PDF, `numbers.json` 키 7개 null; test_mock_end_to_end 60초. **브랜치**: `harness/skeleton`, `harness/reference-record`, `harness/openai-front`, `harness/lineage-ag`, `plots/fig09`.

#### S1E 엔진 spike — G0 10/18까지(b0_engine 10/21, patch v0 10/25), 담당 me(원래 collab), 4 GPU-h

- **목표**: G1의 최대 위험(CUDA graph 아래 `attn_lse` 읽기 가능 여부, Ada에서 Triton decode 기동, 128토큰 분할 비용 방향)을 **1요청 spike**로 G0 시점에 퇴출한다. 계획서 11.1절은 이 위험의 확인 시점을 G1로 적었지만, 그때 알면 fallback(PyTorch 재계산) 재측정이 11/4까지 밀려 S4·S5가 통째로 늦는다 [해석].
- **만드는 것**: `engine_integration/sglang/{README.md, spike/dump_lse.py, ag-sglang-0.5.21.patch(v0), patch.sh, check_patch.sh, remove_patch.sh}`; `ag/systems/sglang_client.py`(서버 launch, `/health` 폴링, SSE 파싱) + `b0_engine.py` + `ag/harness/engine_env.py`; openai_front의 sglang backend; `slurm/ag_engine{,_2gpu,_4gpu}.sbatch`.
- **종료 기준**: T1 32k 1요청의 step마다 `(L_tracked, H, 256)` lse를 Record v1로 저장하고 읽힘 [측정]; eager 대 CUDA graph 둘 다 보고; test_patch_applies 통과, check_patch 멱등; 10/21 `tests/gpu/test_b0_one_request.py` 통과; 32k 1점 TPOT 방향과 release gate 삽입 위치(output streamer 추정 확인)를 DECISIONS의 사실 항목으로 기록.
- **브랜치**: `engine/env-spike`, `engine/sglang-lse-patch`, `harness/sglang-client`, `harness/openai-front-sglang`.

#### S2 E0 임계 경로 — G0 10/18, 담당 me, 31 GPU-h

- **순서**: ⓪ `ag/reference`의 실제 모델 B0 생성 경로(attention 기록 없음)를 첫 PR로 당겨 초안 데이터 200건(T2 8k) pilot 출력을 10/13까지 만든다(≈2 GPU-h [추정]) → ① 사람 라벨 200(≤10/15), LLM grader는 타깃과 다른 Qwen3-8B [추정], 일치율 ≥ 0.95 [제안값] 10/16 → ② 8k·32k 전량 + calib/test 동결 ≤10/15(payloads·background·manifest는 10/14 중간 마감, S2b E0.3의 입력) → ③ 실제 모델 1요청 테스트: T2 8k와 T1 128k(`logits_to_keep=1` 또는 chunked prefill로 전체 logits 33 GB [계산]를 피하고 피크 메모리·시간 기록 → 요청당 시간과 E0.2 GPU-h 재추정) → ④ E0.1 Slurm array(128k 100건은 ③ 통과 뒤) → ⑤ E0.2(J1·J2 offline).
- **만드는 것**: `ag/reference/ledger_oracle.py` 실제 모델 경로 + `cache.py`(content-addressed) + `aggregate.py` + 집계 B; `ag/selector/rollout.py`(4.3절 rollout; shim과 같은 함수) + test_rollout_tiny; `ag/data/success_judge.py` + `scripts/grader_validate.py`; `ag/data/lc_inject/{sources, background, payloads, build, gate, manifest}.py`(gate.json: 정답이 과업 해결 · canary로 주입 가시 · 주입 과업 해결 가능 셋 다 통과한 표본만 manifest에); `ag/data/agentdojo_l.py` + openai_front의 reference_b0 backend(3.11 venv에서 AgentDojo trajectory → pair JSON); `ag/judges/{prompt_guard2, agentwatcher_monitor, offline_runner}.py`(verdict cache는 품질 실험에서 요청마다 reset); `ag/sweep/{expand, submit}.py`(`--array`, `%N` 금지, afterok, 자동 `--test-only`) + `slurm/e0_array.sbatch`; `configs/exp/{e0_1, e0_2}.yaml`; `plots/fig09.py` 확장, `fig10.py` 초안, numbers 3·7; `checks.py` G0 + `agctl g0` → `g0_sheet.md`.
- **실험**: W0-grader, E0.1, E0.2. **그림**: Fig. 9 초안, Fig. 10 초안, 초록 3(32k)·7 초안.
- **종료 기준**: 10/13 pilot 200건(lineage); 10/16 agreement ≥ 0.95; 10/15 `raw/data/<sha>/gate.json` + manifest 동결; 1요청 테스트 2건 통과; `raw/E0.1/` T1·T2 × 8k·32k + T1 128k 100 Record v1, hit@k × 집계 × chunk × 위치 [측정]; `raw/E0.2/` J1·J2 FPR·ROC·판정 창 수 [측정]; fig09·fig10; `g0_sheet.md`(E0.1·E0.2·E0.3·E0.4(pending)·AgentDojo-L 포함 여부·T2-only 여부 열) + `docs/G0_decision.md`.
- **fallback**: AgentDojo-L이 10/15까지 안 되면 4원천으로 G0를 판정하고 10/22 보충. gated 미승인이면 T2-only/J2-only.
- **테스트**: CPU test_ref_ledger_tiny, test_rollout_tiny, test_lc_inject, test_data_gate, test_success_judge, test_offline_runner, test_selector_aggregation_b, test_sweep_expand/submit, test_agentdojo_join, test_openai_front_reference_backend, test_fig10_draft_fixture, test_verdict_cache_reset; GPU test_ref_ledger_real_one, test_ref_ledger_real_128k_one.
- **브랜치**: `e0/b0-pilot`, `e0/grader`, `e0/lc-inject`, `e0/data-gate`, `e0/ref-ledger`, `e0/rollout`, `e0/judge-offline`, `sweep/slurm-array`, `plots/fig10-draft`.

#### S2b G0 참고 측정(부속) — E0.3 10/18, E0.4 10/25, 담당 both, 24 GPU-h

- go/no-go를 바꾸지 않는 E0.3·E0.4. E0.3은 S2의 payloads/background가 10/14에 합류한 뒤 참조만으로. E0.4는 무수정 엔진의 prefill·TPOT(S1E b0_engine, 10/21 뒤)와 LG scan·judge 시간(offline_runner 창 8개 시간 모드)을 `scripts/e04_merge.py`가 `parent_run_ids`로 합친다. 계획서 10.1의 W0 안에 못 들어가므로 일정 이탈을 결정 항목으로 기록하고 G0 sheet에 pending으로 둔다.
- **만드는 것**: `ag/probes/depth_probe.py`(lgs4 학습 코드 재사용 확인), `ag/data/depth_windows.py`, `ag/lg/scan_timing.py`, `configs/exp/{e0_3, e0_4}.yaml`, `plots/fig03.py`·`fig05.py` 초안.
- **종료 기준**: 10/18 `raw/E0.3/` AUROC·TPR·ℓ95 [측정]; 10/25 `raw/E0.4/` merged summary [측정]; 요청당 시간 재추정 갱신.
- **브랜치**: `e0/depth-probe`, `e0/engine-timing`, `plots/fig03-fig05-draft`.

#### S3E 엔진 트랙 — G1 11/1(32k 1점 10/27), 담당 me(원래 collab), 30 GPU-h

- **목표**: patch v0 위에 `LedgerSink` shim을 올려 Record v1을 같은 writer로 쓰고, S2의 oracle 기록과 `diff.py`로 대조해 G1을 통과한다. E2.3은 분할 128/256/512/1,024마다 LC-Inject 공격 부분집합(8k 500·32k 100·128k 20, E2.1과 동일)을 Record v1 + success_judge 결과와 함께 기록해 E6.1의 분할 축을 재실행 없이 계산한다. TPOT 식은 S1의 `per_request.py`를 쓴다(S3를 기다리지 않음).
- **만드는 것**: patch v1 + `shim/{patch_manager, ledger(rollout.py 호출; TP>1 all-gather), hooks}.py`; `ag/systems/adapters/sglang_adapter.py` + `ag_ledger_only.py`; `ag/reference/diff.py`(선검사 → teacher-forcing; 토큰별 head 평균과 chunk per-head 둘 다); `ag/bench/tpot_overhead.py`(ledger_export off|sync|async; warm-up 규칙); `configs/targets/t3.yaml` + `tests/gpu/test_tp2_ledger_one_request.py`; `scripts/count_engine_loc.sh`; `configs/exp/{e2_1, e2_2a, e2_2b, e2_3}.yaml`; `plots/fig14.py`.
- **실험**: E2.1, E2.2, E2.3(1차). **그림**: Fig. 14 초안, 초록 6 초안.
- **종료 기준**: `raw/E2.1/` diff_report max_abs_err ≤ 1e-3, top8 ≥ 0.99 [측정]; `raw/E2.2/` (a) 100 %, (b) ≥ 99.9 % [측정]; 10/27 32k 1점, `raw/E2.3/` off 32k ≤ +5 % [측정] + 분할 4종 Record; test_tp2_ledger_one_request 통과(불가 시 T3를 TP-2 없이 돌릴지 결정 항목).
- **테스트**: CPU test_patch_applies(v1), test_ledger_math, test_tp_allgather_math, test_selector_record_vs_live, test_diff_report, test_shim_patch_manager, test_ledger_export_modes, test_fig14_fixture; GPU test_e21_ledger_diff_small, test_e22_token_identity_small, test_b0_no_patch, test_e23_smoke, test_tp2_ledger_one_request.
- **브랜치**: `engine/ledger-shim`, `engine/adapter-stream`, `engine/tp2-allgather`, `engine/diff-e21-e22`, `engine/e23-overhead`, `plots/fig14`.
- **위험**: TPOT 초과 → 토큰 n개마다 누산하는 sampling + hit@k 손실 보고; 엔진 venv의 torch·SGLang 버전 고정 → `env_hash`로 기록.

#### S3 런타임 조립 — G1 11/1(128k 동결 10/28, E1.4 11/5), 담당 me, 16 GPU-h

- **목표**: offline runner를 judge pool로 흡수하고 release gate를 in-process hook으로 만들어 AG 플러그인을 완성한다. AG 플러그인의 플래그 5종 `block_before_generation`(B1a), `release_before_verdict`(B1b), `selector ∈ {topk, all, oracle, random, truncate, last_layer}`, `scheduler ∈ {none, fifo, deadline}`, `judge`의 조합이 B1a/B1b/B1c/B-full/B-oracle/N1/N4/N5/N6 overlay다. W2-e2e-smoke는 **엔진 ledger → selector → judge pool → gate → sink** 경로가 필수다. 엔진 gate hook이 늦으면 하네스 쪽 gate proxy(openai_front의 ag_gate backend)로 대신하되 ledger는 sglang_adapter에서 와야 한다. 엔진 ledger가 11/1까지 합류하지 않으면 G1 미통과로 보고하고 10.2절의 11/4 재측정 규칙을 따른다(기준 완화 없음).
- **만드는 것**: `ag/runtime/{judge_pool, verdict_cache, release_gate, block_isolate, scheduler(FIFO 골격)}.py`; `engine_integration/sglang/shim/gate_hook.py`; `ag/lg/path.py` + `ag/judges/output_guard.py`; `ag/systems/ag.py` + `configs/systems/{ag, ag_n1..n6, b1a, b1b, b1c, b_full, b_oracle}.yaml` + `adapters/replay_adapter.py`(offline 전용) + `mock_fault_judge.py`; `per_request.py` 확장(release_lag, TPOT_visible, exposed_tokens, judged_windows) + `plots/fig11.py`(mock) + `tests/test_metrics_identity.py`; LC-Inject 128k 전량 동결 ≤10/28(T1 1,007창·T2 1,023창) + `scripts/e01_128k.py`; `ag/data/workloads/wlen_min.py`, `ag/harness/gantt.py` + `plots/fig08.py` 초안; `agctl smoke/g1`.
- **실험**: W2-e2e-smoke, E0.1-128k, E1.4. **그림**: Fig. 8 초안, Fig. 9(128k 점), Fig. 11(mock 시계·식별식), 초록 3(128k).
- **종료 기준**: test_gate_contract_mock(10⁵ chunk, 판정 전 방출 0, P99 ≤ 1 ms 가상 시계; E2.4의 CPU 계약판); 10/28 128k manifest + gate.json; G1: `raw/W2-e2e-smoke/` AgentDojo 1건 + 32k 1건, `producer=sglang-shim`, checks ok [측정]; G1: `raw/E0.1-128k/` T1 128k n=300 hit@8 ≥ 0.85 [제안값] [측정]; 11/5 `raw/E1.4/` gantt + fig08 초안; cpu manifest ≥ 35.
- **테스트**: CPU test_judge_pool_mock, test_verdict_cache_key, test_release_gate, test_gate_contract_mock, test_block_isolate, test_gantt, test_ag_overlays(플래그별 이벤트 순서), test_replay_adapter, test_lg_path_import, test_wlen_min, test_metrics_identity, test_openai_front_gate_proxy, test_fig08/fig11_fixture; GPU test_e2e_smoke_one, test_agentdojo_one_task, test_lg_path_coexist, test_adapter_equivalence.
- **브랜치**: `runtime/judge-pool`, `runtime/release-gate`, `runtime/ag-system`, `runtime/lg-link`, `harness/openai-front-gate`, `systems/replay-adapter`, `runtime/e2e-smoke`, `data/lc-inject-128k`, `e0/128k-300`, `engine/gate-hook`, `plots/fig08-fig11`.

#### S3b 동기 그림 보강(부속) — ≤11/8, 담당 me, 69 GPU-h

- 2절 관찰 3개와 Fig. 1 baseline 점을 S2 기록·offline runner 위에서 만든다. E1.2의 8단계 정상 1,000문서와 E1.1의 음성 창 20만은 ≤10/26 생성(gate.json). E1.2 J1 1,000 × 8단계는 약 26 h [계산]라 rc=3 checkpoint/requeue. E1.5의 AgentWatcher 원본 행은 S3c의 E3.0(a) 뒤.
- **만드는 것**: `lc_inject/build.py` 확장; depth_probe 확장(backbone 4, seed 5); `ag/judges/piguard.py`(J3) + offline_runner 8단계 모드; `ag/reference/concentration.py`, `ag/harness/synthetic_cost.py`; `configs/exp/{e1_1, e1_2, e1_3, e1_5}.yaml`; `plots/fig03` 최종, `fig04`, `fig01` 초안.
- **실험**: E1.1, E1.2, E1.3, E1.5. **그림**: Fig. 3·4 최종, Fig. 1 초안.
- **브랜치**: `e1/depth-scale`, `e1/fpr-accumulation`, `e1/attention-concentration`, `e1/naive-cost`, `judges/piguard`, `plots/fig01-fig04`.

#### S3c baseline 이식(엔진 patch 불필요) — ≤11/1(B2·B4·B5는 11/8), 담당 me, 10 GPU-h

- 원칙 6: 원 설정에서 원 수치를 재현한다(E3.0(a), Table 4, 2.5절 프로토콜). 재구현 B1a/B1c가 원본보다 느리지 않음(E3.0(b))은 S3·S3E 합류 뒤 S4에서. E3.1 1차의 B1a·B3·N3 셀은 Table 4 행 통과에 `afterok`.
- **만드는 것**: `ag/systems/{b2_pishield, b3_attntrace, b3_agentwatcher, b4_pisanitizer, b4_rennervate, b5_melon, n2_guard_attention, n3_shallow_probe}.py`(venv별 파일/HTTP 경계); `configs/table4/*.yaml` + `scripts/table4_run.py` + `plots/table4.py`; E1.5 AgentWatcher 원본 행.
- **종료 기준**: 11/1 `raw/table4/<baseline>/<run_id>/table4_row.json` B1a(J1)·B3 두 종 품질 ±3 pp·지연 ±20 %(또는 B0 대비 배수) [측정]; test_table4_port_equiv ≤ 1 pp(GPU).
- **브랜치**: `baseline/attntrace`, `baseline/agentwatcher`, `baseline/pishield`, `baseline/pisanitizer-rennervate`, `baseline/melon`, `baseline/table4`, `plots/table4`.

#### S4 Table 2 임계 경로 — G2 11/15, 담당 me, 690 GPU-h

- **순서**: ① 11/1–2 **1차-A**: 기록에서 offline 재계산 가능한 행만(B0·B-oracle·N1·N6·B1a/B-full offline; FIFO scheduler 골격, 조건 간 서버 재기동 reset, 품질 열만; B1a 셀은 Table 4 afterok) → ② 11/3 **1차-B**(B3·N3; afterok Table 4) → ③ scheduler·`reset.py` 완성 ≤11/5 → ④ 11/8 E2.4 GPU판(`run_id E2.4-gpu`) + E3.0(b) 속도 확인 → ⑤ 11/5 이후 AG+J1/J2·N5·B1a/B-full의 **live 행**(AgentDojo 셀은 openai_front ag_gate 뒤 live run — record-once 예외) → ⑥ 2차 array(T2 8k·32k·128k, T3 8k·32k TP-2, B2·B4·B5, T1 8k) → ⑦ E3.3 최종은 ⑥의 T2 128k 기록 뒤. E3.1 660 = 1차 330 + 2차 330 [계산].
- **만드는 것**: `ag/runtime/scheduler.py` 완성, `ag/harness/reset.py`; `configs/exp/e2_4_gpu.yaml` + checks; `scripts/b1_speed_check.py`; `ag/analysis/{offline_recompute, leave_one_out}.py`; `configs/exp/{e3_1, e3_3}.yaml` + `slurm/e3_1_array.sbatch`(afterok 체인, rc=3 requeue, array id lineage) + `e3_1_t3.sbatch`(`--gres=gpu:2`); `plots/table2.py`, `fig09.py` 최종, numbers 1·2·3·7, `agctl g2`.
- **실험**: E2.4, E3.1, E3.3. **그림**: Table 2(T3 128k는 S8b), Fig. 9 최종, 초록 2·3 최종, 초록 1 초안.
- **종료 기준**: 1차-A lineage started_at ≤ 11/2; 11/8 `raw/E2.4/` 판정 전 방출 0, P99 ≤ 1 ms [측정], 재구현 B1a/B1c 속도 확인 [측정]; `raw/E3.1/` T1 세 길이·T2 8k·32k × 시스템 ≥ 10 × 원천 5, `table2`, `g2_sheet` + `docs/G2_decision.md`; T3 8k·32k 최소 집합 → numbers 2.
- **브랜치**: `runtime/scheduler`, `runtime/reset`, `baseline/b1-overlays`, `baseline/oracle-nvariants-loo`, `e3/main-table`, `e3/t3-8k-32k`, `e3/selection-quality`, `e2/gate-gpu`, `plots/table2`.

#### S4b 보안·품질 보강(부속) — G2 11/15(늦어도 11/18), 담당 me, 43 GPU-h

- 16k/64k 부분집합을 동결(≤11/8)하고 기록 run(T1, n=300 × 2, ≈8 GPU-h [추정])을 먼저 돌린 뒤 E3.4를 그 run_id 위에서 재계산한다. E3.5 scan 점으로 Fig. 5 최종. E3.2 혼합 문서도 gate.json.
- **만드는 것**: `lc_inject/mixed.py` + 16k/64k 부분집합; `configs/exp/{e3_2, e3_4, e3_5}.yaml`; `plots/table3.py`, `fig10.py` 갱신, `fig05.py` 최종; `ag/bench/lg_path_bench.py`.
- **실험**: E3.2, E3.4, E3.5. **그림**: Table 3, Fig. 5 최종, Fig. 10 갱신, 초록 1 초안.
- **브랜치**: `e3/stage-attack-table`, `e3/fpr-vs-length`, `e3/harm-path`, `data/lc-inject-16k-64k`.

#### S5 성능 임계 경로 — G3 11/22(SLO run 11/15), 담당 me, 208 GPU-h

- **목표**: 첫 job으로 B0 전용 GPU의 P95 TTFT·TPOT(3회, warm-up 제외)를 `configs/slo.yaml`에 적는다(runner와 plots가 같이 읽는 유일한 SLO 원천). E2.3-final은 같은 YAML, `ledger_export`는 E4에서 쓰는 모드, scheduler·gate·judge pool on, 별도 config_hash → `run_id E2.3-final`; off 값은 Fig. 14 하한선만. E4.3은 1-GPU 격자만(2-GPU 행은 S5b 필수, 4-GPU는 S8a). G3에 "얇은 제출본"(Fig. 1, Table 2, Fig. 9·10·11·12·14) 검사를 통과시킨다.
- **만드는 것**: `scripts/slo_baseline.py` + sbatch → `configs/slo.yaml`; `configs/systems/ag_dedicated.yaml` + `info().placement` + `tests/gpu/test_multi_gpu_placement.py`; `ag/data/workloads/{wlen, wqa, wagent}.py` + `arrivals.py` + `trace_replayer.py` + `warmup.py`; `ag/metrics/goodput.py` + `ag/sweep/capacity.py` + 공정성 knob 검사; `scripts/e23_final.py`; `configs/exp/{e4_1, e4_2, e4_3, e4_7, e2_3_final}.yaml`; `plots/fig11` 최종, `fig12`, `fig01` 최종, `fig08` 최종, `fig14` 최종, numbers 4·5·6·7; `agctl g3`, `scripts/check_thin_submission.py`.
- **실험**: E4.1, E4.2(3회), E4.3(1-GPU), E4.7. **그림**: Fig. 11·12·1·8·14 최종, 초록 4·5·6·7 최종.
- **종료 기준**: `configs/slo.yaml` [측정] + source_run_id; `raw/E4.2/` 9 시스템 × 3회; `raw/E4.3/` 1-GPU 격자, fig12, `g3_sheet` + `G3_decision`; `raw/E2.3-final/` E4 모드 TPOT ≤ +2 % [제안값] [측정]; check_thin_submission 통과; cpu manifest ≥ 45.
- **브랜치**: `perf/slo-baseline`, `perf/dedicated-placement`, `perf/workloads`, `perf/trace-replay`, `perf/goodput-slo`, `perf/e4-sweeps`, `engine/e23-final`, `plots/fig11-fig12-fig01`, `plots/fig08-fig14-final`.

#### S5b 성능 보강(부속) — G3 11/22(늦어도 11/25), 담당 me, 160 GPU-h

- E4.3의 gpus=2 행은 6.3절의 기본 구성이라 필수(10.3절 cut-7은 4-GPU만)이며 E4.6과 같은 `--gres=gpu:2` 창에서 돈다. E4.4는 1·2·4장 run(S1E 템플릿), rc=3 requeue.
- **만드는 것**: `ag/sweep/trace_scale.py`, `configs/systems/ag_abl_{1..6}.yaml`, `configs/exp/{e4_3_gpu2, e4_4, e4_5, e4_6}.yaml`, `slurm/e4_6_dedicated.sbatch`, `plots/fig13.py`, `fig16.py`(a).
- **실험**: E4.4, E4.5, E4.6. **그림**: Fig. 13(W-MT 열은 S8c), Fig. 16(a), Fig. 12 2-GPU 행.
- **브랜치**: `perf/trace-scale`, `perf/ablation-chain`, `perf/coexist`, `perf/e43-gpu2`, `plots/fig13-fig16a`.

#### S6 적응 공격·민감도·일반성 — G4 11/29(E5.1 11/18 선행), 담당 me, 295 GPU-h

- E6.1의 분할 축은 S3E의 E2.3 Record 재합산, rollout 유무는 `rollout.py` 재계산, release lag·chunk 5종·선택 규칙 축만 live 엔진 부분집합(32k 50문서, ≈5 GPU-h [추정]). LLMail·E3.6 데이터도 gate.json.
- **만드는 것**: `ag/attacks/{gcg_joint, llm_attacker(Qwen3-8B), decoy, fragment, pipeline_aware}.py`; `ag/reference/sensitivity.py`; `ag/judges/datasentinel.py`(J4) + `scripts/e62_detector_swap.py`; `ag/data/llmail_inject.py`; `configs/exp/{e5_1, e5_2, e5_3, e6_1, e6_2, e6_3}.yaml`; `plots/fig15`, `fig16`(b), `fig10` 최종; `agctl g4` + `check_figures.py`; test_plots_on_fixture 16개 누적 확인.
- **실험**: E5.1, E5.2, E5.3, E6.1, E6.2, E6.3(LLMail; T4는 S8f). **그림**: Fig. 15·16·10 최종, 초록 1 최종.
- **브랜치**: `attack/gcg-joint`, `attack/llm-attacker`, `attack/fragment`, `attack/pipeline-aware`, `e6/sensitivity`, `e6/detector-swap`, `e6/llmail`, `plots/fig15-fig16b`.

#### S7 집필·동결 — cutoff 11/29 → freeze 11/30 → 초록 12/1 → 제출 12/8, 담당 both, 35 GPU-h(재실행)

- 측정이 아닌 산출물을 닫는다. `docs/AE.md` = claims C1–C5 → Fig → E → 스크립트 → 기대값 ± 허용 표 + 그림별 human/machine 시간·HW 표 + 범위 밖 목록; test_ae_commands가 표의 명령을 fixture로 전부 실행. `plots/provenance.py` → `PROVENANCE.md` + `manifest.json`(그림마다 1차 단계·run_id·code_commit·config_hash). `scripts/freeze_check.py`가 freeze 뒤 `ag/`·`engine_integration/` 변경을 거부, `anonymize_check.py`가 hostname·사용자 ID·절대 경로·run_id 노출을 막는다. 개념 그림(Fig. 2·6·7)과 Table 1(`configs/literature.yaml` + `plots/table1.py`)은 여기.
- **브랜치**: `docs/ae`, `plots/table1`, `docs/freeze`, `fix/<issue>`(freeze 뒤에는 plots/·docs/ 렌더링만).

### 4.3 선택 단계 (10.3절 축소 순서의 역순, 코드는 11/29 전까지만)

| 단계 | 내용 | 조건 | GPU-h |
| --- | --- | --- | --- |
| S8a | E4.3 4-GPU 행만 실행(Fig. 12) | G3 통과, 11/27 이후 4장 동시에 빔 | E4.3 183 안 |
| S8b | T3 128k 블록(Table 2) | G2 통과, 2장 빔 | E3.1 660 안 |
| S8c | W-MT prefix cache 결합(verdict cache × KV block 해시, Fig. 13 W-MT 열) | S5b Fig. 13 뒤 | 0 |
| S8d | E3.6 데이터 주입 대조군(참조 경로) | S6 중 여유 | 15 |
| S8e | block_isolate regenerate(E2.4 regen 모드) | S4·S5 완료 | 5 |
| S8f | T4 gpt-oss-20b(E6.3 T4 부분) | S6 LLMail 완료 | E6.3 90 안 |
| S8g | vLLM 0.31.0 + FlashInfer wrapper, E6.4(1인 수행에서는 하지 않는 것으로 봄) | S8c 완료, FlashInfer·CUDA graph 호환 확인 | 40 |

### 4.4 할당표

실험 38개 → 1차 단계(코드와 데이터가 처음 실제로 돌아가는 단계; mock은 1차가 아니다):

| 실험 | 1차 단계 | 비고 |
| --- | --- | --- |
| W0-mooncake | S0 | mooncake_stats + tool_wait_stats |
| W0-grader | S2 | B0 pilot 200건(10/13) → 라벨 ≤10/15; grader = Qwen3-8B |
| E0.1 | S2 | G0; 128k 100건은 1요청 테스트 뒤 |
| E0.2 | S2 | G0 |
| E0.3 | S2b | 10/18 |
| E0.4 | S2b | 10/25; G0 sheet에 pending |
| E2.1 | S3E | G1 |
| E2.2 | S3E | G1; (b) 128k는 전량 동결 또는 n=100 |
| E2.3 | S3E | 1차 G1(off, 분할별 Record); 2차 실행은 S5의 run_id E2.3-final |
| W2-e2e-smoke | S3 | 엔진 ledger 경로 필수 |
| E0.1-128k | S3 | G1; 128k 동결 ≤10/28 |
| E1.4 | S3 | 11/5 체크포인트(G1 항목 아님) |
| E1.1 | S3b | ≤11/8 |
| E1.2 | S3b | 8단계 ≤10/26 생성, rc=3 |
| E1.3 | S3b |  |
| E1.5 | S3b | AgentWatcher 행은 S3c 뒤 |
| E3.0 | S3c | (a) Table 4 ≤11/1; (b) 속도 확인은 S4 ≤11/8 |
| E2.4 | S4 | 1차 = GPU판(11/8); S3는 CPU 계약 테스트만 |
| E3.1 | S4 | 1차-A offline 행(11/1–2), 1차-B 11/3, live 행 11/5 이후, 2차; T3 128k는 S8b |
| E3.3 | S4 | E3.1 2차(T2 128k) 뒤 |
| E3.2 | S4b |  |
| E3.4 | S4b | 16k/64k 기록 run 뒤 |
| E3.5 | S4b | E2.4 GPU 뒤 |
| E4.1 | S5 |  |
| E4.2 | S5 | 3회 |
| E4.3 | S5 | 1-GPU; 2-GPU 행 S5b(필수); 4-GPU S8a |
| E4.7 | S5 |  |
| E4.4 | S5b |  |
| E4.5 | S5b |  |
| E4.6 | S5b |  |
| E5.1 | S6 | 11/18 선행 |
| E5.2 | S6 |  |
| E5.3 | S6 |  |
| E6.1 | S6 | 분할 축 S3E 기록, rollout 재계산, live 부분집합 |
| E6.2 | S6 |  |
| E6.3 | S6 | LLMail; T4는 S8f |
| E3.6 | S8d(선택) |  |
| E6.4 | S8g(선택) |  |

그림 16 · 표 4 · 초록 수치 7 → 1차 단계 / 최종 단계:

| 항목 | 1차 | 최종 | 경로 |
| --- | --- | --- | --- |
| Fig. 1 | S3b | S5 | E1.5 점 → E4.2 × E3.1 |
| Fig. 2 | S7 | S7 | 개념 |
| Fig. 3 | S2b | S3b | E0.3 → E1.1 |
| Fig. 4 | S3b | S3b |  |
| Fig. 5 | S2b | S4b | E0.4 → E3.5 scan 점 |
| Fig. 6 | S7 | S7 | 개념 |
| Fig. 7 | S7 | S7 | 개념 |
| Fig. 8 | S3 | S5 | E1.4 → E4.1 |
| Fig. 9 | S2 | S4 | E0.1 → 128k(S3) → E3.3 |
| Fig. 10 | S2 | S6 | E0.2 → E3.4 → J3/J4 |
| Fig. 11 | S3 | S5 | mock 시계·식별식 → E4.2 + E4.7 |
| Fig. 12 | S5 | S5 | 1-GPU; 2-GPU S5b, 4-GPU S8a |
| Fig. 13 | S5b | S5b | W-MT 열 S8c |
| Fig. 14 | S3E | S5 | E2.3 → E2.3-final |
| Fig. 15 | S6 | S6 |  |
| Fig. 16 | S5b | S6 | (a) → (b) |
| Table 1 | S7 | S7 | 문헌 |
| Table 2 | S4 | S4 | T3 128k S8b |
| Table 3 | S4b | S4b |  |
| Table 4 | S3c | S3c |  |
| 초록 1 | S4 | S6 | E3.1 → E3.4 → E6.2 |
| 초록 2 | S4 | S4 |  |
| 초록 3 | S2 | S4 | 32k → 128k(S3) → E3.3 |
| 초록 4 | S5 | S5 |  |
| 초록 5 | S5 | S5 |  |
| 초록 6 | S3E | S5 | E4 모드 값 |
| 초록 7 | S2 | S5 |  |

### 4.5 엔진 트랙과 하네스 트랙의 접점

두 트랙은 아래 접점에서만 만난다. 서명은 `docs/SPEC.md`에 있고, 접점마다 CPU mock 테스트와 GPU 차등 테스트가 짝이다. 하네스 mock이 먼저 green이 되고(S1: 1·2·4·5, S3: 3), 엔진 PR이 GPU로 통과한다.

**환경 경계** (D-001): `ag` 패키지는 엔진과 같은 3.11 venv에 설치되므로 shim이 `ag.selector`를 직접 import한다. 프로세스 간은 파일(Record npz, pair JSON, events.jsonl)·HTTP(openai_front, SGLang)만. AttnTrace만 별도 venv의 worker로 부른다(SPEC 9.2절). clock: 하네스·엔진·judge pool은 한 Slurm job/한 노드(CLOCK_MONOTONIC 공통)이고 lineage hostname 동일을 검사; 무수정 SGLang(B0)은 서버 타임스탬프가 없어 TTFV는 클라이언트 수신 단일점.

1. **Record schema v1** (`ag/reference/record.py`): npz{output_ids, block_mass fp16[T_out × L_trace × n_blocks] head 평균, chunk_head_mass fp16[3 × L_trace × H × n_blocks], block_trusted, layer_ids, head_weighting, lse_total, t_ns} + index.jsonl(producer, first_token_policy, schema_version, events_ref, success_judge_ref). 참조(S2)·shim(S3E)·tpot_overhead(S3E)·16k/64k 기록(S4b)·replay(S3)가 같은 `write_record()`. E2.1 대조: 토큰별 head 평균과 chunk per-head 둘 다 max_abs_err ≤ 1e-3·top8 ≥ 99 %; fp16 언더플로는 log 저장 기준. 테스트: test_record_roundtrip, test_oracle_tiny(CPU); test_e21_ledger_diff_small, test_e22_token_identity_small(GPU).
2. **LedgerSink / selector·rollout + EngineAdapter** (`shim/ledger.py` → `ag/selector/{select, rollout}.py`): shim은 `LedgerSink.on_decode_step(req_id, layer_id, lse_split[B,H,S], lse_total[B,H], split_tokens=128)`만 호출하고(TP>1은 all-gather 뒤 1회), 수식·rollout·chunk 누산·top-k는 `ag/` 순수 Python(복제본 금지). `ledger_export off|sync|async`는 config_hash 분리. EngineAdapter 구현 셋: mock(S1), replay(S3, offline 전용), sglang(S3E). 테스트: test_ledger_math, test_selector_record_vs_live, test_replay_adapter, test_ledger_export_modes(CPU); test_adapter_equivalence(GPU).
3. **ReleaseGate hook + openai_front** (`shim/gate_hook.py` → `ag/runtime/release_gate.py`): shim은 `gate.on_tokens/poll`만. 도구 호출 완료 경계 규칙은 S1 골격에서 정하고 S1E가 확인. gate hook이 늦는 기간엔 openai_front의 ag_gate backend가 같은 API로 gate를 대신하되 ledger는 sglang_adapter에서 온다. 테스트: test_gate_contract_mock, test_ag_overlays, test_openai_front_mock, test_metrics_identity(CPU); test_e2e_smoke_one, test_agentdojo_one_task, test_e24_load(GPU).
4. **System.info() / patch 상태 / 공정성 knob**: `info() → {engine_version, attn_backend, patch_applied, patch_sha, shim_active, model_sha, venv_env_hash, placement, gpu_mem_gib}`; checks.json의 patch_state; 공통 knob는 config_hash에 포함하고 B0·AG 동일 검사. 테스트: test_patch_applies, test_fairness_knobs(CPU); test_b0_no_patch, test_multi_gpu_placement(GPU).
5. **E2.3 오버헤드 계약** (`ag/bench/tpot_overhead.py` → Fig. 14): TPOT 식은 S1의 `per_request.py`. S3E 초기값 G1 ≤ +5 %(export off, 분할 4종 Record + 공격 부분집합), S5 최종값 G3 ≤ +2 %(E4에서 쓰는 export 모드, scheduler·gate·judge pool on; run_id E2.3-final). 초록 6 = E4 모드 값 + `count_engine_loc.sh`.
6. **E0.4 합성**: `scripts/e04_merge.py`가 엔진 run_id와 scan·judge run_id를 `parent_run_ids`·clock offset과 함께 합친다.

**엔진이 늦을 때**: 하네스는 참조 기록으로 E3.3·B-oracle·N-variants·E6.1 재합산과 E3.1 1차-A(offline 행)만 먼저 채우고, 전량·지연 열·smoke·E1.4·Fig. 8·11·12·14는 엔진 ledger(G1)를 기다린다. G1 미달이면 10.2절 fallback(11/4 재측정)을 보고하고 기준은 바꾸지 않는다.

## 5. 브랜치·PR·태그 전략

**규칙(CLAUDE.md 3.2)**: main 직접 push 금지, 작업마다 `<주제>/<짧은-설명>` 브랜치 + PR. main 불변식은 두 CPU 명령 green — `python -m pytest tests -x -q`(`tests/gpu/conftest.py`가 CUDA 없으면 skip; 기존 `tests/conftest.py` 미수정)와 `pytest $(cat tests/manifests/cpu.txt)`. `tools/pr_check.sh`가 두 명령 + `<<<<<<<`·토큰·`/home/` grep. CI `.github/workflows/cpu.yml`은 Python 3.11. GPU 차등 테스트는 CI 밖: Slurm 결과(diff_report, run_id, lineage)를 PR 본문에 첨부. 미추적: `results/`, `*.jsonl`, `*.npz`, 체크포인트, `env/*.env`, `figures/*.pdf`(논문 저장소에서만 추적; plots 출력은 `$AG_OUT/figures`); 예외 `tests/fixtures/**`(행 파일은 `.jsonl.txt`), `tests/data/**`. PR당 모듈 1–2개 + CPU 테스트(≤ 500줄). PR 설명에 재현 명령·run_id·수치(출처 표기).

**단계별 브랜치 접두사**: S0 `spec/`, `env/`, `data/`, `docs/` · S1 `harness/`, `plots/` · S1E·S3E·S8g `engine/` · S2·S2b `e0/`, `sweep/` · S3 `runtime/`, `systems/`, `data/` · S3b `e1/`, `judges/` · S3c `baseline/` · S4·S4b `runtime/`, `e2/`, `e3/` · S5·S5b `perf/` · S6 `attack/`, `e6/` · S7 `docs/`, `fix/`.

**게이트 태그**: `g0-2026-10-18`, `g1-2026-11-01`, `g2-2026-11-15`, `g3-2026-11-22`, `g4-2026-11-29`, `freeze-2026-11-30`, `submit-2026-12-08`. 태그 메시지에 gN_sheet 요약, GN_decision 결론, run_id 목록, DECISIONS 번호. 코드 cutoff 11/29; freeze 뒤 `fix/*`는 `plots/`·`docs/` 렌더링만(`freeze_check`가 `ag/`·`engine_integration/` 변경 거부). 게이트 기준 변경은 DECISIONS 항목이 선행한다(3.6); G1 완화 없음.

**엔진 브랜치 통합**: `engine/*`는 patch 1개(`# AG:` hunk) + `shim/`. 허용 경로: `engine_integration/**`, `ag/systems/adapters/{sglang,vllm}_adapter.py`, `b0_engine.py`, `sglang_client.py`, openai_front의 sglang backend, `ag/reference/diff.py`, `ag/bench/`, `configs/targets/t3.yaml`, `plots/fig14.py`, `scripts/count_engine_loc.sh`, `scripts/e23_final.py`. 접점 서명 변경은 SPEC PR 선행. PR 조건: 두 CPU 명령 green + test_patch_applies; Slurm GPU 차등 테스트 첨부; `count_engine_loc.sh` 출력 [측정]; checks.json의 patch_state. PR 합류 후 클러스터의 3.11 venv에서 `pip install -e .`를 다시 하고 `agctl doctor`가 설치된 commit과 HEAD가 같은지 검사한다.

**결과와 commit(lineage)**: 결과는 `$AG_OUT`(`/home/212aig08/ag_results`)의 `raw/<E>/<run_id>/`. CPU 전용 작업은 `LG_CPU_PARTITIONS` 관례 또는 로그인 노드 nohup + `$AG_OUT/logs`. lineage 필드: code_commit(dirty), config_hash, env_hash.{harness, engine}, gpu_name, gpu_count, placement, hostname, slurm_job_id, slurm_array_job_id, slurm_array_task_id, started_at, finished_at, model_sha.{target, guard, judge}, data_manifest_sha, engine_version, patch_sha, producer, clock_offset_ns, parent_run_ids. 모든 JSONL 행에 run_id. `plots/_common.py`는 lineage 불일치 실패, `--require-tag`, `-dirty` 거부. 논문 그림의 run_id·code_commit·config_hash·1차 단계는 `figures/PROVENANCE.md`·`manifest.json`으로 논문 저장소에 태그와 묶는다.

## 6. 지금 결정할 것과 첫 행동

### 6.1 코드 전에 답이 필요한 결정 (DECISIONS.md의 첫 항목들)

3.3절의 15개에 단계 설계에서 새로 생긴 것을 더했다. 번호는 DECISIONS.md를 만들 때 붙인다.

| 결정 | 기본안(답이 없으면) | 기한 | 걸린 단계 |
| --- | --- | --- | --- |
| Python 환경 | **결정(D-001)**: 3.11 venv 하나, AttnTrace만 별도 venv | — | S0, S1E |
| Record v1 스키마(fp16 head 평균 기본, 전체 head는 calibration 100건만, `schema_version=1`) | 기본안으로 시작, 변경 시 bump | 10/12 | S1, S2, S3E |
| TTFV의 sink 점: 클라이언트 수신 vs 엔진 gate 직후 | 두 점 모두 기록, 논문은 gate 직후(B0는 클라이언트) | 10/12 | S1, S3 |
| grader LLM = 타깃과 다른 Qwen3-8B; B0 pilot 200건으로 라벨 | 기본안 | 10/13 | S2 |
| AgentDojo-L이 10/15까지 안 되면 4원천으로 G0 판정, 10/22 보충 | 기본안 | 10/15 | S2 |
| gated(Llama-3.1, Prompt Guard 2) 10/14 미승인 시 T2-only/J2-only로 시작 | 기본안 | 10/14 | S2 |
| E0.4를 10/25로(계획서 W0 밖), E1.1/E1.2/E1.5 ≤11/8, E1.4 11/5, E4.1/E4.2 W5 — 10.1절 일정 이탈 | 기본안 | 10/12 | S2b, S3, S3b, S5 |
| E2.4를 CPU 계약 테스트(S3)와 GPU판(S4, 11/8)으로 분할 | 기본안 | 10/28 | S3, S4 |
| E3.1 1차-A를 offline 재계산 행만으로 11/1–2 시작(reset·scheduler 완성 전) | 기본안 | 10/30 | S4 |
| E2.3-final의 ledger_export 모드 = E4에서 쓰는 모드; off는 하한선 | 기본안 | 11/10 | S5 |
| 품질 실험의 verdict cache: 요청마다 reset, config_hash 분리 | 기본안 | 10/16 | S2, S4 |
| E4.3 2-GPU 행 필수(10.3절 cut-7은 4-GPU만) | 기본안 | 11/15 | S5b |
| E4.*의 기본 judge J1; Fig. 1의 AG+J2는 E4.2에서 | 기본안 | 11/10 | S5 |
| Fig. 5 ← E0.4(+E3.5 scan 점), Fig. 8 ← E4.1(E1.4는 초안) | 기본안 | 10/12 | S2b, S3, S4b, S5 |
| 128k 창 수 분모: T1 1,007 / T2 1,023 | 기본안 | 10/28 | S3, S7 |
| Table 4 지연 ±20 %: GPU가 다르면 B0 대비 배수 | 기본안 | 10/25 | S3c |
| TP-2 ledger(T3) 불가 시: T3를 8k·32k만 TP 없이? | 보고 후 결정 | 10/27 | S3E, S4 |
| 엔진 변경 LoC 범위: patch 파일만 "엔진 변경", shim은 "AG 런타임" | 기본안 | 11/20 | S7 |
| Mooncake `arrival_scale`·20분 구간 규칙; 128k는 W-Len만 | 보고 후 결정 | 11/8 | S5 |
| LG 테스트가 3.11 CI 조건에서 미green이면 CLAUDE.md 3.4 명령 문구 조정 | 측정 후 | 10/12 | S0 |

### 6.2 첫 주(10/10–10/18) 행동 순서

1. 10/10: gated 모델 신청, Python venv 질문, DECISIONS.md 열기, 로그인 노드 다운로드 시작(`registry_ag`).
2. 10/11–12: SPEC.md(2.2·2.4·4.5절 내용), pyproject·CI·manifest·pr_check, LG 테스트 27모듈 CI 조건 측정, Mooncake·tool 대기 집계. [S0]
3. 10/12–14: 걷는 뼈대 PR 5개(harness/skeleton → reference-record → openai-front → lineage-ag → plots/fig09). 60초 mock end-to-end green. [S1]
4. 10/12–16: 3.11 venv 생성과 engine_env_check, 1요청 `attn_lse` spike 스크립트를 S2의 1요청 실측과 같은 날 제출. B0 플러그인·patch v0는 G0 뒤. [S1E, 1인 수행으로 조정]
5. 10/13: B0 pilot 200건 → 사람 라벨 → 10/16 grader 일치율. 10/14 payloads/background → 10/15 데이터 동결 + gate.json. [S2]
6. 10/15–17: 1요청 실측(T2 8k, T1 128k) → E0.1 array → E0.2. 10/18 G0 sheet + `docs/G0_decision.md`. [S2]
7. 병행: E0.3(10/18), E0.4 준비(10/25). [S2b]

## 부록 A. 이 문서를 만든 방법

- 저장소 14개는 2026-10-10에 shallow clone해 읽기만 했고(실행 안 함), 저장소마다 조사 에이전트 1개가 구조·driver·config·분리·baseline·지표·결과·그림·테스트·AE 문서를 파일 경로와 함께 기록했다. 합성 에이전트 1개가 관례 비교표·깊이·권장 배치·지표 정의·baseline 이식·반례를 썼다.
- 계획서의 실험·그림·구성 요소·baseline·데이터·게이트 목록은 추출 에이전트 1개가 만들고 검증 에이전트 2개(포괄성, 의존 관계)가 80개를 지적해 수정했다.
- 단계별 계획은 관점이 다른 초안 3개(뼈대 우선, 위험·임계 경로 우선, 그림 가치 우선)를 심사 에이전트 3개가 9개 기준으로 채점(합계 216 / 224 / 146)하고 합성한 뒤, 검증 에이전트 2개(포괄·의존, 제약·완전성)가 2라운드(138 → 126개 지적)로 고쳤다. 남은 지적 대부분은 결정 항목 번호 체계와 표기 통일에 관한 것이었다.
- 이 문서의 수치 출처 표기는 CLAUDE.md 3.5를 따른다. 조사 저장소의 LoC·테스트 수는 [측정, clone 기준], 계획서 GPU-h는 [추정], 단계 GPU-h 합계는 [계산]이다.
