# CLAUDE.md — 이 저장소에서 일할 때의 계약

이 파일은 Claude(및 사람)가 이 저장소에서 작업할 때 지켜야 하는 기준 문서, 클러스터 사실, 규칙을 적는다. 코드보다 먼저 읽는다.

## 1. 기준 문서

| 역할 | 파일 | 비고 |
| --- | --- | --- |
| **기준 계획서** | `docs/asymmetric-guard-osdi27-plan-2026-10-10.md` | 비대칭 가드(AG), OSDI '27. 주장 C1–C5 → Fig. 1–16 → 8.1절 지표 → E0–E6 순서로 모든 결정이 내려온다. 코드는 "그림 16개를 재현하는 프로그램"이다 |
| 이전 계획서 | `docs/plan_osdi_2026-09-29.md` | Landmark Guard(LG) v2. 참고만 한다. LG는 AG의 **유해 콘텐츠 경로**(설계 1)로 그대로 재사용한다 |
| 설계도 | `docs/SPEC.md` | 구성 요소 인터페이스, 지표 정의식, 결과 파일 스키마, 시스템 플러그인 API |
| 결정 기록 | `docs/DECISIONS.md` | 계획서와 다르게 구현해야 할 때 먼저 여기에 적는다(아래 규칙 3.6) |
| 기존 LG 코드 | `lgs1`–`lgs8`, `s1ctl.py`–`s8ctl.py`, `slurm/`, `tests/` | LG S1–S8 산출물. AG 코드는 이것을 유해 경로로 import하고, 되도록 고치지 않는다 |
| 구축 계획 | `docs/build-plan-ag-osdi27-2026-10-10.md` | 단계 S0–S7(+선택 S8a–g), 걷는 뼈대, 브랜치·태그 전략. 모든 단계 담당은 강지의(D-002), 코드는 Claude가 쓴다 |

계획서의 번호를 그대로 쓴다. 주장은 C1–C5, 실험은 E0.1–E6.4, 그림은 Fig. 1–16, 표는 Table 1–4, baseline은 B0/B-oracle/B-full/B1a–B6/N1–N6, 타깃은 T1–T4, judge는 J1–J4, workload는 W-QA/W-Agent/W-MT/W-Len, 게이트는 G0–G4, 주는 W0–W7.

## 2. 개발 원칙 (8개, 순서대로)

1. **주장에서 역산한다.** 주장 → 그림 → 지표 → 실험 → 코드. 새 코드를 쓰기 전에 그것이 어느 Fig./Table의 어느 값을 만드는지 적는다.
2. **시스템보다 측정이 먼저다.** 동기 그림(Fig. 3–5)과 go/no-go(E0, G0)는 엔진을 건드리지 않고 참조 구현(HF transformers, SDPA prefill + eager decode)으로 잰다.
3. **정답 기준(oracle)을 두고 차등 테스트한다.** 느리지만 확실한 참조 구현을 먼저 만들고, 빠른 구현(SGLang 안의 ledger)은 그것과 대조한다(E2.1 블록 질량 오차 ≤ 10⁻³·상위 8 일치율 ≥ 99 %, E2.2 토큰 100 % 동일).
4. **시스템과 하네스를 분리한다.** B0부터 AG까지 모든 시스템이 같은 인터페이스 `system.run(request) → events`로 하네스에 꽂힌다. baseline 추가는 플러그인 하나 추가다. 측정은 전부 하네스가 한다.
5. **설정은 파일, 실행은 기록.** 실험 조건은 YAML. 실행마다 commit hash, config hash, env, GPU 종류를 결과 디렉터리에 같이 쓴다(lineage). 조건이 다른 결과는 섞이지 않는다.
6. **baseline은 원 수치를 먼저 재현한다.** 우리 설정에 쓰기 전에 원 논문 설정에서 원 수치가 나오는지 확인한다(Table 4 이식 검증표. 품질 ±3 pp, 지연 ±20 %).
7. **그림 하나 = 스크립트 하나, 논문 숫자는 자동 생성.** `plots/figNN.py`는 결과 디렉터리(`$AG_OUT/raw`, `$AG_OUT/processed`; SPEC 7.1)만 읽어 그림을 만든다. 초록의 수치 7개(계획서 9.3절)는 `numbers.json`에서 나온다. 재실행하면 전부 갱신된다.
8. **작게 끝까지, 그다음 크게.** mock 모델로 CPU에서 요청 하나가 데이터 → 시스템 → 측정 → 결과 파일 → 그림까지 통과한 뒤 규모를 키운다.

## 3. 규칙

### 3.1 언어

- 코드, 주석, docstring, CLI 출력, 로그, 결과 파일의 키: **영어**.
- README, `docs/`, 커밋 메시지 본문, PR 설명: **한국어**. 커밋 제목은 어느 쪽이든 되지만 한 커밋 안에서 섞지 않는다.

### 3.2 git

- **main에 직접 push 금지.** 작업마다 브랜치를 만들고 PR로 합친다. 브랜치 이름은 `<주제>/<짧은-설명>` (예: `e0/ref-ledger`, `spec/plugin-api`).
- 커밋은 작게, 제목은 무엇을 왜 바꿨는지. 결과 파일(`results/`, `*.jsonl`, 체크포인트, 모델)은 커밋하지 않는다. 그림(`figures/*.pdf`)은 논문 저장소 쪽에서만 추적한다.
- `env/*.env`(비밀값 포함)는 커밋하지 않는다. `env/*.env.example`만 올린다. `hf_…`, `ghp_…` 토큰이 보이는 파일은 커밋 전에 막는다.

### 3.3 lineage

- 모든 실행은 결과 디렉터리에 `lineage.json`을 남긴다. 최소 필드: `code_commit`(dirty 표시 포함), `config_hash`(YAML의 정규화 해시), `env_hash`(`pip freeze` + 주요 환경 변수), `gpu_name`, `gpu_count`, `hostname`, `slurm_job_id`, `started_at`, `finished_at`, `model_sha`(타깃·guard·judge 각각), `data_manifest_sha`.
- 결과 행(JSONL)마다 `run_id`를 붙인다. `run_id`로 `lineage.json`을 찾을 수 있어야 한다. `lgs1/lineage.py`에서는 `canonical_hash`, `dir_fingerprint`, `code_commit`, `iter_rows`만 재사용하고, 비교·쓰기는 AG 필드 전체를 다루는 `ag/harness/{lineage,writer}.py`로 한다(D-038, SPEC 7.2).
- 같은 `run_id`에 다른 lineage로 이어 쓰기(resume) 금지. 불일치는 오류로 끝낸다.

### 3.4 mock 모델로 CPU 통과

- 새 실험 코드(데이터 생성기, 시스템 플러그인, 하네스, 지표, 그림 스크립트)는 **mock 모델로 CPU에서 끝까지** 돌아야 한다. `tests/`에 그 경로의 테스트를 둔다. GPU 없이 `python -m pytest tests -x -q`와 `pytest $(cat tests/manifests/cpu.txt)`가 green이어야 PR을 연다. GPU 테스트는 `tests/gpu/`에 두고 CUDA가 없으면 skip한다.
- mock은 결정적(deterministic)이어야 하고, 실제 모델과 같은 인터페이스(SPEC의 `Target`, `HarmGuard`, `Judge`, `EngineAdapter`)를 구현한다. mock의 attention은 합성 규칙(예: 주입 창에 질량 집중)으로 만들어 hit@k 계산 경로를 검증한다.
- 규모 확대는 "mock CPU → 실제 모델 1 요청 GPU → 실험 부분집합 → 전량" 순서로만 한다.

### 3.5 수치 표기

- 문서, PR 설명, 보고에 적는 모든 수치에는 출처를 붙인다. `[측정]` 우리가 잰 값, `[계산]` 산술로 얻은 값, `[문헌]` 논문·문서·소스 코드에서 확인한 값, `[추정]` 근거가 약한 어림값, `[해석]` 사실에서 끌어낸 판단.
- 계획서의 통과 기준은 전부 **제안값**이다. 기준을 바꾸면 `docs/DECISIONS.md`에 적는다.
- 새 방향(AG)의 측정값은 아직 없다. 계획서에서 `[측정]`으로 표시된 값은 모두 LG의 값이다.

### 3.6 계획서와 다르게 해야 할 때

- 계획서와 다르게 구현해야 하면 **코드를 쓰기 전에** `docs/DECISIONS.md`에 항목을 추가하고(날짜, 계획서의 절 번호, 원래 내용, 바꾸려는 내용, 이유, 영향 받는 실험·그림) **나(강지의)에게 물어본다.** 답을 받기 전에는 구현하지 않는다.
- 계획서 12.2절의 "확인이 필요한 사실"에 해당하는 것을 확인하면 그 결과도 `docs/DECISIONS.md`에 적는다(확인됨/반증됨, 근거).
- SPEC의 가정(`[가정]` 표시)이 깨지면 같은 절차를 따른다.

### 3.7 코드 작성 범위

- 사용자가 "코드 금지" 또는 "설계만"이라고 한 작업에서는 `.py` 파일을 만들지 않는다. 인터페이스는 `docs/SPEC.md`에 Python 시그니처로 적는다.
- 기존 LG 코드(`lgs*`, `s*ctl.py`)는 AG가 유해 경로로 쓰는 부분만 import한다. 고쳐야 하면 별도 PR로 분리한다.
- 엔진(SGLang, vLLM) 수정은 fork의 patch 파일 또는 별도 디렉터리에 두고 LoC를 셀 수 있게 한다(논문에 엔진 변경 LoC를 따로 적는다).

## 4. 클러스터 사실

| 항목 | 값 |
| --- | --- |
| 저장소 | `https://github.com/ACPLJE/Efficient-Guardrail` (public). 클러스터 clone 위치는 `/home/212aig08/guardrail/main/Efficient-Guardrail`(로그인 노드에서 clone·pull) |
| 스케줄러 | Slurm |
| GPU | RTX 6000 Ada 48 GB. 파티션 `RTX6000ADA` |
| Python | **AG 작업은 Python 3.11 venv 하나**(D-001). 제안 경로 `/home/212aig08/ag_venv/ag311`(생성 전). 하네스·SGLang·shim·judge·AgentDojo를 모두 여기에 설치한다. AttnTrace(B3)만 별도 venv(Python 3.10.15·torch 2.5.1 고정)에 두고 파일·subprocess 경계로 부른다(SPEC 9.2·13절). 기존 3.9 venv `source /home3/212aig08/servsec/exp/secu/bin/activate`는 LG 기존 결과 재현용이고 AG 코드에는 쓰지 않는다 |
| 네트워크 | **계산 노드는 외부 네트워크 없음.** `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` 필수 |
| HF 캐시 | `/home/212aig08/.cache/huggingface` (`HF_HOME`). 모델·데이터는 **로그인 노드에서 미리 받는다** (gated 모델은 로그인 노드에서 `HF_TOKEN`) |
| 큰 데이터·결과 | `/home/212aig08` 쪽에 둔다. `/home3`는 쿼터가 작다 |
| job 제출 | 모든 GPU job에 `--gres=gpu:1` **필수**(QOSMinGRES). QOS는 `normal`. GPU 2·4장은 `--gres=gpu:2`, `--gres=gpu:4` |
| job array | `%N` throttle **금지**(예: `--array=0-9%2` 쓰지 말 것). 동시 실행 수는 array 크기나 의존성(`--dependency=afterok`)으로 조절 |
| 상태 조회 | `sacct` **없음**. `squeue -u 212aig08`, `scontrol show job <id>`, `sbatch --test-only`, `sinfo` 사용. 완료 job의 기록은 우리가 남기는 `lineage.json`과 Slurm 로그로 대신한다 |
| 공통 스크립트 | `slurm/common.sh`가 env 파일을 읽고 offline 모드·python을 정한다. 새 sbatch는 이것을 source한다 |

sbatch 템플릿의 최소 헤더(AG job은 S0에서 만드는 `slurm/ag_common.sh`를 source한다. 이 스크립트가 `env/ag.env`를 읽고 3.11 venv를 활성화하며 offline 모드를 강제한다. 기존 `slurm/common.sh`는 LG용이다):

```bash
#!/bin/bash
#SBATCH -p RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out
source "$(dirname "$0")/../slurm/ag_common.sh"
```

## 5. 작업 시작 체크리스트

1. 기준 계획서의 해당 실험(E*.*)과 그림(Fig./Table) 번호를 확인한다.
2. `docs/SPEC.md`의 인터페이스와 결과 스키마를 따른다. 바꿔야 하면 SPEC을 먼저 고치고 PR에 적는다.
3. 브랜치를 만든다. mock CPU 테스트를 먼저 쓴다.
4. 실행 전 `sbatch --test-only`로 헤더를 검증한다. 실행마다 `lineage.json`이 생기는지 본다.
5. PR 설명에 재현 명령, 결과 디렉터리, 수치(출처 표기)를 적는다.
