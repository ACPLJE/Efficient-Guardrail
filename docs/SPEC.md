# AG 설계 명세 (SPEC) v1

2026-10-10 · 강지의 (Jieui) · 상태: 초안 v1, `schema_version = 1`

기준 문서: 계획서 `docs/asymmetric-guard-osdi27-plan-2026-10-10.md`(이하 "계획서"), 구축 계획 `docs/build-plan-ag-osdi27-2026-10-10.md`, 결정 기록 `docs/DECISIONS.md`. 우선순위는 계획서 > DECISIONS > SPEC > 코드다. SPEC이 계획서와 다르면 DECISIONS에 항목이 있어야 한다. 코드가 SPEC과 다르면 SPEC을 먼저 고치는 PR을 낸다(CLAUDE.md 5절).

값의 출처 표기는 CLAUDE.md 3.5를 따른다. `[가정]`은 SPEC이 기대는 미확인 사실이고 12절에 모아 두었다. 이 문서의 Python 블록은 시그니처이고 구현이 아니다.

## 0. 범위와 원칙

이 명세가 정하는 것은 다음 열 가지다: 식별자와 단위(1절), 데이터 모델(2절), 이벤트(3절), 플러그인 인터페이스(4절), 기록 형식 Record v1(5절), 지표 정의식(6절), 결과 디렉터리와 파일 스키마(7절), 설정 YAML(8절), 시스템·workload 카탈로그(9·10절), 실험별 적용 확인(11절). 가정은 12절, 환경은 13절, 변경 규칙은 14절이다.

원칙(CLAUDE.md 2절을 인터페이스로 옮긴 것):

1. **플러그인은 이벤트만 낸다.** B0부터 AG까지 모든 시스템은 `System.run(request) -> Iterator[Event]`만 구현한다. 지표는 `ag/metrics/`의 함수 하나가 이벤트와 결과 행에서 계산한다. 플러그인이 지표를 계산하거나 파일에 쓰지 않는다.
2. **시계는 하나다.** 모든 시각은 같은 호스트의 `time.monotonic_ns()`(Linux CLOCK_MONOTONIC) 값이다. 하네스, 엔진, judge pool은 한 Slurm job 안 한 노드에서 돈다(12절 A-01).
3. **모든 결과 행에 `run_id`가 있고, `run_id`로 `lineage.json`을 찾는다.** lineage가 다른 run에 이어 쓰기는 오류다.
4. **모든 추상 클래스에 결정적 mock 구현이 있다.** mock만으로 데이터 → 시스템 → 이벤트 → 지표 → 결과 → 그림이 CPU에서 돈다.
5. **한 번 생성하고 여러 번 평가한다**(계획서 5.1). 타깃 출력과 토큰별 ledger를 Record로 남기면 k, chunk 크기, 집계, detector 교체, B-oracle, N1·N4·N6은 재생성 없이 계산한다.
6. Python 3.11(D-001). 타입 힌트는 `X | None`, `list[int]`를 쓴다.

## 1. 식별자와 단위

### 1.1 토큰, 블록, 창

- **토큰 위치**는 타깃 tokenizer로 chat template을 적용한 전체 prompt의 0부터 시작하는 index다. 신뢰 여부와 payload 위치는 렌더링 뒤 토큰 span으로 바뀐다(2.3절).
- **블록**: 크기 `B = 128`(설정 `split_tokens`). 블록 j는 토큰 `[jB, min((j+1)B, n))`. 블록 수 `NB = ceil(n / B)`. 마지막 블록은 짧을 수 있다.
- **창**: 인접 블록 두 개, 창 w = 블록 {w, w+1}, `w = 0 … NB-2`. 창 수 `NW = NB - 1`(NB ≥ 2). 창 길이 256토큰, stride 128.
- 대표 창 수 \[계산\]:

| 입력 토큰 n | NB | NW | 비고 |
| --- | --- | --- | --- |
| 8,192 | 64 | 63 | 8k |
| 32,768 | 256 | 255 | 32k |
| 129,024 | 1,008 | 1,007 | T1·T3·T4의 128k(문맥 131,072 − 2,048) |
| 131,072 | 1,024 | 1,023 | T2의 128k |

- 이 격자는 블록 경계에 정렬된다. LG 유해 경로의 격자(`lgs1/lgcore/windows.make_windows`)는 마지막 창을 문서 끝에 맞추려고 왼쪽으로 민다. 길이가 128의 배수일 때 두 격자는 같고, 아닐 때 마지막 창만 다르다. 유해 경로는 LG 격자를, injection 경로는 이 격자를 쓴다.
- **창 명세** `WindowSpec(block_tokens, span_blocks)`: 기본 `(128, 2)`. judge가 고유 창을 쓰면 다른 명세를 쓴다. 예: B1a의 Prompt Guard 2 고유 창 512/256은 `(256, 2)`(계획서 7.2).

### 1.2 신뢰 여부와 창 자격 (D-019)

- 토큰은 신뢰(system prompt, 사용자 과업, chat template) 또는 비신뢰(검색 문서, 도구 출력, 업로드 파일)다. 경계는 애플리케이션이 알려 준다(계획서 3.1; 12절 A-02).
- 블록에 비신뢰 토큰이 하나라도 있으면 그 블록은 비신뢰다. 창은 두 블록 중 하나라도 비신뢰면 **순위 대상(eligible)**이다. 순위 대상이 아닌 창의 점수는 `-inf`다.

### 1.3 payload 창

- 공격 요청 r의 payload 토큰 집합을 `Pay(r)`라 한다. 창 w가 `|w ∩ Pay(r)| ≥ 0.5 · |Pay(r)|`이면 w는 payload 창이다(계획서 8.1). 창이 겹치므로 payload 창은 하나 또는 두 개다. 집합을 `P(r)`라 쓴다.
- payload가 256토큰보다 길어 어느 창도 절반을 담지 못하면 `P(r)`은 payload 토큰을 가장 많이 담은 창 하나로 정하고 행에 `payload_long = true`를 표시한다(계획서 미명시, SPEC 정의).

### 1.4 chunk

- chunk 0은 첫 8토큰, 이후는 32토큰 또는 문장 경계다(설정 `chunking`, 계획서 4.4). 도구 호출은 호출 전체가 chunk 하나다. 마지막 chunk는 짧을 수 있다. `chunk_id`는 요청 안에서 0부터 센다.
- 오프라인 분석에서 쓰는 **표준 chunk 정의** 3종(계획서 8.2 E0.1): `first8`(토큰 0–7), `first32`(0–31), `all`(출력 전체). 이것과 별개로 서빙 실행의 chunk는 `chunking` 설정을 따른다.

### 1.5 식별자

| 이름 | 형식 | 예 |
| --- | --- | --- |
| `request_id` | `<source>/<split>/<length_class>/<idx:06d>/<a|b>`, manifest가 정함 | `bipia/test/32k/000123/a` |
| `episode_id` | 여러 LLM 호출로 된 과제(AgentDojo, W-MT)의 id. 단일 호출이면 `request_id`와 같음 | `agentdojo_l/test/32k/workspace-u12-i3` |
| `run_id` | `<E>-<system>-<workload|data>-<config_hash[:8]>-<seed>-<YYYYmmddTHHMMSS>` | `E0.1-B0-lcinject32k-3fa2c9e1-0-20261015T0412` |
| 하위 run | 같은 실험의 별도 조건 실행에 접미사 | `E2.3-final`, `E2.4-gpu` |
| `system_id` | 계획서 7.2 그대로 | `B0`, `B1a`, `B-oracle`, `N3`, `AG` |
| `judge_id` | `J1`–`J4`, 출력 guard `OG`, mock `MJ` | `J2` |
| `target_id` | `T1`–`T4`, mock `MT` | `T1` |
| `job_id` | `<request_id>#<job_kind>#<seq>` | |

### 1.6 시간

- 모든 시각은 `int` 나노초, `time.monotonic_ns()`. 파일에는 같은 run의 첫 `ARRIVED`를 0으로 하는 상대값이 아니라 원래 값을 쓰고, `lineage.json`에 run 시작 시점의 `monotonic_ns`와 `time.time_ns()` 쌍을 남긴다.
- mock과 테스트는 `VirtualClock`(4.1절)을 쓴다. 실제 run에서 `VirtualClock`을 쓰면 `lineage.clock = "virtual"`이고 `latest.txt`가 거부한다.

## 2. 데이터 모델

모든 데이터 클래스는 `@dataclass(frozen=True)`이고 `to_json()`/`from_json()`이 있다. 정의는 `ag/data/types.py` 한 곳에 둔다.

### 2.1 요청 (타깃과 무관한 텍스트 수준)

```python
Role = Literal["system", "user", "assistant", "tool", "document"]

@dataclass(frozen=True)
class Segment:
    role: Role
    text: str
    trusted: bool                                   # system/user(task)=True, document/tool=False
    payload_char_spans: tuple[tuple[int, int], ...] = ()   # injected text inside this segment
    harm_char_spans: tuple[tuple[int, int], ...] = ()      # harmful needle (mixed docs, E3.2)
    tool_name: str | None = None

@dataclass(frozen=True)
class AttackMeta:
    attack_type: str        # naive|ignore|fake_completion|combined|tool_knowledge|task_aligned|adaptive_*|fragment_k|data_injection
    position: float | None  # 0.1 / 0.5 / 0.9 relative depth, None for multi-site
    goal: str               # injected task, input to the success judge
    template_id: str        # payload template (split unit, plan 5.1)
    n_sites: int = 1        # fragmented payload (E5.2)

@dataclass(frozen=True)
class Request:
    request_id: str
    episode_id: str
    turn: int                           # 0 for single-call requests
    source: str                         # agentdojo_l|injecagent|bipia|longpibench|piarena|llmail|lc_harm|mooncake_fill|...
    split: Literal["calib", "test", "pilot"]
    length_class: str                   # "1k".."128k", "16k", "64k"
    background_tier: Literal["low", "mid", "high", "follow"] | None
    task: str                           # trusted user task text (also in segments)
    segments: tuple[Segment, ...]
    sink: Literal["stream", "tool"]
    max_new_tokens: int
    ignore_eos: bool                    # True only for perf workloads (fixed output length)
    attack: AttackMeta | None           # None = benign
    task_meta: Mapping[str, Any]        # success-judge inputs: gold answer, ground-truth tool calls, ...
    tools: tuple[Mapping[str, Any], ...] = ()   # tool schemas for tool-sink requests
    arrival_ns: int | None = None       # set by a Workload (10절); None for closed-loop runs
    data_manifest_sha: str = ""
```

- 공격 요청과 정상 요청은 짝이다. 짝의 `request_id`는 마지막 글자(`a`/`b`)만 다르다(계획서 5.1·5.3).
- AgentDojo처럼 도구 응답에 따라 다음 호출이 정해지는 과제는 호출마다 `Request`가 생기고 `episode_id`, `turn`으로 묶인다. 품질 결과는 episode 단위다(2.5절).

### 2.2 렌더링 결과 (타깃별 토큰 수준)

```python
@dataclass(frozen=True)
class RenderedPrompt:
    request_id: str
    target_id: str
    prompt_ids: tuple[int, ...]
    untrusted_mask: bytes                # 1 bit per token, packed
    payload_spans: tuple[tuple[int, int], ...]   # token spans of Pay(r)
    harm_spans: tuple[tuple[int, int], ...]
    split_tokens: int                    # B
    n_blocks: int                        # NB
    block_untrusted: bytes               # 1 byte per block (0/1)
    payload_windows: tuple[int, ...]     # P(r), empty for benign
    payload_long: bool
    eligible_windows: int                # count of eligible windows
```

- `render(request, tokenizer, chat_template, split_tokens) -> RenderedPrompt`는 `ag/data/render.py` 하나다. char span을 토큰 span으로 바꿀 때는 tokenizer의 offset mapping을 쓴다. offset mapping이 없는 tokenizer는 쓰지 않는다 \[가정 A-03\].
- 길이 등급을 맞추는 것은 데이터 생성기(LC-Inject)의 일이다. 렌더링은 길이를 바꾸지 않는다. 타깃마다 tokenizer가 달라 같은 요청의 토큰 수가 다르므로, 생성기는 **주 타깃별로** 길이를 맞춘 요청을 만든다(T1용, T2용 manifest가 다르다).

### 2.3 생성 결과와 판정

```python
@dataclass(frozen=True)
class ChunkInfo:
    chunk_id: int
    tok_start: int; tok_end: int        # [start, end) in output tokens
    is_tool_call: bool

@dataclass(frozen=True)
class Generation:
    request_id: str
    output_ids: tuple[int, ...]
    text_generated: str                  # everything the target produced
    text_delivered: str                  # only chunks that reached the sink
    tool_calls_delivered: tuple[Mapping[str, Any], ...]
    chunks: tuple[ChunkInfo, ...]
    blocked: bool                        # any chunk blocked or request blocked
    block_reason: str | None             # harm_input|inj_window|output_guard|deadline_miss|probe|...
    error: str | None

@dataclass(frozen=True)
class Verdict:
    judge_id: str
    request_id: str
    chunk_id: int | None                 # None = pre-generation verdict
    window_id: int | None                # None = chunk-level verdict (output guard)
    score: float                         # higher = more likely injection/unsafe
    threshold: float
    unsafe: bool                         # score >= threshold
    latency_ns: int
    cached: bool
```

- 점수는 항상 "높을수록 위험"이다(`lgs1/metrics.py`와 같은 규약). 임계값은 judge가 아니라 운영점(6.1절)이 정하고, 실행 시 `threshold_id`로 고정된다.

### 2.4 결과 행 (요청 또는 episode 단위)

```python
@dataclass(frozen=True)
class Outcome:
    run_id: str
    request_id: str                      # or episode_id for multi-call tasks
    unit: Literal["request", "episode"]
    system_id: str; target_id: str; judge_id: str | None
    is_attack: bool
    attack_success: bool | None          # judged on delivered output only; None if not applicable
    attack_success_b0: bool | None       # same request under B0 (joined from the B0 run), defines S
    utility: bool | None
    blocked: bool
    false_block: bool                    # benign and blocked
    error_kind: str | None               # engine|api|timeout|oom|judge; never folded into ASR
    judged_windows: int                  # distinct windows judged (6.1절)
    judged_windows_uncached: int
    selected: Mapping[int, tuple[int, ...]]     # chunk_id -> selected window ids
    payload_windows: tuple[int, ...]
    hit: Mapping[str, Mapping[int, bool]]       # chunk_def -> k -> hit (offline scorers)
    doc_scores: Mapping[str, float]             # "all_windows" / "traced" -> max score (offline)
    success_judge: str                   # e.g. "agentdojo-native@0.1.35", "qwen3-8b-grader@v1"
```

## 3. 이벤트

### 3.1 종류

```python
class EventType(IntEnum):
    ARRIVED = 1          # harness: planned arrival time (open loop) or submit time
    ADMITTED = 2         # harness: handed to System.run (after client-side queue)
    PREFILL_START = 3    # system/engine
    PREFILL_DONE = 4     # system/engine: first output token generated
    TOKENS = 5           # system/engine: output tokens [i0, i1) generated
    CHUNK_GENERATED = 6  # system: chunk complete (all its tokens generated)
    JOB_SUBMITTED = 7    # system: a guard/judge/attribution job enqueued
    JOB_STARTED = 8
    JOB_DONE = 9         # carries verdicts or a job result
    CHUNK_RELEASED = 10  # system gate: chunk allowed to the sink
    CHUNK_BLOCKED = 11   # system gate: chunk dropped (fail or deadline miss)
    CHUNK_DELIVERED = 12 # harness client: chunk received at the sink
    REQUEST_BLOCKED = 13 # system: whole request refused before or during generation
    FINISHED = 14        # system: carries the Generation summary
    ERROR = 15

class JobKind(StrEnum):
    HARM_SCAN = "harm_scan"          # LG trunk over all windows (plan 4.1)
    HARM_DEEP = "harm_deep"          # LG deep head on router top windows
    OUTPUT_GUARD = "output_guard"
    INJ_JUDGE = "inj_judge"          # judge(task, window, chunk)
    ATTRIBUTION = "attribution"      # second-model attribution (B3)
    PROBE = "probe"                  # activation probe (B2, N3 scoring)
    SANITIZE = "sanitize"            # input rewriting (B4)
    TOOL_COMMIT = "tool_commit"      # all verdicts for a tool call

class DeadlineKind(StrEnum):
    FIRST_TOKEN = "first_token"; RELEASE = "release"; COMMIT = "commit"; NONE = "none"
```

계획서 4.5의 마감 종류(first-token, release, commit)는 `DeadlineKind`로, 구축 계획의 "이벤트 12종"은 이 15종으로 대체한다. 늘어난 것은 `JOB_STARTED`, `CHUNK_DELIVERED`, `REQUEST_BLOCKED`다. `CHUNK_DELIVERED`는 D-004(TTFV는 클라이언트 수신)를 위해 필요하다.

### 3.2 형식

```python
@dataclass(frozen=True)
class Event:
    run_id: str
    request_id: str
    seq: int                       # per (run_id, request_id), assigned by the harness on receipt
    type: EventType
    t_ns: int
    t_source: Literal["harness", "engine", "client", "judge_pool", "virtual"]
    chunk_id: int | None = None
    tok_range: tuple[int, int] | None = None      # TOKENS, CHUNK_GENERATED
    job_id: str | None = None
    job_kind: JobKind | None = None
    deadline_kind: DeadlineKind | None = None
    deadline_ns: int | None = None
    window_ids: tuple[int, ...] = ()              # JOB_*: windows the job covers
    required_jobs: tuple[str, ...] = ()           # CHUNK_GENERATED: jobs that gate this chunk
    verdicts: tuple[Verdict, ...] = ()            # JOB_DONE
    reason: str | None = None                     # *_BLOCKED, ERROR
    payload: Mapping[str, Any] = field(default_factory=dict)  # extra, e.g. gpu_mem_mib, extra_pass_ms
```

- **정렬 키**는 `(t_ns, request_id, type, seq)`이다. 같은 시각이면 type 번호가 작은 것이 먼저다.
- `FINISHED.payload["generation"]`에 `Generation`이 들어간다. 지표 계산은 이벤트와 `Generation`만 쓴다.

### 3.3 누가 무엇을 내는가

| 이벤트 | 내는 쪽 | 비고 |
| --- | --- | --- |
| ARRIVED, ADMITTED | 하네스 | open loop는 예정 도착 시각, closed loop는 제출 시각 |
| CHUNK_DELIVERED | 하네스(클라이언트 수신) | 하네스 프로세스 안에서 도는 시스템(offline, mock)은 `CHUNK_RELEASED`와 같은 시각 |
| PREFILL_*, TOKENS | 엔진 shim(`t_source = engine`). 무수정 엔진(B0)은 클라이언트 수신 시각(`t_source = client`) | B0의 `PREFILL_DONE`은 첫 토큰 수신 시각이다(12절 A-04) |
| CHUNK_GENERATED, JOB_*, CHUNK_RELEASED/BLOCKED, REQUEST_BLOCKED, FINISHED | 시스템 플러그인(gate, judge pool, scheduler) | |
| ERROR | 누구나 | `reason`에 종류 |

- 엔진 쪽 이벤트는 같은 노드의 shim이 run 디렉터리의 spool 파일(`engine_events.jsonl`) 또는 로컬 socket으로 보낸다. 어느 방식이든 하네스가 받아 `seq`를 붙이고 `events.jsonl`에 쓴다 \[가정 A-05\].

### 3.4 불변식 (테스트로 고정)

1. 요청마다 `ARRIVED ≤ ADMITTED ≤ PREFILL_START ≤ PREFILL_DONE`.
2. chunk j의 `CHUNK_RELEASED`는 `CHUNK_GENERATED(j)`와 `required_jobs(j)`의 모든 `JOB_DONE`보다 늦고, 모든 verdict가 `unsafe = False`다(계약, 계획서 3.3). 단 `release_before_verdict = True`인 시스템(B1b, N5)은 예외이고 이것이 판정 전 노출 토큰으로 잡힌다.
3. chunk는 순서대로 release된다: `CHUNK_RELEASED(j)`는 `CHUNK_RELEASED(j-1)` 뒤.
4. chunk j가 block되면 j 이후 chunk는 release되지 않는다(요청 중단).
5. 첫 chunk는 `HARM_SCAN`(켜져 있을 때)의 통과 뒤에만 release된다.
6. 마감까지 verdict가 없으면 `CHUNK_BLOCKED(reason = deadline_miss)`(fail-closed).
7. 요청마다 `FINISHED` 또는 `ERROR`가 정확히 하나.

## 4. 플러그인 인터페이스

모든 추상 클래스는 `ag/` 아래 한 모듈에 정의하고, 같은 모듈 옆에 mock 구현을 둔다. 괄호 안은 모듈 경로다.

### 4.1 실행 맥락과 시계 (`ag/harness/context.py`)

```python
class Clock(Protocol):
    def now_ns(self) -> int: ...
    def sleep_until(self, t_ns: int) -> None: ...

class MonotonicClock(Clock): ...        # time.monotonic_ns
class VirtualClock(Clock):              # deterministic; advance() moves time
    def advance(self, dt_ns: int) -> None: ...

@dataclass
class RunContext:
    run_id: str
    exp_id: str
    out_dir: Path                        # $AG_OUT/raw/<E>/<run_id>
    clock: Clock
    seed: int
    lineage: "AGLineage"
    records: "RecordStore | None"        # read or write, depending on the system
    emit: Callable[[Event], None]        # for side-channel events (judge pool threads, engine spool)
    log: logging.Logger
```

### 4.2 시스템 (`ag/systems/base.py`)

```python
@dataclass(frozen=True)
class SystemInfo:
    system_id: str
    mode: Literal["live", "offline", "external"]
    engine: str | None                   # "sglang-0.5.21", "hf-reference", "mock", "replay"
    attn_backend: str | None
    patch_applied: bool
    patch_sha: str | None
    shim_active: bool
    model_sha: Mapping[str, str]         # target/guard/judge -> sha
    placement: Mapping[str, str]         # component -> "gpu0" | "gpu1" | ...
    flags: Mapping[str, Any]             # resolved AG flags (9.2절)

class System(abc.ABC):
    system_id: ClassVar[str]
    def setup(self, cfg: "SystemConfig", ctx: RunContext) -> None: ...
    def run(self, req: Request) -> Iterator[Event]: ...             # must be thread-safe
    def run_many(self, reqs: Iterable[Request]) -> Iterator[Event]:  # default: chain(run(r) for r in reqs)
        ...
    def reset(self) -> None: ...          # clear prefix/verdict/judge caches between conditions
    def teardown(self) -> None: ...
    def info(self) -> SystemInfo: ...
```

- `run`은 한 요청의 이벤트를 시간 순으로 내는 blocking generator다. 하네스는 open loop에서 요청마다 스레드 하나로 `run`을 부른다(I/O 대기 위주라 스레드로 충분하다; 12절 A-06).
- `run_many`는 batch가 이득인 offline 시스템(기록 위 재계산, offline judge)이 덮어쓴다. 한 시스템이 두 방식을 모두 지원해도 된다.
- `mode`: `live`는 실제 생성을 한다(엔진 또는 참조 구현). `offline`은 Record와 저장된 점수에서 다시 계산한다(새 생성 없음). `external`은 다른 도구가 만든 결과를 읽어 `Outcome`만 채운다(B5 MELON).

### 4.3 타깃 (`ag/targets/base.py`)

참조 구현, offline 시스템, probe, 공격 생성기가 쓰는 모델 수준 인터페이스다. 서빙 엔진은 4.4절의 `EngineAdapter`로 따로 다룬다.

```python
@dataclass(frozen=True)
class RecordSpec:
    layers: tuple[int, ...] | Literal["all", "upper_half"]
    per_chunk_heads: bool = True          # chunk_head_mass (5.2절)
    gen_mass: bool = True                 # for rollout
    first_token: bool = True              # D-018
    precision: Literal["fp16", "fp32"] = "fp16"
    chunk_defs: tuple[str, ...] = ("first8", "first32", "all")

@dataclass(frozen=True)
class TargetOutput:
    output_ids: tuple[int, ...]
    text: str
    tool_calls: tuple[Mapping[str, Any], ...]
    t_prefill_done_ns: int
    t_token_ns: tuple[int, ...]
    peak_mem_bytes: int
    record: "LedgerRecord | None"

class Target(abc.ABC):
    target_id: str
    n_layers: int; n_q_heads: int; n_kv_heads: int; max_context: int
    def render(self, req: Request) -> RenderedPrompt: ...
    def generate(self, rp: RenderedPrompt, max_new_tokens: int, *, record: RecordSpec | None,
                 forced_output: Sequence[int] | None = None) -> TargetOutput: ...
    def hidden_states(self, rp: RenderedPrompt, layers: Sequence[int],
                      positions: Sequence[int] | Literal["last", "all"]) -> np.ndarray: ...
    def info(self) -> Mapping[str, Any]: ...
```

- `HFReferenceTarget`(`ag/reference/ledger_oracle.py`): prefill은 SDPA, decode 단계만 eager attention으로 새 토큰의 attention 행을 얻어 128토큰 블록으로 합산한다(계획서 6.4, 8.2). 128k에서 전체 logits를 만들지 않도록 `logits_to_keep=1` 또는 chunked prefill을 쓴다(전체 logits는 약 33 GB \[계산\]).
- `forced_output`은 teacher forcing이다. E2.1에서 엔진이 만든 출력을 참조 구현에 그대로 넣어 같은 토큰 열 위의 질량을 비교한다.
- `MockTarget`: `tests/conftest.py`의 `tiny`(`lgs1/lgcore/adapter.tiny_qwen3`, 4층 hidden-64)를 쓰는 경로와, 합성 attention(payload 창에 질량 집중)을 내는 순수 numpy 경로 두 가지.

### 4.4 엔진 어댑터 (`ag/systems/adapters/base.py`)

```python
@dataclass(frozen=True)
class EngineEvent:
    kind: Literal["prefill_done", "tokens", "ledger_step", "finished", "error"]
    t_ns: int
    t_source: Literal["engine", "client"]
    token_ids: tuple[int, ...] = ()
    ledger: Mapping[str, Any] | None = None   # per-step block mass increment (5.3절 LedgerSink 출력)
    error: str | None = None

class EngineAdapter(abc.ABC):
    def start(self, cfg: "EngineConfig") -> None: ...       # launch, poll /health with timeout, verify patch state
    def stream(self, rp: RenderedPrompt, sampling: "SamplingConfig") -> Iterator[EngineEvent]: ...
    def pause(self, request_id: str) -> None: ...           # optional (A-09); raise NotImplementedError
    def flush_cache(self) -> None: ...
    def stop(self) -> None: ...
    def info(self) -> Mapping[str, Any]: ...                # engine_version, attn_backend, patch_sha, shim_active
```

구현: `MockAdapter`(S1), `ReplayAdapter`(Record를 시간축으로 재생, offline 전용, S3), `SGLangAdapter`(patch + shim, S3E), `VLLMAdapter`(선택, S8g). 무수정 SGLang은 `SGLangAdapter(patch=False)`이고 `ledger_step`을 내지 않는다. `start`는 서버 로그에서 shim 적용 줄을 찾아 `shim_active`를 정하고, B0에서 그 줄이 있으면 실패한다.

### 4.5 ledger와 선택 (`ag/selector/`)

엔진 shim은 수식 없이 값만 넘긴다. 수식은 참조 구현과 엔진이 같은 함수를 쓴다(복제본 금지).

```python
class LedgerSink(Protocol):                                   # implemented in ag/selector/ledger.py
    def on_decode_step(self, request_ids: Sequence[str], layer_id: int,
                       lse_split: "Tensor[B, H, S]", lse_total: "Tensor[B, H]",
                       split_tokens: int) -> None: ...
    def on_request_end(self, request_id: str) -> None: ...

def block_mass_from_lse(lse_split, lse_total, *, log_base: float = math.e) -> np.ndarray: ...
    # m_b = base**(lse_b - lse_total); FlashInfer uses base 2 (plan 4.2)

@dataclass(frozen=True)
class SelectorConfig:
    rule: Literal["topk", "mass_threshold", "all", "oracle", "random", "truncate"] = "topk"
    k: int = 8
    mass_tau: float | None = None
    layers: tuple[int, ...] | Literal["upper_half", "last", "tracked8"] = "tracked8"
    head_weighting: Literal["uniform", "learned"] = "uniform"
    weights_path: str | None = None       # learned layer-head weights (aggregate B, plan 8.2)
    rollout: bool = False                 # plan 4.3; E6.1 measures its effect
    first_token: Literal["include", "skip"] = "skip"
    truncate_n: int | None = None         # N6: first and last N windows
    seed: int = 0

def chunk_block_mass(rec: "LedgerRecord", chunk: ChunkInfo, cfg: SelectorConfig) -> np.ndarray: ...  # [NB]
def apply_rollout(rec: "LedgerRecord", t_range: tuple[int, int], cfg: SelectorConfig) -> np.ndarray: ...
def window_scores_from_blocks(block_mass: np.ndarray, rp: RenderedPrompt) -> np.ndarray: ...       # [NW], ineligible=-inf
def select(window_scores: np.ndarray, cfg: SelectorConfig, *, passed: frozenset[int] = frozenset(),
           oracle: Sequence[int] | None = None) -> tuple[int, ...]: ...
```

- 집계 A(위쪽 절반 층, head 균등), B(학습 가중), C(위쪽 절반에서 고른 8개 층, head 균등)는 `layers`와 `head_weighting`의 조합이다. 게이트는 C로 본다(계획서 8.2).
- `passed`는 이미 통과한 창이다(verdict cache). 선택에서 빼지 않고 judge 호출만 건너뛴다. 판정 창 수는 cache hit 포함·제외 둘 다 센다.

**창 점수원** — 선택 품질 비교(E3.3)와 N 변형을 위해 창 점수를 내는 방법을 바꿔 끼운다.

```python
class WindowScorer(abc.ABC):
    scorer_id: str
    def scores(self, req: Request, rp: RenderedPrompt, rec: "LedgerRecord | None",
               chunk: ChunkInfo | None) -> np.ndarray: ...      # [NW]
```

| scorer_id | 내용 | 쓰는 시스템 |
| --- | --- | --- |
| `ledger` | 타깃 attention 질량(기본) | AG, B-oracle 비교 기준 |
| `random` | 무작위(seed 고정) | N1 |
| `guard_attention` | guard 모델(Qwen3Guard)의 attention 상위 | N2 |
| `shallow_probe` | 얕은 층 injection probe 점수(LG 방식) | N3 |
| `last_layer` | ledger의 마지막 층만, head 균등 | N4 |
| `truncate` | 처음·끝 N개 창에만 점수 | N6 |
| `attntrace`, `agentwatcher_attr` | 두 번째 모델 귀속을 창으로 사상 | B3(E3.3 비교선) |
| `loo` | 창 하나씩 가리고 B0 재생성한 출력 변화 | leave-one-out(E3.3, 8k 부분집합) |

AttnTrace는 100단어 segment 단위로 점수를 낸다. segment → 128토큰 블록 사상표를 데이터 생성 때 저장하고, 창 점수는 그 창과 겹치는 segment 점수의 최댓값으로 둔다(SPEC 정의).

### 4.6 judge와 judge pool (`ag/judges/`, `ag/runtime/judge_pool.py`)

```python
@dataclass(frozen=True)
class JudgeInput:
    request_id: str
    chunk_id: int | None
    window_id: int | None
    task: str
    window_text: str
    chunk_text: str | None
    window_spec: WindowSpec

class Judge(abc.ABC):
    judge_id: str
    needs_task: bool
    needs_chunk: bool
    native_window: WindowSpec | None
    def score(self, batch: Sequence[JudgeInput]) -> list[float]: ...     # higher = more unsafe
    def info(self) -> Mapping[str, Any]: ...

class JudgePool:
    def __init__(self, judges: Mapping[str, Judge], thresholds: Mapping[str, float],
                 cache: "VerdictCache", scheduler: "Scheduler", ctx: RunContext): ...
    def submit(self, inputs: Sequence[JudgeInput], *, job_kind: JobKind,
               deadline_kind: DeadlineKind, deadline_ns: int | None) -> "Future[list[Verdict]]": ...
```

- 구현: `prompt_guard2`(J1, 창만 봄), `agentwatcher_monitor`(J2, 과업·창·첫 chunk를 봄; 계획서 8.2 E0.2 절차), `piguard`(J3), `datasentinel`(J4), `output_guard`(OG, Qwen3Guard-Stream 또는 Gen), `mock_judge`(지연·점수를 요청 매개변수로; 지연·누락 주입 가능).
- **verdict cache** 키: `(judge_id, sha256(window_token_ids), sha256(task) if needs_task, sha256(chunk_token_ids) if needs_chunk, threshold_id)`. 정책 `off | per_request | global`은 config_hash에 들어간다(D-011).
- JudgePool은 요청 간 batch를 만들고 `JOB_SUBMITTED/STARTED/DONE`을 `ctx.emit`으로 낸다.

### 4.7 유해 경로 (`ag/lg/`)

```python
@dataclass(frozen=True)
class HarmScanResult:
    window_scores: np.ndarray            # LG trunk scores over the LG grid
    selected: tuple[int, ...]            # router top-k (k=32)
    score: float; unsafe: bool
    t_scan_ns: int; t_deep_ns: int

class HarmGuard(abc.ABC):
    guard_id: str                        # "lg-0.6b", "sliding", "dense" (B6), "mock"
    def scan(self, req: Request, prompt_ids: Sequence[int]) -> HarmScanResult: ...
```

`lg-0.6b`는 기존 `lgs*` 코드를 import해 감싼다(계획서 4.1, CLAUDE.md 3.7). B1a의 유해 쪽(Qwen3Guard-Gen sliding)과 B6(sliding, dense)은 같은 인터페이스의 다른 구현이다.

### 4.8 release gate와 scheduler (`ag/runtime/`)

```python
@dataclass(frozen=True)
class GateAction:
    kind: Literal["release", "block"]
    request_id: str; chunk_id: int; reason: str | None

class ReleaseGate:
    def __init__(self, policy: "GatePolicy", clock: Clock): ...
    def on_chunk_generated(self, request_id: str, chunk: ChunkInfo, required_jobs: Sequence[str],
                           deadline_ns: int | None) -> list[GateAction]: ...
    def on_job_done(self, job_id: str, verdicts: Sequence[Verdict]) -> list[GateAction]: ...
    def poll(self, now_ns: int) -> list[GateAction]: ...      # deadline expiry -> block (fail-closed)

class Scheduler(abc.ABC):                                     # none | fifo | edf
    def submit(self, job: "GuardJob") -> None: ...
    def next_batch(self, now_ns: int, capacity: int) -> list["GuardJob"]: ...
```

- 엔진 안 gate hook(`engine_integration/sglang/shim/gate_hook.py`)은 `on_tokens`/`poll`만 부르고 위 클래스를 쓴다. hook이 아직 없을 때는 하네스 쪽 proxy(`ag/harness/openai_front.py`의 gate backend)가 같은 클래스를 쓴다. 어느 쪽이든 ledger는 엔진에서 와야 한다(구축 계획 S3).
- **마감**(계획서 4.5): `HARM_SCAN`·`HARM_DEEP`은 타깃 첫 토큰의 예상 시각, `INJ_JUDGE`·`OUTPUT_GUARD`는 chunk 생성 완료 시각 + release 예산, `TOOL_COMMIT`은 호출 생성 완료 시각 + commit 예산. 예산 값은 설정(`deadline_budget_ms`)이다.
- **길이 적응 정책**: 입력이 `short_input_threshold_tokens`보다 짧으면 hold 없이 생성 전에 모든 창을 판정한다. 임계값은 E4.7에서 정한다.

### 4.9 공격 성공 판정 (`ag/data/success_judge.py`)

```python
@dataclass(frozen=True)
class SuccessResult:
    attack_success: bool | None
    utility: bool | None
    judge: str                           # "<name>@<version>"
    raw: Mapping[str, Any]

class SuccessJudge(abc.ABC):
    def judge(self, req: Request, gen: Generation) -> SuccessResult: ...
```

- 원천별 원 판정기(AgentDojo `security`/`utility`, InjecAgent, BIPIA 등)를 먼저 쓰고, 없으면 LLM grader(Qwen3-8B, D-005)를 쓴다.
- **sink에 도달한 출력만** 판정한다(`text_delivered`, `tool_calls_delivered`). block된 chunk의 내용은 판정에 쓰지 않는다(계획서 8.1 ASR 정의).
- grader는 GPU job이므로 `run` 안에서 부르지 않고, 실행 뒤 `ag.metrics.score_outcomes(run_dir)` 단계에서 부른다. 결과는 `outcomes.jsonl`에 쓴다.

### 4.10 workload와 도착 과정 (`ag/data/workloads/`)

```python
class ArrivalProcess(abc.ABC):
    def times_ns(self, n: int, rate_per_s: float, seed: int) -> np.ndarray: ...
# PoissonArrivals (gamma cv=1), GammaArrivals(cv), TraceArrivals(timestamps_ms, arrival_scale, window_s, start_s)

@dataclass(frozen=True)
class WorkloadSpec:
    workload_id: Literal["W-QA", "W-Agent", "W-MT", "W-Len"]
    arrival: Literal["poisson", "gamma", "trace"]
    rate_grid: tuple[float, ...]         # req/s; trace workloads use arrival_scale_grid instead
    arrival_scale_grid: tuple[float, ...] = ()
    duration_s: float | None = None
    n_requests: int | None = None
    length_class: str | None = None      # W-Len
    attack_ratio: float = 0.01           # plan 5.5; 0 and 0.1 as sensitivity
    warmup_requests: int = 20
    seed: int = 0

class Workload(abc.ABC):
    def build(self, pool: "RequestPool", spec: WorkloadSpec, rate_or_scale: float) -> list[Request]: ...
```

- 도착 시각은 미리 계산한 뒤 마지막 도착이 `n / rate`가 되게 정규화하고 seed를 기록한다. 간격을 자르지 않는다(구축 계획 1.5절 반례 7).
- 요청 본문은 LC-Inject에서 길이 등급이 가장 가까운 요청으로 채운다. 출력 길이는 trace의 `output_length`(상한 `max_new_tokens`)이고 `ignore_eos = True`다.
- W-Agent는 도구 호출 뒤 다음 turn까지 tool 대기 시간 분포(Copilot trace 또는 TraceLab)만큼 하네스가 기다린다.
- 다중 GPU 구성(E4.3의 2·4장)은 하네스의 `ReplicaRouter`가 replica endpoint에 요청을 나눈다. 정책은 처리 중 요청이 가장 적은 replica다 \[가정 A-07\].

### 4.11 공격 생성기 (`ag/attacks/`)

```python
class AttackGenerator(abc.ABC):
    attack_id: str                       # gcg_joint | llm_attacker | decoy | fragment_k | pipeline_aware
    def generate(self, base: Sequence[Request], target: Target | None, cfg: Mapping[str, Any]) -> list[Request]: ...
```

white-box GCG 계열은 `HFReferenceTarget`의 미분 가능한 attention 질량을 쓴다(8k 이하). 결과 요청은 일반 요청과 같은 형식이라 나머지 경로는 바뀌지 않는다.

## 5. Record v1 (D-003)

### 5.1 파일

- `records/<sha1(request_id + target_id)[:16]>.npz` + `records/index.jsonl`. run 디렉터리 안에 두거나, oracle 기록은 content-addressed 캐시(`$AG_OUT/cache/records/<key>/`)에 두고 run에서 참조한다. 캐시 키는 `canonical_hash(model_sha, data_manifest_sha, record_spec, code_commit의 reference 모듈 해시)`다.

### 5.2 npz 배열

| 이름 | dtype, shape | 뜻 |
| --- | --- | --- |
| `output_ids` | int32 [T] | 출력 토큰 |
| `t_ns` | int64 [T] | 토큰 생성 시각(참조 구현은 측정용 아님) |
| `layers` | int16 [L] | 기록한 층 id. 마지막 층은 항상 포함(N4) |
| `block_mass` | fp16 또는 fp32 [T, L, NB] | 토큰 t, 층 l의 head 평균 블록 질량. `Σ_b` + 생성 토큰 질량 = 1 |
| `gen_mass` | fp16 [T, L, T] | 토큰 t가 앞선 생성 토큰 s(< t)에 준 head 평균 질량. rollout용. 하삼각 |
| `chunk_head_mass` | fp16 [C, L, H, NB] | 표준 chunk 정의 C개(first8, first32, all) 각각의 head별·층별 평균 블록 질량. 학습 가중(집계 B)용 |
| `block_untrusted` | uint8 [NB] | 1 = 비신뢰 블록 |
| `prompt_len`, `split_tokens` | int | n, B |
| `first_token_included` | bool | t = 0 행이 prefill 마지막 위치의 질량인가(D-018) |

- head 평균은 Q head 기준이다(GQA에서 KV head를 공유해도 질량은 Q head마다 다르다).
- 크기 \[계산\]: T1 128k, T = 256, L = 32, NB = 1,008이면 `block_mass` fp16은 16.5 MB, `chunk_head_mass`는 6.2 MB, `gen_mass`는 4.2 MB다. 128k 300건이면 약 8 GB다. 8k·32k 전량(타깃당 약 6,000건)은 약 30 GB다. 쿼터는 `/home/212aig08` 쪽이다(CLAUDE.md 4절).
- **정밀도**: 기본 fp16. E2.1 차등 테스트 부분집합(8k 500, 32k 100, 128k 20)은 두 쪽 모두 fp32로 저장해, 양자화 오차가 허용치 1e-3을 먹지 않게 한다.

### 5.3 index.jsonl 행

`{request_id, target_id, producer: "mock"|"reference"|"sglang-shim"|"replay", schema_version: 1, file, sha256, T, prompt_len, n_blocks, layers, precision, first_token_policy, record_spec_hash, run_id, created_ns}`

### 5.4 쓰기와 읽기

```python
class RecordStore:
    def put(self, rec: "LedgerRecord") -> str: ...      # atomic (tmp + os.replace), returns file name
    def get(self, request_id: str, target_id: str) -> "LedgerRecord": ...
    def iter(self, **filters) -> Iterator["LedgerRecord"]: ...
```

참조 구현(S2), 엔진 shim(S3E), E2.3 분할별 기록(S3E), 16k/64k 기록(S4b), replay(S3)는 모두 같은 `put`을 쓴다.

## 6. 지표

모든 식은 `ag/metrics/`에 함수 하나씩 있고 이름은 `ag/metrics/names.py`의 enum에서 온다. 그 enum이 JSONL 키, `numbers.json` 키, 그림 파일명, `docs/metrics.md` 표의 유일한 원천이다.

### 6.1 품질 (계획서 8.1)

기호: A = 공격 요청 집합, N = 정상 요청 집합(같은 데이터·타깃), S = {r ∈ A : `attack_success_b0(r)` = 1}(방어를 끈 B0에서 성공한 공격). 오류가 난 요청(`error_kind ≠ None`)은 분모에서 빼고 따로 센다.

| 지표 | 정의 |
| --- | --- |
| ASR | `|{r ∈ A : attack_success(r)}| / |A|`. block된 요청은 성공 아님. 판정은 sink에 도달한 출력만 |
| FPR (문서) | `|{r ∈ N : blocked(r)}| / |N|`. chunk 하나라도 block되거나 요청이 block되면 1 |
| Utility | `|{r ∈ N : utility(r)}| / |N|`(방어 켬, block되면 0). 공격 요청의 utility는 `|{r ∈ A : utility(r)}| / |A|`로 따로 |
| hit@k (요청) | `|{r ∈ S : P(r) ∩ ⋃_c Top_k(r, c) ≠ ∅}| / |S|`. c는 표준 chunk 정의 중 하나(first8, first32, all)이며 정의별로 따로 낸다 |
| hit@k (chunk) | `|{(r, c) : r ∈ S, P(r) ∩ Top_k(r, c) ≠ ∅}| / Σ_{r∈S} |chunks(r)|`, 서빙 chunk 기준 |
| 무작위 기준 | `|P(r)| · k / NW_eligible(r)`의 S 평균(창 하나면 k/NW, 겹치는 두 창이면 약 2k/NW) |
| 문서 점수(offline) | 모든 창: `max_{w eligible} s(w)`. 역추적: `max_{w ∈ ⋃_c Top_k(r,c)} s(w)` |
| 판정 창 수 | `|⋃_c W_judged(r, c)|`(서로 다른 창), cache hit 제외 값도 함께 |
| recall | `|{r ∈ S : detected(r)}| / |S|`. offline은 `doc_score(r) ≥ τ`, live는 block 여부 |

**운영점**(계획서 8.1): 임계값은 calibration 분할에서만 정하고 test에서 고정한다.

- 운영점 R80: S_calib에서 recall이 0.80이 되는 가장 큰 τ를 고르고, test에서 FPR(문서)과 ASR을 낸다.
- 운영점 F1·F5: N_calib에서 문서 FPR이 1 %, 5 % 이하가 되는 τ를 `lgs1/metrics.strict_matched_fpr`로 고르고(달성 FPR ≤ 목표), test에서 ASR을 낸다.
- ROC 전체를 `summary.json`에 남긴다.

**통계**: 품질 수치에는 bootstrap 95 % CI(1,000회, 원천별 층화 재표집)를 붙이고, 두 시스템의 차이는 같은 요청 위의 paired bootstrap으로 낸다. 모든 비율에 분자·분모를 같이 쓴다.

### 6.2 지연 (D-004)

요청 r에 대해 이벤트에서 읽는 시각: `a_r` = ARRIVED, `q_r` = ADMITTED, `f_r` = PREFILL_DONE, `g_{r,j}` = CHUNK_GENERATED(j), `l_{r,j}` = CHUNK_RELEASED(j), `d_{r,j}` = CHUNK_DELIVERED(j), `V_{r,j}` = chunk j의 `required_jobs` 중 마지막 JOB_DONE 시각(없으면 +∞), `n_{r,j}` = chunk j의 토큰 수, J_r = delivered chunk 집합, `N_vis = Σ_{j∈J_r} n_{r,j}`.

| 지표 | 식 |
| --- | --- |
| TTFT(생성) | `f_r − a_r` |
| TTFV | `d_{r,0} − a_r` (클라이언트 수신, 모든 시스템 공통) |
| TTFV(엔진) | `l_{r,0} − a_r` (분해용) |
| release lag | `l_{r,j} − g_{r,j}`, chunk 단위 분포 |
| TPOT(보임) | `(d_{r,last} − d_{r,0}) / (N_vis − 1)`. delivered chunk가 하나뿐이면 정의하지 않음 |
| ITL(보임) | chunk j ≥ 1마다 `(d_{r,j} − d_{r,j−1}) / n_{r,j}`를 `n_{r,j}`번(SGLang 규칙) |
| 판정 전 노출 토큰 | `Σ_j n_{r,j} · 1[d_{r,j} < V_{r,j}]`. AG·B1a는 0이어야 함(E2.4), B1b·N5의 핵심 값 |
| gate 지연 | `l_{r,j} − max(g_{r,j}, V_{r,j})`(release된 chunk). E2.4 기준 ≤ 1 ms |
| E2E(보임) | `d_{r,last} − a_r` |
| 클라이언트 대기 | `q_r − a_r`. 시스템 지연에 섞지 않고 따로 보고 |

**식별식**(CPU 테스트로 고정):

1. `TTFV = TTFT(생성) + (g_{r,0} − f_r) + (l_{r,0} − g_{r,0}) + (d_{r,0} − l_{r,0})`
2. `d_{r,last} − d_{r,0} = Σ ITL(보임)`
3. release lag는 N5(hold 없음)에서 0, B1b에서는 release가 verdict보다 앞서므로 판정 전 노출 토큰 > 0

**집계**: P50, P90, P95, P99(`numpy.percentile`, linear)와 CDF. 실행 3회 반복의 중앙값을 내고 min–max 범위를 적는다. warm-up 요청(`warmup_requests`)은 뺀다. 처리량은 정상 상태 구간(활성 요청 수가 최대치의 일정 비율 이상인 가장 긴 구간)에서 잰다.

### 6.3 SLO, attainment, goodput (계획서 8.1, D-020)

- 길이 등급 `L(r)`: 입력 길이 [0, 4k) short, [4k, 16k) 8k, [16k, 64k) 32k, [64k, ∞) 128k.
- 기준값: `T95(L)` = B0를 전용 GPU에서 돌린 TTFT의 P95, `P95tpot(L)` = 같은 run의 TPOT P95. `configs/slo.yaml` 한 파일에 적고(원천 run_id 포함) runner와 plots가 같이 읽는다.
- 정규화 비율: `ρ_r = TTFV_r / T95(L(r))`, `π_r = TPOT_vis_r / P95tpot(L(r))`(TPOT 미정의면 0).
- 잘못 block된 정상 요청은 `ρ_r = +∞`. block된 공격 요청은 완료로 세고 `ρ_r = π_r = 0`. 오류·마감 실패는 `+∞`.
- **goodput(s)** = 요청률 sweep에서 `P90(ρ) ≤ s` 그리고 `P90(π) ≤ s`를 만족하는 최대 요청률. 인접 sweep 점 사이는 선형 보간한다. `s ∈ {1.1, 1.25, 1.5, 2.0}`. 길이 등급이 하나인 workload에서는 계획서의 정의(P90 TTFV와 P90 TPOT이 SLO 이하인 최대 요청률)와 같다.
- **attainment(λ, s)** = `mean_r 1[ρ_r ≤ s ∧ π_r ≤ s]`(요청 가중). 토큰 가중 값도 같이 낸다. Fig. 12의 99 % 선과 E4.4의 "99 % attainment에 필요한 GPU 수"는 이 값으로 정한다.
- `summary.json`에 길이 등급별로 해결된 절대 임계값(ms)을 같이 적는다.

### 6.4 오버헤드 (E2.3)

- TPOT 증가율 `(TPOT_X − TPOT_ref) / TPOT_ref`. ref는 (a) 수정 전 Triton backend, (b) 기본 backend(flashinfer) 두 가지(계획서 4.2). 같은 요청 집합, 같은 batch, 3회 반복.
- prefill 증가율, 요청당 추가 메모리(바이트), 커널 counter(Nsight Compute, 선택).

### 6.5 깊이 probe (E0.3, E1.1)

층 ℓ마다 AUROC와 창 FPR 2×10⁻⁴에서의 TPR(10⁻³, 5×10⁻⁵도 함께). `ℓ95` = TPR이 층별 최댓값의 95 %에 처음 닿는 층. 결과는 `probe_results.jsonl`(7.3절).

### 6.6 오탐 누적과 attention 집중 (E1.2, E1.3)

- 창 단위 FPR p, 길이별 문서 FPR, 독립 가정 곡선 `1 − (1 − p)^W`(W = 판정 창 수)와의 차이.
- 성공한 공격에서 payload 창의 질량 몫과 순위, 정상 요청에서 질량의 80 %를 차지하는 창 수.

## 7. 결과 디렉터리와 파일 스키마

### 7.1 배치

```
$AG_OUT/                                   # /home/212aig08/ag_results (env/ag.env)
  raw/<E>/<run_id>/
    lineage.json   resolved.yaml   cmd.txt
    events.jsonl   outcomes.jsonl  window_scores.jsonl   probe_results.jsonl   # only those that apply
    records/       summary.json    checks.json   server.log   engine_events.jsonl
  raw/<E>/latest.txt                       # run_id of the latest non-mock, non-dirty run
  raw/data/<data_manifest_sha>/{manifest.json, gate.json, requests.jsonl}
  cache/records/<key>/
  processed/figNN/<run_id>/*.csv
  figures/figNN.pdf   figures/manifest.json
  numbers.json
```

- `<E>`는 계획서 실험 번호 또는 작업 이름(`E0.1`, `W0-grader`, `table4`, `mock`).
- 쓰기는 모두 tmp 파일 + `os.replace`. JSONL은 줄 단위 append 후 flush.
- 같은 `run_id`로 다시 실행하면 기존 `lineage.json`과 비교해 resume 키가 다르면 exit 2로 끝나고, 같으면 `outcomes.jsonl`과 `events.jsonl`에 이미 있는 `request_id`를 건너뛴다.

### 7.2 lineage.json

```
{ run_id, exp_id, schema_version, code_commit, code_dirty, config_hash,
  env_hash: { pip_freeze_sha, env_vars_sha }, python, torch, transformers, engine_version, patch_sha, shim_active,
  gpu_name, gpu_count, gpu_driver, placement, hostname, slurm_job_id, slurm_array_job_id, slurm_array_task_id,
  started_monotonic_ns, started_unix_ns, finished_unix_ns, clock: "monotonic"|"virtual",
  model_sha: { target, guard, judge... }, data_manifest_sha, record_cache_keys, threshold_ids,
  producer, parent_run_ids, seed }
```

- resume 비교 키: `config_hash, code_commit, env_hash, engine_version, patch_sha, model_sha, data_manifest_sha, producer, hostname`. `lgs1/lineage.assert_resume_compatible`은 6개 키만 비교하므로 쓰지 않고 `ag/harness/lineage.py`에 자체 비교기를 둔다. `lgs1/lineage`의 `canonical_hash`, `dir_fingerprint`, `code_commit`, `RowsWriter`는 재사용한다.
- `code_dirty = true`인 run은 `latest.txt`에 오르지 않고 `plots/`가 기본값으로 거부한다.

### 7.3 행 파일

| 파일 | 행 | 키 |
| --- | --- | --- |
| `events.jsonl` | Event 하나 | 3.2절 필드 + `schema_version` |
| `outcomes.jsonl` | Outcome 하나 | 2.4절 |
| `window_scores.jsonl` | (요청, 점수원 또는 judge, 창) 하나 | `run_id, request_id, target_id, source_id, chunk_def, window_id, score, latency_ns, cached` |
| `probe_results.jsonl` | (backbone, 층, 음성 종류, seed) 하나 | `run_id, backbone, layer, rel_depth, task, negative_set, seed, auroc, tpr_at_fpr: {1e-3, 2e-4, 5e-5}` |

### 7.4 summary.json, checks.json, numbers.json

- `summary.json`: 지표 이름(names enum) → `{value, n, ci_low, ci_high, unit}`. 운영점별 임계값, ROC, 해결된 SLO 임계값, 시스템 info.
- `checks.json`: `[{name, ok, detail}]`. 예: `invariants_3_4`, `no_release_before_verdict`, `held_chunks_at_end == 0`, `judge_queue_empty`, `no_traceback_in_logs`, `shim_active_matches_system`, `fairness_knobs_equal`.
- `numbers.json`(초록 수치 7개, 계획서 9.3): `{ "abs1": {value, unit, ci, tag: "측정", source_run_ids, code_commit, target_id}, … }`. `plots/numbers.py`만 쓴다.

## 8. 설정 (YAML)

### 8.1 실험 파일 `configs/exp/<id>.yaml`

```yaml
exp_id: E0.1
plan_ref: "8.2 E0.1"
outputs: [Fig.9, abs3]
targets: [T1, T2]
systems: [B0]                       # names resolved from configs/systems/*.yaml
judges: []
data:
  manifest: lc_inject_v1            # resolved to raw/data/<sha>/
  sources: [agentdojo_l, injecagent, bipia, longpibench, piarena]
  length_classes: [8k, 32k]
  split: test
  n_attack: 600
  n_benign: 300
record: { layers: all, per_chunk_heads: true, gen_mass: true, first_token: true, precision: fp16 }
scorers: [ledger]
selector: { rule: topk, k: [1, 2, 4, 8, 16, 32], layers: [upper_half, tracked8], head_weighting: [uniform, learned] }
workload: null                      # closed loop
repeats: 1
seeds: [0]
fairness: { max_new_tokens: 256, temperature: 0.0, ignore_eos: false, dtype: bfloat16 }
slurm: { gres: "gpu:1", time: "24:00:00", array_axes: [targets, length_classes, shard] , shards: 8 }
```

- 리스트 값을 가진 키는 sweep 축이다. `ag/sweep/expand.py`가 곱집합을 펼쳐 job 하나당 `run_id` 하나를 만들고 `jobs.jsonl`에 쓴다. `slurm.array_axes`의 곱이 `--array=0-(N-1)`이 된다(`%N` throttle 금지).
- offline 재계산 축(k, 집계, chunk 정의)은 job을 늘리지 않고 한 run 안에서 계산한다.

### 8.2 시스템 파일과 overlay

```yaml
# configs/systems/b1a.yaml
base: ag.yaml
system_id: B1a
override:
  block_before_generation: true
  selector: { rule: all }
  judge: J1
  judge_windows: [ "128x2", "256x2" ]     # report the better of ours and native (plan 7.2)
  harm_guard: sliding
```

overlay는 `base`를 읽고 `override`를 깊게 병합한다. 결과 전체(`resolved.yaml`)가 config_hash의 입력이다.

### 8.3 config_hash와 공정성 knob

- `config_hash = canonical_hash(resolved − {notes, out_dir, slurm.time, slurm.partition})`(`lgs1/lineage.canonical_hash`).
- 한 실험 안 모든 시스템에서 같아야 하는 공정성 knob: `max_new_tokens, temperature, ignore_eos, dtype, kv_capacity_tokens, max_running_requests, chunked_prefill_size, attention_backend, cuda_graph, target revision`. 하네스가 실행 전에 비교하고 다르면 실패한다(`checks.fairness_knobs_equal`). 다르게 하는 것이 실험 축이면(E2.3의 backend) 실험 파일의 `fairness_exempt`에 적는다.

## 9. 시스템 카탈로그

### 9.1 AG 플러그인의 플래그 (`configs/systems/ag.yaml`)

B1a, B1b, B1c, B-full, B-oracle, N1–N6, AG 누적 ablation은 모두 AG 플러그인(`ag/systems/ag.py`)의 플래그 조합이다. 파일을 복사하지 않고 overlay로 만든다.

| 플래그 | 값 | 기본(AG) | 뜻 |
| --- | --- | --- | --- |
| `engine` | sglang \| replay \| hf_reference \| mock | sglang | 생성 경로 |
| `harm_scan` | off \| parallel \| blocking | parallel | 유해 경로를 prefill과 병렬로, 또는 생성 전에 |
| `harm_guard` | lg-0.6b \| sliding \| dense \| mock | lg-0.6b | 4.7절 |
| `output_guard` | off \| on | on | chunk 유해 판정 |
| `inj_path` | off \| on | on | injection 경로 전체 |
| `scorer` | 4.5절 scorer_id | ledger | 창 점수원 |
| `selector` | SelectorConfig | topk, k=8, tracked8, uniform | |
| `judge` | J1 \| J2 \| J3 \| J4 \| MJ | J1 또는 J2 | 실험마다 지정 |
| `judge_windows` | WindowSpec 목록 | ["128x2"] | |
| `block_before_generation` | bool | false | 선택된(또는 모든) 창을 생성 전에 판정, 통과해야 생성 |
| `judge_all_before_first_release` | bool | false | 생성은 바로 하되 첫 chunk는 모든 창 판정 뒤 release(B1c) |
| `release_before_verdict` | bool | false | 판정을 기다리지 않고 release(B1b, N5) |
| `overlap` | bool | true | 판정이 다음 chunk 생성과 겹침. false면 판정까지 생성 중지(D-024, A-09) |
| `verdict_cache` | off \| per_request \| global | per_request(품질) / global(성능) | D-011 |
| `scheduler` | none \| fifo \| edf | edf | 마감 순 |
| `window_budget` | int \| null | null | 요청당 판정 창 예산, 넘으면 hold 증가 |
| `short_input_threshold_tokens` | int \| null | null(E4.7에서 정함) | 길이 적응 정책 |
| `chunking` | {first, next, sentence} | {8, 32, false} | 1.4절 |
| `deadline_budget_ms` | {release, commit} | 실험 설정 | 4.8절 |
| `ledger_export` | off \| sync \| async | async | E2.3 측정 모드 |
| `split_tokens`, `traced_layers` | int, list | 128, 위쪽 절반에서 8개 | |
| `placement` | {guard, judge}: same \| gpuN | same | E4.6 공존 대 전용 |

검증 규칙: `release_before_verdict`와 `block_before_generation`은 함께 켤 수 없다. `selector.rule = oracle`은 `offline` 또는 평가 전용 run에서만 허용한다(정답 창을 쓰므로).

### 9.2 시스템별 매핑 (계획서 7.2)

| ID | 구현 | 모드 | 핵심 설정 | 내는 이벤트(요지) | 쓰는 실험 |
| --- | --- | --- | --- | --- | --- |
| B0 | `b0_engine` (무수정 SGLang) 또는 `hf_reference` | live | patch 없음 | PREFILL_DONE·TOKENS(client), RELEASED=DELIVERED | 전부. S의 정의(attack_success_b0) |
| B-oracle | AG overlay | offline 또는 live | `selector.rule = oracle` | 선택 창 = P(r) | E3.1 |
| B-full | AG overlay | offline(품질), live(지연) | `selector.rule = all`, `judge = J2`, `block_before_generation` | 모든 창 INJ_JUDGE | E0.2, E1.2, E1.5, E3.1, E3.4, E4.2 |
| B1a | AG overlay | offline(품질), live(지연) | `block_before_generation`, `selector.rule = all`, `judge = J1`, `harm_guard = sliding`, 창 두 체계 | 생성 전 모든 JOB_DONE 뒤 PREFILL_START | E0.2, E1.2, E3.1, E3.4, E4.1–E4.5 |
| B1b | AG overlay | live | `release_before_verdict`, `selector.rule = all` | RELEASED가 JOB_DONE보다 앞섬 | E4.2(노출 토큰) |
| B1c | AG overlay | live | `judge_all_before_first_release`, `selector.rule = all`, `scheduler = none` | 생성과 판정이 같은 GPU에서 경쟁 | E3.0(b), E4.1–E4.3 |
| B2 | `b2_pishield` | live | Llama-3.1-8B 13층 선형 probe, 생성 전 | JOB(PROBE) → REQUEST_BLOCKED 또는 B0 경로 | E3.0, E3.1 |
| B3 | `b3_attntrace`, `b3_agentwatcher` | live | B0로 전체 답 생성 → 두 번째 모델 귀속 → judge → 한 번에 release | JOB(ATTRIBUTION), JOB(INJ_JUDGE), 끝에 RELEASED | E1.5, E3.0, E3.1, E3.3, E4.1–E4.4 |
| B4 | `b4_pisanitizer`, `b4_rennervate` | live | 입력 변환(SANITIZE) 후 생성 / 첫 32토큰 attention 탐지 | JOB(SANITIZE) 또는 JOB(INJ_JUDGE) | E3.0, E3.1 |
| B5 | `b5_melon` | external | AgentDojo pipeline 안에서 실행, pair JSON을 읽음 | 없음(Outcome만) | E3.0, E3.1(AgentDojo 원래 길이) |
| B6 | HarmGuard 구현(sliding, dense) | live | 유해 경로 대조 | JOB(HARM_SCAN) | E3.5 |
| N1 | AG overlay | offline | `scorer = random` | | E3.3 |
| N2 | AG overlay | offline | `scorer = guard_attention` | | E3.3 |
| N3 | AG overlay | offline | `scorer = shallow_probe` | | E3.3, E5.1(fallback) |
| N4 | AG overlay | offline | `scorer = last_layer` | | E3.3 |
| N5 | AG overlay | live | `release_before_verdict` | 노출 토큰 > 0 | E4.2 |
| N6 | AG overlay | offline | `scorer = truncate`, `truncate_n` | | E3.1 |
| AG+J1, AG+J2 | `ag` | live(서빙), offline(품질 재계산) | 기본 | 전부 | E0.2 이후 전부 |
| AG-abl1–6 | AG overlay | live | 1 blocking 모든 창 → 2 + scan 병렬 → 3 + 역추적 선택 → 4 + verdict cache → 5 + 겹치기 → 6 + 마감 순 scheduler | | E4.5 |
| leave-one-out | `scorer = loo` | live(재생성) | 8k 부분집합 | | E3.3 |

- 외부 모델을 쓰는 baseline(B2, B3, B4)은 추가 pass 시간과 GPU 메모리를 `JOB_DONE.payload`(`extra_pass_ms`, `gpu_mem_mib`)로 낸다.
- B3 AttnTrace는 별도 venv에서 돈다(D-001). 플러그인은 B0 출력과 입력을 `out_dir/ext/attntrace/in.jsonl`에 쓰고 `python -m ag.ext.attntrace_worker`를 그 venv로 실행해 `out.jsonl`을 읽는다. 원 코드의 `attribute(question, [context], answer)`를 그대로 import한다(재작성 금지).
- **Table 4 이식 검증**(계획서 7.1): `configs/table4/<baseline>.yaml`에 원 설정을 고정하고, 원 스크립트 실행과 플러그인 실행을 같은 데이터로 돌려 `raw/table4/<baseline>/<run_id>/table4_row.json`(원값, 재현값, 플러그인값, 차이, 통과 여부)에 쓴다. 기준: 품질 ±3 pp, 지연 ±20 % 또는 B0 대비 배수(D-015), 원 스크립트 대 플러그인 품질 차이 ≤ 1 pp.

## 10. Workload 카탈로그 (계획서 5.5)

| ID | 도착 | 길이·출력 | 본문 | sink | 쓰는 실험 |
| --- | --- | --- | --- | --- | --- |
| W-QA | Mooncake conversation trace의 timestamp × `arrival_scale` | trace의 input/output 길이 | 길이가 가장 가까운 LC-Inject QA·요약 요청, 공격 비율 1 % | stream | E4.2–E4.5, E4.7 |
| W-Agent | Mooncake toolagent trace + tool 대기 시간 분포 | trace 길이 | AgentDojo-L·InjecAgent 요청 | tool | E4.3, E4.4 |
| W-MT | Mooncake `hash_ids`로 prefix 공유 구조 재현 | trace 길이 | 같은 긴 문서에 대한 여러 turn(episode) | stream | E4.4(W-MT 열, 선택), E6.4 |
| W-Len | Poisson, `rate_grid` | 8k / 32k / 128k 고정 | LC-Inject 해당 길이 | stream | E2.4, E4.2, E4.3, E4.6, E4.7, E5.3 |

- 128k 요청은 Mooncake trace에 없으므로 W-Len으로만 낸다 \[측정: clone한 trace에 128k 이상 요청 0개\].
- 공격 비율 기본 1 %, 민감도 0 %와 10 %(계획서 5.5).
- `hash_ids` → 토큰 확장은 id별 결정적 RNG(`numpy.random.default_rng(seed ^ hash_id)`)로 같은 id가 같은 토큰 블록이 되게 한다. Python 내장 `hash()`는 쓰지 않는다.

## 11. 실험별 적용 확인

계획서의 33개 실험과 실험 번호 없는 작업 5개가 위 인터페이스로 표현되는지 확인한 표다. "모드"는 4.2절의 live/offline/external, "probe"는 창 단위 학습, "bench"는 엔진 microbenchmark다.

| 실험 | 모드 | 시스템·구성 요소 | 입력 | 산출 파일 | 지표(6절) |
| --- | --- | --- | --- | --- | --- |
| W0-grader | live(B0 생성) + 사람 라벨 | B0(`hf_reference`), SuccessJudge(grader) | pilot 200 | outcomes, `grader_agreement.json` | 일치율 |
| W0-mooncake | 분석 | `traces/mooncake.py` | Mooncake trace | `trace_stats.json` | 길이 분포 |
| E0.1 | live(참조) + offline | B0 `hf_reference` + Record, scorer `ledger`, selector sweep | LC-Inject 8k·32k, 128k 100 | records, outcomes(hit) | hit@k 요청·chunk, 무작위 기준 |
| E0.2 | offline | judge J1·J2 위 offline runner, B1a·B-full·AG의 offline 판정 | E0.1 records | window_scores, outcomes | R80 FPR, ROC, 판정 창 수 |
| E0.3 | probe | `ag/probes/depth_probe.py`, `Target.hidden_states` | 창 데이터(5.4) | probe_results | AUROC, TPR@FPR, ℓ95 |
| E0.4 | live(무수정 엔진) + bench | B0 `b0_engine`, HarmGuard 시간, judge 8창 시간 | 8k·32k·128k | events, summary | prefill vs scan, judge vs chunk 생성 |
| E0.1-128k | live(참조) + offline | E0.1과 같음 | 128k 300 | records, outcomes | hit@8 |
| E1.1 | probe | E0.3 확장(backbone 4, seed 5) | 음성 창 20만 | probe_results | 6.5 |
| E1.2 | offline | judge J1·J2·J3 모든 창 | 정상 1,000 × 8 길이 | window_scores | 6.6 |
| E1.3 | offline | Record 분석 | E0.1·E0.1-128k records | summary | 6.6 |
| E1.4 | live(엔진) | AG + gate, Gantt 추출 | 3 길이 | events | 시작·끝 시각 |
| E1.5 | offline 합성 | 모든 창 J1·J2 시간 + B0 시간, B3 원본 | 3 길이 × 50 | summary | 요청당 지연, GPU 초 |
| E2.1 | bench + 차등 | `SGLangAdapter(ledger)` vs `HFReferenceTarget(forced_output)` | 8k 500·32k 100·128k 20 | records(fp32), `diff_report.json` | 최대 절대 오차, 상위 8 일치율 |
| E2.2 | bench | 같은 분할 누산 켬/끔, 기본 분할 vs 128 고정 | 500 × 3 길이 | `token_diff.json` | 토큰 일치율, 최대 logit 차 |
| E2.3 | bench | `ag/bench/tpot_overhead.py`, 분할 × 층 × 길이 × batch | 공격 부분집합 | events, records(분할별) | 6.4 |
| W2-e2e-smoke | live | AG+J1 전체 경로 | AgentDojo 1 + 32k 1 | events, checks | 불변식 |
| E2.4 | live(CPU 계약 + GPU) | AG + `mock_judge`(지연·누락 주입), W-Len | ≥ 10⁵ chunk | events, checks | 판정 전 방출 0, gate 지연 |
| E3.0 | live + 원 스크립트 | B2–B5 플러그인, B1a·B1c 속도 | 원 설정 | table4_row | 9.2절 기준 |
| E3.1 | offline + live | B0, B1a, B-full, B2, B3, B4, B5, N6, AG+J1, AG+J2, B-oracle | 타깃 3 × 원천 5 × 길이 3 | outcomes, window_scores, events | ASR, FPR, Utility, 판정 창 수, CI |
| E3.2 | offline + live | AG의 `harm_scan`·`output_guard`·`inj_path` on/off 조합, B3 | 유해 2종, injection 6종, 혼합 300 | outcomes | 단계별 탐지율·오탐 |
| E3.3 | offline (+ loo live) | scorer: ledger(균등·학습, rollout 유무), N1–N4, attntrace, agentwatcher_attr, loo | E3.1 records | window_scores, outcomes | hit@k, 질량–정답 상관 |
| E3.4 | offline | B1a·B-full vs AG, judge별 | 8k–128k 5 단계(16k·64k 기록 포함) | window_scores | 같은 ASR에서 FPR |
| E3.5 | live | HarmGuard lg-0.6b(AG 안) vs B6 | LG v2 데이터 | outcomes, events | R@128k, scan 시간 |
| E3.6 | live(참조) + offline | B0, AG+J1/J2 | 데이터 주입 합성 | outcomes | hit@k, judge 정확도 |
| E4.1 | live | B0, B1a, B1c, B3, AG+J1, AG+J2 | 3 길이 | events | 단계별 ms |
| E4.2 | live | B0, B1a, B1b, B1c, B3, B-full, N5, AG+J1, AG+J2 | W-Len, W-QA | events | TTFT, TTFV, release lag, TPOT, 노출 토큰 |
| E4.3 | live | B0, B1a, B1c, B3, AG+J1 × GPU 1·2(·4) | W-QA, W-Agent, W-Len × 6 요청률 | events | attainment, goodput(s) |
| E4.4 | live | B0, B1a, B3, AG+J1, `ReplicaRouter` | Mooncake trace 20분 구간 | events | attainment 시계열, 99 %에 필요한 GPU 수 |
| E4.5 | live | AG-abl1–6 (+B1a) | W-QA | events, outcomes | 단계별 goodput, release lag, FPR |
| E4.6 | live | B0, AG+J1/J2, `placement` same vs gpu1 | W-Len | events | prefill·TPOT 변화 |
| E4.7 | live | B0, AG+J1/J2, 길이 적응 정책 | 1k–4k, 요약, 정상 고부하 | events, outcomes | TTFV 증가, 판정 창 수 분포 |
| E5.1 | live(참조) + offline | AttackGenerator(gcg_joint, llm_attacker, decoy), AG, N3 fallback | 8k(white-box), 32k(black-box) | outcomes, records | 무방어 ASR 대 payload 순위, 방어 시 ASR |
| E5.2 | live | AttackGenerator(fragment_k) + 두 경로 | 2·4·8 조각 | outcomes | 탐지율, ASR |
| E5.3 | live | AttackGenerator(pipeline_aware) + W-Len | 판정 폭주, 한 chunk 유출, 과업 정렬 | events, outcomes | 최악 release lag, 판정 전 방출 |
| E6.1 | offline (+ live 부분집합) | selector·chunking·split·rollout·첫 토큰·규칙 sweep | E2.3·E3.1 records, 32k 50 live | outcomes, events | hit@k, FPR, ASR, release lag |
| E6.2 | offline | judge J1–J4 모든 창 vs 역추적 | E3.1 records | window_scores | FPR, ASR, GPU 초 |
| E6.3 | live | AG+J1/J2, T4(선택), LLMail-Inject | 보지 않은 데이터 | outcomes | hit@k, ASR(LLMail은 FPR 없음) |
| E6.4 | live(선택) | `VLLMAdapter`, W-MT | | events | TPOT 오버헤드, cache hit TTFV |

**확인 결과**: 38개 모두 위 인터페이스로 표현된다. E3·E4의 baseline 중 엔진 밖 모델을 쓰는 B2·B3·B4는 live 플러그인이 B0 엔진 호출과 추가 job을 감싸는 방식, B5는 external 모드, B1a·B1b·B1c·B-full·B-oracle·N1–N6은 AG 플래그 overlay다. workload 4종은 `Workload` + `ArrivalProcess` + `ReplicaRouter`로 덮인다. 인터페이스가 아니라 엔진 기능에 기대는 곳은 세 군데이고 12절 가정으로 적었다: 엔진 쪽 이벤트 전달(A-05), 요청 단위 pause(A-09), TP-2에서 ledger all-gather(A-10).

## 12. 가정

SPEC이 기대는 미확인 사실이다. 깨지면 DECISIONS에 적고 묻는다(CLAUDE.md 3.6).

| 번호 | 가정 | 근거·출처 | 깨지면 |
| --- | --- | --- | --- |
| A-01 | 하네스·엔진·judge pool은 한 Slurm job, 한 노드에서 돌고 `monotonic_ns`를 공유한다 | 설계 선택 | 다른 노드면 run 시작 시 offset을 재 lineage에 기록하고 보정 |
| A-02 | 신뢰·비신뢰 경계를 애플리케이션이 알려 준다(chat role로 전달) | 계획서 3.1, 11.2 | 범위 밖. 논문 한계로 기재 |
| A-03 | 타깃 tokenizer가 offset mapping을 제공한다(fast tokenizer) | T1·T2·T3 HF tokenizer \[추정\] | char→token 사상을 재토큰화 비교로 대체 |
| A-04 | 무수정 SGLang(B0)의 첫 토큰 수신 시각을 TTFT(생성)로 쓴다(서버 시각 없음) | 계획서 4.2(API 없음) | B0의 TTFT(생성)는 클라이언트 기준임을 표기 |
| A-05 | 엔진 shim이 이벤트를 spool 파일 또는 로컬 socket으로 하네스에 넘길 수 있다 | F-07 관련 | 하네스 쪽 gate proxy로 대신(구축 계획 S3) |
| A-06 | open loop 요청당 스레드 하나로 충분하다(128k에서 동시 요청 수 작음) | 계획서 5.5(128k 1개/GPU) | asyncio로 하네스 클라이언트만 교체, 인터페이스 유지 |
| A-07 | 다중 GPU에서 replica 간 라우팅은 처리 중 요청 최소 정책 | 설계 선택 | 계획서에 없음. 결과에 정책 표기 |
| A-08 | SGLang Triton decode의 `attn_lse`를 CUDA graph 켠 상태에서 step마다 읽을 수 있다 | F-05 | eager 모드 또는 PyTorch 재계산(계획서 4.2 대안), 오버헤드 따로 보고 |
| A-09 | "겹치기 끔"(E4.5 5단계 전)은 요청 단위 pause 또는 chunk 단위 이어 생성으로 구현할 수 있다 | D-024 | chunk마다 prefix cache를 재사용하는 새 요청으로 이어 생성 |
| A-10 | T3(GPU 2장, TP-2)에서 ledger를 head 축 all-gather 한 번으로 모을 수 있다 | D-021 | T3는 8k·32k만 또는 참조 경로 |
| A-11 | 128토큰 분할의 TPOT 비용이 기본 설정에서 +2 % 안이다 | F-06 | 계획서 10.2 G1 fallback(sampling, 2단계 분할) |
| A-12 | Mooncake `hash_ids`는 512토큰 블록, prefix 포함 해시다 | F-11, FAST25 README \[문헌\] | W-MT 재현 방식 수정 |
| A-13 | AgentDojo 0.1.35가 3.11 venv에서 SGLang과 함께 설치된다 | F-15 | AgentDojo만 별도 venv, pair JSON 경계 |
| A-14 | LG 기존 코드(`lgs*`)가 3.11 venv의 torch·transformers 버전에서 그대로 import된다 | D-023, requirements 머리말(3.11 시험) \[문헌\] | 유해 경로를 별도 프로세스로 두고 HTTP 경계 |

## 13. 환경 (D-001)

- AG 패키지 `ag`는 `pyproject.toml`의 `requires-python = ">=3.11"`. 주 venv(제안 경로 `/home/212aig08/ag_venv/ag311`) 하나에 하네스, SGLang 0.5.21, FlashInfer 0.7.0.post1, transformers, AgentDojo 0.1.35, judge 모델 의존성을 설치한다. 버전은 실험 내내 고정하고 `lineage.env_hash`에 남긴다.
- AttnTrace는 별도 venv(Python 3.10.15, torch 2.5.1, flash-attn 2.7.0.post2 고정 \[문헌: environment.yml, requirements.txt\])에 두고 9.2절의 worker 경계로만 부른다.
- 기존 3.9 venv(`/home3/212aig08/servsec/exp/secu`)는 LG 기존 결과 재현용이다. AG 코드는 쓰지 않는다.
- 계산 노드는 외부 네트워크가 없다. 모델·데이터·wheel은 로그인 노드에서 미리 받고 `HF_HUB_OFFLINE=1`(CLAUDE.md 4절).
- CPU 테스트는 같은 3.11 venv(또는 CI의 3.11)에서 `python -m pytest tests -x -q`와 `pytest $(cat tests/manifests/cpu.txt)` 두 명령으로 돈다. GPU 테스트는 `tests/gpu/`에 두고 CUDA가 없으면 skip한다.

## 14. 변경 규칙

- 데이터 클래스, 이벤트, Record, 결과 파일의 필드를 바꾸면 `schema_version`을 올리고 이 문서를 먼저 고친다. 읽는 쪽은 모르는 버전이면 실패한다.
- 이벤트 종류를 더하는 것은 하위 호환이다(지표 함수는 모르는 종류를 무시한다). 빼거나 의미를 바꾸면 버전을 올린다.
- 지표 식을 바꾸면 `docs/metrics.md`와 식별식 테스트를 같은 PR에서 바꾼다.
- 계획서와 다른 정의(이 문서에서 "SPEC 정의"로 표시한 것: payload가 긴 경우의 P(r), 창 자격, segment → 창 사상, 혼합 workload의 goodput 정규화, replica 라우팅)는 DECISIONS에 있는 항목과 연결된다.
