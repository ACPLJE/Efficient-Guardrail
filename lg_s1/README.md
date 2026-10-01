# lg_s1 — Landmark Guard 계획서 단계 S1 실행 패키지

계획서 §2의 **S1(검증 장치·자산 정리·다운로드)** 8개 작업을 끝내는 코드입니다. 날짜와 무관하게 아래 순서대로 돌리면 S1의 산출물이 전부 `$LG_OUT`에 생깁니다.

| 계획서 | 이 패키지 | 어디서 |
| --- | --- | --- |
| D1 경로 확인 | `python s1ctl.py assets` | 로그인 노드 |
| D2 쿼터 | `python s1ctl.py quota` (dry-run) → `--apply-move` | 로그인 노드 |
| E0-1 등가성 6건 | `tests/test_correctness.py` | GPU 1장 (`s1ctl.py tests --sbatch`) 또는 CPU |
| E0-2 계보 해시·resume 중단 | `lgs1/lineage.py` + `tests/test_lineage.py` | 같은 잡 |
| E0-3 유일성·달성 FPR·hard-fail | `lgs1/metrics.py` + `tests/test_metrics.py` | 같은 잡 |
| D3 다운로드 (revision 고정) | `bash slurm/s1_download_login.sh` | **로그인 노드만** (계산 노드는 외부망 차단) |
| E0-6 zero-shot 12종 ±1pt | `python s1ctl.py zeroshot --sbatch` → `python s1ctl.py compare` | GPU B (job array) |
| M1 교수님 확인 | `python s1ctl.py m1` | — |

## 0. 설치 (한 번)

```bash
cd /home/212aig08 && unzip lg_s1.zip && cd lg_s1
vi env/s1.env                      # 경로·파티션 확인 (LG_OUT, LG_SOURCES, LG_V04_ROOT, LG_ROUND3_ROOT, LG_PARTITION)
$LG_PYTHON -m pip install -r requirements.txt    # v0.4 venv에 설치 (torch/transformers는 이미 있으면 유지)
python s1ctl.py tests              # CPU에서도 5초: 69개 테스트가 green이면 설치 끝
```

`env/s1.env` 값은 계획서·메모리의 경로를 그대로 넣어 두었고, 실제 트리와 다르면 이 파일만 고치면 됩니다. 모든 스크립트가 이 파일을 읽습니다.

## 0.5 저장 위치 결정 (9/17 quota 결과 반영)

`quota -s` 결과 `/home3`(sdc1)는 **500G 하드리밋에 328G 사용(여유 172G)**, `/home`(NFS)은 6000G 중 1331G 사용입니다. HF 캐시 145G와 S1·S2·S3 산출물을 `/home3`에 두면 D3(모델 10종 ≈145G 추가)에서 한도에 걸립니다. 그래서 새로 만드는 것은 전부 `/home`에 둡니다:

```bash
# env/s1.env 세 줄만 바꿈
export LG_OUT=/home/212aig08/lg_s1_out
export LG_SOURCES=/home/212aig08/lg_data_v2/sources
export HF_HOME=/home/212aig08/.cache/huggingface          # 이미 있는 디렉터리(LG3·Qwen3Guard 복사본 49G)

# /home3 캐시를 /home으로 합치고(중복은 건너뜀), /home3 쪽은 심볼릭 링크로 남겨 다른 프로젝트 경로를 유지
rsync -a --ignore-existing /home3/212aig08/.cache/huggingface/hub/ /home/212aig08/.cache/huggingface/hub/
mv /home3/212aig08/.cache/huggingface /home3/212aig08/.cache/huggingface.bak_0917
ln -s /home/212aig08/.cache/huggingface /home3/212aig08/.cache/huggingface
mv /home3/212aig08/lg_s1_out /home/212aig08/lg_s1_out                    # 이미 만든 보고서 이동
python s1ctl.py quota            # 이제 /home3 .bak 쪽이 duplicate 후보로 잡힘 → 확인 후 --apply-move
```

`.bak_0917`을 지우면 /home3에 145G가 돌아옵니다. 계산 노드는 `/home`도 마운트하므로(`lg_mlsys_v04_runs`가 거기서 돕니다) 잡에는 영향이 없습니다. 이미 제출한 잡은 예전 `$LG_OUT`이 sbatch 헤더에 박혀 있으니 `scancel` 후 다시 제출하는 편이 깔끔합니다.

## 0.6 patch1 (9/17)

`lg_s1_patch1.zip`은 다음만 바꿉니다(`unzip -o lg_s1_patch1.zip -d /home/212aig08/`, `env/s1.env`는 건드리지 않음):

* `s1ctl.py status`: D3를 레지스트리 43개 대비로 세고(이전엔 1개만 받아도 DONE), E0-6 ±1pt는 summary가 없으면 PENDING.
* `s1ctl.py doctor` 추가: venv 패키지, HF 캐시에 있는/없는 B1 가드, 여유 공간을 한 번에 확인.
* zero-shot 하네스: 벤치 파일을 HF 허브 캐시(`datasets--allenai--wildguardmix`, `datasets--lmsys--toxic-chat`)에서도 찾음; 모델이 캐시에 없고 오프라인이면 긴 traceback 대신 `[skip] … --only <name>` 안내와 exit 3.
* quota 후보의 `__pycache__` 사유 표기.

## 0.7 patch2 (9/17) — 다음 할 일을 CLI 8줄로

`lg_s1_patch2.zip`(`unzip -o lg_s1_patch2.zip -d /home/212aig08/`)에 `locate`·`env-set`·`--dep-last`와 `slurm/s1_relocate_home.sh`가 추가됐습니다. 순서:

```bash
cd /home/212aig08/lg_s1
python s1ctl.py locate --write            # 1. v0.4 코드·Round 3 디렉터리를 파일 이름으로 탐색, 유일하면 s1.env에 기록
bash slurm/s1_relocate_home.sh            # 2. 저장 위치 이동 dry-run (크기·여유·실행할 명령만 출력)
bash slurm/s1_relocate_home.sh --apply    # 3. /home3 HF 캐시 → /home 복사 + 심볼릭 링크, 보고서 이동, s1.env 갱신
python s1ctl.py doctor                    # 4. torch·transformers·pytest가 있는 python 목록, 캐시된 가드, 여유 공간
scancel 953593 953595                     # 5. 예전 경로로 제출된 잡 취소
python s1ctl.py tests --sbatch            # 6. E0-1/2/3
python s1ctl.py zeroshot --sbatch --dep-last   # 7. E0-6 array, 방금 제출한 tests 잡에 afterok
bash slurm/s1_download_login.sh           # 8. D3 (gated는 먼저 huggingface-cli login)
```

`doctor`에서 `python=python3`이거나 패키지가 `MISSING`이면 torch가 있는 인터프리터를 고릅니다: `python s1ctl.py env-set LG_PYTHON=/home3/212aig08/servsec/exp/secu/bin/python` 처럼 `doctor`가 `OK`로 표시한 경로를 넣고 `그 python -m pip install -r requirements.txt`. `locate`가 v0.4 코드를 못 찾으면(`landmark_guard.py` 없음) 서버에 v0.4가 풀려 있지 않은 것이므로 로컬 ZIP v0.4.0을 올려 풀고 다시 `locate --write`. Round 3 후보가 여러 개면 `env-set LG_ROUND3_ROOT=<dir>`로 직접 지정한 뒤 `assets --seal`.

## 0.8 patch4 (9/17) — D3 실패 9건의 원인과 수정

9건 중 7건(pg19·wildchat·stack·arxiv·wiki_en·wiki_zh·pubmed)은 제 버그였습니다: 스트리밍 항목은 `sources/<name>/docs.jsonl.gz`를 쓰는데 그 디렉터리를 `revision.json`을 쓸 때에야 만들고 있었습니다(`FileNotFoundError`). 나머지 2건은 HF id가 틀렸습니다. 모두 Hub API로 확인해 고쳤습니다:

| 항목 | 이전 | 수정 (2026-09-17 Hub에서 확인) |
| --- | --- | --- |
| polyguard_qwen_7b | ToxicityPrompts/PolyGuard-Qwen-7B (없음) | `ToxicityPrompts/PolyGuard-Qwen` |
| bingoguard_8b | Salesforce/BingoGuard-8B (없음) | `fanyin3639/bingoguard-llama-8b` (저자 공개, gated 아님) |
| gliguard | id 없음 | `fastino/gliguard-LLMGuardrails-300M` |
| pg19 | deepmind/pg19 (스크립트 데이터셋, datasets≥3 불가) | `emozilla/pg19` (parquet 미러) |
| arxiv | armanc/scientific_papers (스크립트) | `ccdv/arxiv-summarization` config `document`, 필드 `article` |
| longsafetybench | manual | `LutherXD/LongSafetyBench` (HarmfulNIAH·ManyShotJailbreak 등 parquet) |
| msj | manual | LongSafetyBench의 `ManyShotJailbreak` parquet로 대체 |

`slurm/s1_relocate_home.sh`는 D3가 돌고 있으면 시작을 거부하고(캐시 디렉터리를 옮기는 중 다운로드가 깨지는 것 방지), 옮긴 뒤 `/home3/.../lg_s1_out`을 심볼릭 링크로 남겨 이미 제출한 잡의 로그 경로가 그대로 유효합니다.

## 0.9 patch5 (9/17) — 파티션을 GPU 메모리 티어별로 자동 확장

한 파티션(RTX6000ADA)만 쓰던 것을 고쳤습니다. 이제 제출할 때마다 `sinfo`로 파티션별 GPU 모델·메모리·상태를 읽고, `scontrol show partition`의 AllowQos/DenyQos로 QOS `normal`이 허용되는 파티션만 남긴 뒤(조회가 안 되면 클러스터 메모의 목록 RTX6000ADA·A6000·L40S·A5000·RTX4090·A100-40GB-PCIe 사용), 잡이 필요로 하는 GPU 메모리 이상인 파티션을 전부 `--partition=A,B,C`로 넣습니다. Slurm은 그중 먼저 자리가 나는 파티션에서 잡을 시작합니다.

| 티어 | 대상 | 파티션(메모 기준; scontrol이 허용하면 RTX3090·A100-40GB 등도 자동 추가) |
| --- | --- | --- |
| 8 GB | tests, assets | ≥24 GB 목록 전부(QOS normal에 24 GB 미만 파티션이 없음) |
| 24 GB | 0.6B–9B 가드 10종 | RTX6000ADA, A6000, L40S, A5000, RTX4090, A100-40GB-PCIe |
| 40 GB | Llama Guard 4-12B | RTX6000ADA, A6000, L40S, A100-40GB-PCIe |
| 48 GB | gpt-oss-safeguard-20b (bf16 ≈42 GB) | RTX6000ADA, A6000, L40S |

`python s1ctl.py zeroshot --sbatch`는 티어별로 job array를 나눠 제출하고(24 GB 10개, 40 GB 1개, 48 GB 1개), `python s1ctl.py partitions`가 어떤 파티션이 왜 선택됐는지 보여 줍니다. 강제하려면 `env-set LG_PARTITIONS_OVERRIDE=RTX4090,A5000`.

## 0.10 (9/17) — D3 다운로드가 멈추면 (xet 끔 + 이어받기)

PolyGuard(≈15 GB, safetensors 4샤드)가 "Fetching 17 files: 10/17"에서 크기가 그대로 멈췄고, 죽이고 다시 켜자 캐시 크기가 11 GB → 2.4 GB로 떨어졌습니다. 원인은 huggingface_hub의 xet 백엔드: 로그인 노드에서 전송이 멈추는 데다 죽은 샤드를 이어받지 못해 매번 0부터 다시 받습니다(크기가 줄어든 것은 손상이 아니라 재시작 진행분).

* `slurm/s1_download_login.sh`가 이제 `HF_HUB_DISABLE_XET=1`로 실행 → 일반 HTTP 경로(cdn-lfs). 멈춰서 죽여도 `<sha>.incomplete`를 Range 요청으로 이어받음.
* `lgs1.download`는 fetch 전에 xet가 남긴 "전체 크기인데 구멍 난" `.incomplete`(sparse)를 지웁니다 — 일반 경로가 그걸 '다 받은 것'으로 착각해 손상 safetensors를 만들 수 있기 때문. 짧은 partial은 그대로 두고 이어받습니다.
* 진행 확인: `du -sm $HF_HOME/hub/models--ToxicityPrompts--PolyGuard-Qwen` 을 20초 간격으로 두 번 — 늘면 정상.

```bash
pkill -f lgs1.download
bash slurm/s1_download_login.sh --only polyguard_qwen_7b,bingoguard_8b,gliguard,longsafetybench,pg19,stack,arxiv
```

## 1. 실행 순서

```bash
# --- 로그인 노드 (gsai-login-3) ---
python s1ctl.py assets                     # D1: reports/assets_check_<MMDD>.md — FAIL 항목 확인
python s1ctl.py assets --seal              # Round 3 경로를 s1.env에 넣은 뒤: 읽기 전용 봉인
python s1ctl.py quota                      # D2: 후보 목록(dry-run) → 보고 확인 후
python s1ctl.py quota --apply-move         #     후보를 $LG_OUT/_to_delete/<MMDD>/ 로 이동(되돌릴 수 있음)
python s1ctl.py m1                         # M1: reports/M1_decisions.md 생성 → 결정 기입
huggingface-cli login                      # gated 저장소(Llama*, WildGuardMix, WildChat, Aegis) 라이선스 수락 후
bash slurm/s1_download_login.sh --list     # D3: 레지스트리 확인
bash slurm/s1_download_login.sh            #     전부 받기 (nohup, 로그: $LG_OUT/logs/download_*.log)

# --- 계산 노드 (sbatch) ---
python s1ctl.py tests --sbatch             # E0-1/2/3 (GPU A, 5분) → reports/pytest_e0.status
python s1ctl.py zeroshot --sbatch --dep <tests jobid>    # E0-6 12종 job array (GPU B, 순차 %1)
python s1ctl.py zeroshot --guards wildguard_7b --limit 32 --sbatch   # 한 모델 smoke
python s1ctl.py all --sbatch               # D1·D2·M1 즉시 + tests 잡 + zeroshot array(afterok)

# --- 집계 ---
python s1ctl.py compare                    # reports/e06_reproduction.md (±1pt PASS/FAIL, 파싱 실패율)
python s1ctl.py status                     # 8개 작업의 완료 여부 + squeue
```

`--dry-run`을 붙이면 생성된 sbatch 스크립트만 출력합니다. 스크립트는 `$LG_OUT/slurm/`에 남고 로그는 `$LG_OUT/logs/`에 쌓입니다. QOS `normal`은 `--gres=gpu:1`을 요구하므로 모든 잡이 GPU 1장을 잡습니다(D1·D2는 로그인 노드에서 직접 실행).

## 2. 산출물과 통과 기준

`docs/E0_pass_criteria.md`에 계획서 표와 1:1로 정리했습니다. 요약:

* `reports/pytest_e0.status` = `PASS` — E0-1(a)–(f), E0-2, E0-3, 파서·하네스 테스트 69건.
* `reports/assets_check_<MMDD>.md` — 항목 전부 OK, Round 3 `writable=no`.
* `reports/quota_<MMDD>.md` — 첫 블록의 quota 출력에서 여유 ≥150G.
* `$LG_SOURCES/download_status.json` — `failed=0`; `manual`(LongSafetyBench·NINJA·MSJ·AgentDojo)은 S3에서 처리.
* `reports/e06_reproduction.md` — 공개 수치가 있는 행 전부 PASS(|Δ|≤1pt). `n/a` 행은 `lgs1/zeroshot/published.yaml`에 논문 수치를 채워야 Table 3에 들어갑니다.
* `reports/M1_decisions.md` — 4건 결정, `DECIDED`.

## 3. E0-1 테스트가 검증하는 것 (tiny Qwen3, CPU 5초)

`tests/test_correctness.py`는 `lgs1/lgcore/adapter.py`가 내보내는 함수만 호출합니다. 기본값은 이 패키지의 참조 구현(`lgs1/lgcore/model.py`)이고, 이 구현이 곧 v0.4 코드가 만족해야 하는 사양입니다.

| 테스트 | 검증 내용 |
| --- | --- |
| (a) `test_a_*` | 수동 layer loop = HF forward (전체 깊이+norm, ℓ층 절단 = `hidden_states[ℓ]`, 패딩 포함) |
| (b) `test_b_*` | 윈도우 배치 block-local trunk = 윈도우 단독 실행 (4개 그리드) |
| (c) `test_c_*` | 윈도우 밖 토큰을 전부 바꿔도 해당 윈도우의 hidden·token score·window score 불변 |
| (d) `test_d_*` | token-max = 명시적 max, mean과 다름, 스파이크 토큰이 max에서만 1위, route()가 정렬된 top-k와 θ_k 반환, deep 단계는 길이와 무관하게 k개 |
| (e) `test_e_*` | stride 2b에서도 윈도우 길이 2b, overlap 그리드는 모든 b-span 포함, disjoint는 포함 실패 |
| (f) `test_f_*` | θ = route()의 k번째 점수(같은 스케일), needle 포함 윈도우 ≤3개, suffix 슬롯은 needle 토큰 점수를 못 바꿈, success = router 회피 AND end-to-end safe |

### v0.4 코드에 연결하기

`LG_V04_ROOT`가 설정되고 `landmark_guard`가 import되면 `adapter._wire_v04()`가 이름을 v0.4 함수로 바꿉니다. `make_windows`는 매핑 예시가 들어 있고, 나머지(`trunk_windows`, `pool_scores`, `route`, `LandmarkGuard`, `run_attack`)는 계약(모듈 docstring)에 맞춰 같은 블록에 추가하면 됩니다. 테스트 시작 시 `adapter: <name> <- v04|ref`가 출력되어 어느 쪽을 검증했는지 보입니다. 매핑 전에는 참조 구현이 검증되고, 매핑 후에는 v0.4가 같은 사양을 만족하는지 검증됩니다.

### run_all.sh 훅

`docs/E0_pass_criteria.md`의 두 줄(pytest 선실행 + `lgs1.lineage check`)을 v0.4 `run_all.sh`의 STAGE 분기 앞에 넣으면 "테스트 실패 시 실험 시작 금지"와 "계보 불일치 시 exit 2"가 모든 실험에 적용됩니다.

## 4. E0-2 / E0-3 사용법 (러너에 넣을 코드)

```python
from lgs1.lineage import Lineage, RowsWriter, assert_resume_compatible
from lgs1.metrics import assert_unique_ids, strict_matched_fpr, check_domains, summarize_matched

lin = Lineage.build(model_dir, router_path, cfg_dict, recipe_json_path, manifest_path, code_dir)
assert_resume_compatible(rows_path, lin)              # 불일치 → LineageMismatch (CLI는 exit 2)
with RowsWriter(rows_path, lin, resume=True) as w:
    w.write({"split": "test", "subset": "main", "id": rid, "L_target": 28672, "domain": "pg19", "score": s, "label": y})

rows = list(iter_rows(rows_path))
assert_unique_ids(rows)                               # (split, subset, id) 중복 → DuplicateIdError
check_domains(rows, ["govreport", "pg19", "wildchat", "stack", "arxiv"])   # 누락 → MissingDomainError
th = strict_matched_fpr(calib_benign_scores, 0.05)    # th.tau, th.achieved_fpr (항상 ≤ 0.05)
table = summarize_matched(test_rows, calib_rows)      # recall@5pct·1pct + calib/test 달성 FPR 열
```

## 5. E0-6 zero-shot 하네스

* 12개 B1 가드(`lgs1/zeroshot/guards.py`의 `B1_ORDER`) + 선택 4개. 각 어댑터는 프롬프트 빌더·파서·(가능하면) 첫 토큰 P(unsafe)를 갖고, `template_status`가 `verify`인 것은 모델 카드와 한 번 대조해야 합니다: `python -m lgs1.zeroshot.run --guard polyguard_qwen_7b --bench wgtest --show-prompt`가 row 0의 실제 프롬프트를 출력하므로 카드의 예시와 diff 하면 됩니다.
* 결과 행마다 raw 출력·파싱 라벨·score·decode 토큰 수·계보 해시가 남고, 같은 계보면 자동 resume 됩니다.
* 벤치 로딩 순서: `--wgtest/--toxicchat PATH` → `$LG_SOURCES/<name>/revision.json`의 스냅샷 → `$LG_EXTERNAL_DATA/audited/*` → (로그인 노드에서 `--online`) HF.
* F1은 harmful 클래스 기준, 파싱 실패는 safe로 집계하고 실패율을 따로 보고합니다. Qwen3Guard는 strict/loose 둘 다 기록하고 `published.yaml`의 `mode`로 비교합니다.
* 예상 시간(RTX 6000 Ada, 4,609 프롬프트): 짧은 출력 가드 10–25분/모델, gpt-oss-20b ≈1h, GuardReasoner-8B 2–4h. 전체 ≈8–10 GPU-h.
* `published.yaml`에는 보고서 §2에 있는 수치만 들어 있습니다(Qwen3Guard 0.6B/8B, WildGuard, GuardReasoner, LlamaGuard3-8B, BingoGuard, gpt-oss ToxicChat). 나머지는 `null`이라 `n/a`로 표시되며, 논문에서 채운 뒤 `compare`를 다시 돌리면 됩니다.

## 6. D3 레지스트리 주의

`python -m lgs1.download --list`로 전부 볼 수 있습니다. `gated`는 라이선스 수락+토큰이 필요하고, `verify`는 HF id를 확정하지 못한 항목입니다(HarmBench, The Stack, arXiv, PubMed, LEDGAR, BingoGuard, GLiGuard, LongSafetyBench, NINJA). 실패해도 다음 항목으로 넘어가며 `download_status.json`에 오류와 대체 소스가 기록됩니다. 대용량(Wikipedia en/zh, PubMed, WildChat, PG-19, The Stack)은 스트리밍으로 `max_docs`만큼만 `docs.jsonl.gz`로 받습니다(레지스트리에서 조정). 모델은 safetensors·토크나이저·config만 받고 `original/`·gguf는 제외합니다.

## 7. 문제 해결

* `OSError: ... offline` on a compute node → 그 모델/데이터가 `$HF_HOME/hub`에 없음. 로그인 노드에서 `bash slurm/s1_download_login.sh --only <name>`.
* gated 401/403 → 해당 HF 페이지에서 라이선스 수락 후 `huggingface-cli login`.
* transformers 4.4x에서 `manual_forward` 오류 → 레이어가 tuple을 반환하는 버전도 처리하지만, rotary가 레이어 내부에 있는 4.44 이전 버전이면 `transformers>=4.51`로 올리는 편이 안전합니다.
* gpt-oss-safeguard-20b 로딩 실패 → `transformers>=4.55`, MXFP4 커널이 없으면 bf16 디양자화(≈40GB, 48GB 안에 들어감). `--dtype bfloat16` 유지.
* Llama Guard 4-12B OOM → 128k가 아닌 짧은 벤치라 bf16 24GB면 충분; `--batch 1`로.
* GuardReasoner-8B가 느림 → `--batch 8`, `max_new_tokens 1024`(guards.py에서 조정). 디코드 토큰 수는 rows에 기록되어 E4의 "추론 토큰 비용" 열에 씁니다.
* 파싱 실패율이 높음 → `rows.jsonl`의 `raw`를 보고 `guards.py`의 파서 정규식을 넓힌 뒤, 새 출력 디렉터리(`--out`)로 `run`을 다시 돌립니다(raw가 저장돼 있어 원인 확인은 재생성 없이 가능; pred 갱신은 새 rows 파일에 기록해 계보를 섞지 않음).

## 8. 이 패키지에서 검증된 것 / 클러스터에서 확인할 것

검증됨(이 환경, CPU, transformers 5.17과 4.51 둘 다): 테스트 69건 green, tiny 모델로 하네스 end-to-end(로딩·생성·첫 토큰 점수·rows 계보·resume·summary), `s1ctl.py` 스크립트 생성.
클러스터에서 확인할 것: (1) `env/s1.env` 경로 실재, (2) `template_status=verify`인 어댑터 6개의 프롬프트를 카드와 대조, (3) `verify` 레지스트리 id 9개, (4) v0.4 어댑터 매핑, (5) Slurm 파티션 이름(`sinfo -o "%P %G"`).
