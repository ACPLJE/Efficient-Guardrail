# lg_s1 — Landmark Guard 계획서 단계 S1 실행 패키지

> **깊이 비대칭 PoC (2026-10-11, 브랜치 `claude/practical-clarke-pbd1dv`)**: 같은 backbone에서 유해 콘텐츠는 얕은 층에서,
> injection은 지시문이 많은 정상 텍스트 대비 깊은 층에서야 갈린다는 주장(비대칭 가드 계획서 E0.3 / E1.1)을 재는 코드는
> `agdepth/` + `depthctl.py`이고, 실행 안내는 [`docs/depth_poc_runbook.md`](docs/depth_poc_runbook.md)다.
> `python depthctl.py smoke` → `python depthctl.py download` (로그인 노드) → `python depthctl.py run --run e03 --sbatch`.

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
* E0-6 runner: `<bench>.rows.jsonl`에 지금 로더가 만들지 않는 id(옛 ToxicChat 로더의 행)가 있으면 `*.rows.stale_<시각>.jsonl`로 치워 두고 새로 돌립니다 — 이어받기(resume)가 옛 행 위에 쌓여 summary n이 5400/2400처럼 어긋나던 문제. summary는 현재 벤치마크 id만 셉니다.
* `python s1ctl.py zeroshot --redo-missing --sbatch` (summary 없는 B1 가드만 재제출)는 이 zip부터 있습니다.
* `published.yaml`: Qwen3Guard-Gen-4B를 arXiv 2510.14276 Table 2로 채움(ToxicChat loose 82.8, WGTest strict 85.6). 논문의 4B WGTest strict(85.6)는 자체 0.6B(87.7)·8B(88.9)보다 낮은 이상치라 우리 4B strict 88.7은 FAIL(+3.1)로 나오고, loose는 85.1 vs 86.06으로 ±1pt 안 → compare가 `FAIL(other mode PASS)`로 표시하고 다른 모드의 논문값·Δ 열을 함께 찍습니다(Table 3 각주 + M1 확인용).
* job array에 `%1` 동시 실행 제한을 걸지 않습니다(9/17 결정: GPU가 많고 스케줄러가 기다림). 이미 제출된 array는 `scontrol update JobId=<id> ArrayTaskThrottle=0`으로 제한을 풉니다. 필요할 때만 `--parallel N`.
* `zeroshot --sbatch`(및 `--redo-missing`)는 이미 PENDING/RUNNING 중인 가드를 다시 제출하지 않습니다(`squeue -r` + array의 guards_*.txt로 판별; 같은 rows 파일에 두 잡이 동시에 쓰는 사고 방지). 일부러 중복하려면 `--force`.
* Qwen3Guard-Stream-8B 판정 수정(harness s1-e06-v2): v1은 사용자 턴 안의 **어느 토큰이든** Unsafe면 unsafe로 세어 정상 프롬프트를 과다 차단(WGTest strict 62.7). 모델 카드대로 `<|im_end|>`까지 넣고 `risk_level[-1]` 하나만 판정으로 씁니다. 재실행: `python s1ctl.py zeroshot --guards qwen3guard_stream_8b --fresh --sbatch` (`--fresh` = 기존 rows/summary를 `*.stale_<시각>`으로 치우고 0부터; 큐에 있는 가드는 건드리지 않음). 논문에 Stream의 벤치별 수치는 없어 n/a로 남고, Gen-8B(89.1)와 몇 pt 안이면 정상으로 봅니다.
* (9/18) resume와 code_commit: 풀어 놓은 패키지는 git이 아니라 `code_commit`이 모든 *.py의 해시라 zip을 새로 풀 때마다 바뀌고, 그동안은 그것도 resume 차단 조건이어서 코드 수정 뒤 이어받기(gpt-oss 1,260행, 스크리닝 array, B1 장기 잡)가 `LineageMismatch`로 죽었습니다. 이제 model_sha·cfg_hash·recipe_hash·data_manifest_sha만 차단하고 code_commit은 행마다 기록 + 경고(`[lineage] … resuming N rows across a code change`)로 남깁니다. `python -m lgs1.lineage check --strict`가 옛 동작.
* (9/18) gpt-oss-safeguard-20b OOM: bf16 dequant 42 GB가 48 GB 카드(44.4 GiB)에 올라가면 활성화 여유가 ~5 GB뿐인데 transformers의 MoE 추론 경로는 32 expert 전부를 dense로 계산해 prefill에 토큰당 ~1.2 MB가 들어 padded 3.4k 토큰 배치에서 OOM. `GuardSpec.max_batch_tokens=2048`(길이순 정렬 후 배치×패딩길이 상한)로 배치를 만들고, 그래도 OOM이면 그 배치를 한 프롬프트씩 재시도, 단독으로도 OOM인 프롬프트는 `error: oom` 행(pred None → parse fail로 집계, summary `n_error`)으로 남기고 잡은 계속 갑니다. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`도 runner가 기본으로 켭니다. 이 zip 풀고 `LG_TIME_ZEROSHOT`은 `python s1ctl.py env-set LG_TIME_ZEROSHOT=48:00:00`으로(파일이 셸 값을 덮어씀).
* (9/18) Llama-Guard-4-12B: Llama 4 텍스트 설정의 `cache_implementation="hybrid"` 때문에 generate()가 static hybrid 캐시를 만들다 `min(sliding_window, None)` TypeError로 죽었습니다(torchvision 설치 뒤 두 번째 실패). 그렇다고 `past_key_values`만 넘기면 "Passing both `cache_implementation` and `past_key_values` is unsupported"로 다시 죽습니다(세 번째 실패). 그래서 `load_model`이 generation_config·config·text_config의 `cache_implementation`을 None으로 지웁니다(`[load] cache_implementation cleared on …` 로그). 그 다음 `DynamicCache(config=…)`는 Llama 4의 chunked 층을 `sliding_window=None`으로 만들어 마스크 계산에서 `'>=' int vs None`으로 죽었습니다(네 번째 실패). 다섯 번째: Llama-Guard-4-12B의 config.json이 `attention_chunk_size: null`(= full attention)인데 transformers ≥4.53의 Llama 4 forward가 chunked 마스크를 무조건 만들어 "Could not find an `attention_chunk_size` argument"로 죽습니다(llama-stack #2871, 5.17에서도 재현). → 로드 시 모든 config 객체(top-level·text_config·서브모듈 사본)에 `attention_chunk_size = max(2^20, max_position_embeddings)`를 넣어 chunked == causal == full attention이 되게 합니다(tiny Llama 4에서 명시적 full-attention 모델과 토큰 일치 확인, `[load] Llama 4 config patched: …` 로그). 그리고 `vlm_generate`가 전략 사다리를 탑니다 — `plain` (`DynamicCache()` full 층만; chunk 8k보다 짧은 프롬프트라 판정 동일, tiny Llama 4로 chunk보다 긴 입력까지 토큰 일치 확인) → `config` → `default` → `nocache`(`use_cache=False`, 매 스텝 재계산; 16 토큰이라 감당 가능). 처음 성공한 전략을 모델별로 기억하고, 실패는 `[vlm] generate strategy '…' failed: …` 한 줄로 남깁니다. OOM은 사다리를 타지 않고 그대로 올라가 `score_chunk_safe`가 처리합니다. torchvision은 venv에 `pip install "torch==<설치된 버전>" torchvision`으로 넣어야 합니다(Llama4ImageProcessorFast 요구).
* (9/18) 스크리닝 결과 예산 미달 5개(pg19 7.3M·wildchat 16.6M·stack 5.3M·wiki_zh 4.8M·agentdojo 29k 토큰). 합성은 도메인 풀에서 문서를 연속으로 이어 배경을 만들고 같은 문서를 반복하지 않으므로, 풀이 L보다 작으면 그 L을 만들 수 없고 L에 가까우면 행마다 거의 같은 배경이 됩니다. → `Pools.eligible`: split 풀 토큰 ≥ 2×L인 도메인만 그 L에 사용(`POOL_FACTOR=2`), 제외는 `[synth] … skipped (pool N tok < 2 x L)` 한 줄로 기록, 남는 도메인이 없으면 SystemExit. 실제 영향은 agentdojo뿐(L ≤ 8k에서만 benign_long·ood_domain에 등장). pg19·stack은 배경 재사용이 늘 뿐 모든 L 가능 — 데이터 카드 스크리닝 표의 budget_met 열에 그대로 남깁니다.
* (9/18) dry-run 260행 중 20행이 길이 ±10% 밖: 배경의 마지막 문단을 문장/줄 단위로만 잘랐는데, wildchat 한 줄짜리 긴 메시지·minified code처럼 자를 단위가 없는 문단이면 아무것도 못 붙이고 끝나 짧은 행이 됐습니다. `_cut_to_fit`이 문장/줄 → 단어 경계 순으로 자르고, 토크나이저가 있으면 잘라낸 조각을 정확히 세어 최대 3번 다시 자릅니다(합성 본 실행·variants 모두 `pools.tok` 전달). 검증 리포트에 `len_bad_examples`(id·도메인·L·n_tok)가 남고 dry-run 출력에도 찍힙니다.
* (9/18) embed(bge-m3) 전 task 즉사: BAAI/bge-m3는 Hub에 safetensors가 없고 `pytorch_model.bin`(2.3 GB)뿐인데 D3 모델 패턴이 `*.bin`을 제외해 가중치 없는 snapshot이 "ok"로 남았고, 오프라인 노드의 from_pretrained가 `'NoneType' object has no attribute 'endswith'`로 죽었습니다. registry의 bge_m3 항목이 `pytorch_model.bin`을 허용하도록 바꿨고(onnx/·imgs/·*.pt 제외), embed는 로드 실패 시 재다운로드 명령을 안내하며, variants의 `related`는 임베딩 5개가 없으면 시작 전에 같은 안내와 함께 멈춥니다. 재다운로드: `bash slurm/s1_download_login.sh --only bge_m3 --no-skip`.
* (9/18) 첫 전량 합성에서 qwen3 tokspans 검사 실패(전체 0.3–5%의 행이 L ±10% 밖; s3_finish `DependencyNeverSatisfied`). 원인 셋: ① 문단별 토큰 수를 더해 배경을 만들지만 문단을 "\n\n"으로 이을 때 문단마다 토큰이 늘어 wildchat(줄 단위)·코드에서 +5–15% 초과, ② 자를 단위가 없는 긴 줄을 자르며 `est ≥ need_lo`면 받아들여 3,000토큰 줄이 통째로 들어감(1024 목표에 4,062), ③ variants가 (L − needle − 40)에 ±10%를 다시 적용해 framed가 0.8999 L. 수정: `bg_bounds(L, extra, tol, n_items)`가 행 기준 창을 계산(구분자 0–2토큰/항목 여유 + 0.2% 마진), `build_background`가 이어붙인 텍스트를 토크나이저로 정확히 세어 창 안으로 다듬음(`_exact_fit`), 자르기는 문장/줄 → 단어 → 글자 비율 순이고 창 안에 들어올 때만 채택, needle_repeat는 반복 수를 정확히 맞춤, related는 chunk 조립 뒤 같은 다듬기. `write_rows`가 창 밖 행을 세어 synth/variants 잡이 rc=2로 즉시 실패하고(체인이 나쁜 데이터로 진행되지 않음), `s3ctl.py status`의 D8이 report.json 검사까지 반영합니다. 실제 Qwen3 토크나이저로 chat/code/prose 배경·decoy flood 64개·benign-long을 검사하는 `tests/test_s2_length_real_tok.py` 추가(캐시에 토크나이저가 있을 때만 실행).
* (9/18, 두 번째 전량 합성) 그래도 train/main 42/64,000행·calib/benign_long 3/4,000행이 ±10% 밖(나머지 7세트는 0). 배경이 아주 짧은 채팅 줄(3–5토큰)로만 이뤄지면 줄마다 구분자 토큰이 하나씩 붙어 문단 합계보다 수백 토큰이 길어지는데, `_exact_fit`이 한 번에 문단 하나씩 6번만 잘라 창 안으로 못 들어왔고, benign twin이 harmful needle보다 최대 20% 길어도 창은 harmful 기준이었습니다. 수정: ① 구분자 비용을 토크나이저로 한 번 재서(`sep_cost`, Qwen3 = 1) 수집 단계 추정에 포함, ② `_exact_fit`이 초과분만큼 문단을 한꺼번에 버리고 다시 재어 몇 라운드 안에 수렴, ③ 정확히 세어 짧으면 같은 시작점에서 목표를 올려 다시 모으고 그래도 안 되면 다른 시작점(최대 4), ④ `bg_bounds(..., extra_max=)`로 창이 twin까지 포함, ⑤ `build_rows_checked`가 배경 하나에 얹는 모든 (depth, h/b) 행을 실제로 세어 하나라도 창 밖이면 배경을 다시 만듦(최대 4회) — 생성기가 센 값을 행에 남기므로 `write_rows`는 그 행을 다시 세지 않습니다(D8 tokspans가 독립 재검증). 실패한 세트는 manifest를 쓰지 않아 `s3ctl.py status` D6가 줄어들고, 잡 스크립트는 본문이 실패해도 `[s1] done rc=N`을 찍습니다. `s3ctl.py finish`(gates+datacard+pilot) 서브커맨드와 `LG_CPU_PARTITIONS`(CPU 잡을 GPU 큐 밖으로) 추가 — `docs/S3_runbook.md` "일부만 다시 돌릴 때"·"CPU 파티션".
* (9/18, variants) framed(array task 7)가 366행 밖(예: L=131072 wildchat 행이 115,869 = 0.88 L)으로 rc=2. 변형 생성기 6종(decoy·decoy_flood·fragmented·multi_needle·long_needle·framed)은 `TRAIN_DOMAINS[pi % 5]`로 도메인을 L과 무관하게 골라 main 세트의 자격 규칙(split 풀 ≥ 2×L)을 건너뛰었고, framed·ood_domain의 창은 twin 길이를 반영하지 않았습니다. 수정: `_domain_for`가 L마다 자격 있는 도메인 위에서 round-robin(main과 동일, 제외는 `[synth] test/<variant> L=…: <dom> skipped` 한 줄), 창은 needle·twin(framed는 프레임 포함) 중 짧은 쪽/긴 쪽으로 `bg_bounds(..., extra_max=)`, 그리고 `_checked_rows`가 배경 하나에 얹는 모든 행을 실제로 세어 창 밖이면 측정값 중앙값으로 창을 다시 맞추고 새 시작점에서 다시 만듭니다(최대 4회; main의 `build_rows_checked`도 같은 재조정). 실패한 변형은 manifest 없이 남으므로 `python s3ctl.py variants --variant framed --sbatch`로 그 변형만 다시 만들고 tokspans를 `--dep <calib synth>:<framed>`로 다시 냅니다.
* (9/18, calib 재실행) 앞 항목의 코드로 calib/main 110행·calib/benign_long 146행이 짧게(0.88 L) 나와 rc=2 — train(≤16k)은 0이었는데 큰 L에서만. 원인: 구분자 비용을 "상한 1"로 추정했는데 Qwen3는 `.\n\n`·`!\n\n`·`)\n\n`처럼 문장부호 뒤 빈 줄을 한 토큰으로 합쳐 산문·코드·문장부호로 끝나는 채팅 줄은 구분자 비용이 0입니다. 추정이 문단 수만큼 과대해 "추정 ≥ lo가 된 첫 문서 경계"에서 수집을 멈추면 실제 토큰은 lo 아래였고, 목표를 올려 다시 모으는 재시도도 같은 문서 경계에서 멈춰 소용이 없었습니다(문서가 8k, 문단이 짧을수록 확률↑: 128k에서 20–50%). 수정: `build_background`가 문단 단위 커서(`_Cursor`)로 통째 문단을 추정치로 목표까지 모은 뒤 **정확히 세어** 창보다 길면 끝에서 다듬고(`_exact_fit`), 짧으면 부족분만큼 문단을 더 뽑아 다시 세는 것을 반복합니다(최대 8라운드, 시작점 4개). 추정치는 더 이상 판정에 쓰이지 않습니다. 실제 Qwen3로 산문/문장부호 채팅/맨 단어 줄/코드 × 4k–128k 전부 0/12 실패, 128k 배경 하나 ~0.55 s. `tests/test_s2_length_real_tok.py`에 산문·문장부호 채팅 케이스 추가.
* (9/19) 전량 합성·변형·tokspans 5종 통과 뒤 E0-4 게이트 3·4 FAIL. ③ `duplicate_ids=6`: train의 `random2+ends`가 무작위 depth 두 개를 소수 4자리로 반올림하는데 8,000 (pair, L) 중 3개에서 둘이 같은 값 → 같은 id·같은 본문의 h/b 행이 두 번 생성. `positions_for`가 겹치거나 0/1과 같은 depth를 다시 뽑도록 고쳤고(충돌 때만 난수를 더 쓰므로 나머지 pair는 그대로), 이미 만든 데이터는 `python s3ctl.py dedupe --split train --subset main`으로 중복 id의 두 번째 행을 버리고 manifest(sha256·rows·counts)와 tok/<5 모델>/train/main/spans를 함께 정리합니다(train/main 64,000 → 63,994행; 버린 행은 남은 행과 동일). ④ `long_needles_outside_subset=16961`: train/native 행은 풀의 moderation 프롬프트를 그대로 담으므로 128토큰을 넘는 게 당연한데 게이트가 native를 예외로 두지 않았음 → native 제외(`native_rows_gt128_exempt`로 개수만 기록). finish 잡은 gates가 실패하면 datacard·pilot을 건너뛰므로 `python s3ctl.py finish --cpu`(로그인 노드, 몇 분)로 세 단계를 다시 돌린 뒤 `gates --seal`.
* (9/19, S4 시작) 0.6B head s0가 TITAN RTX(n7)에서 **2.8 s/step**(58,120 step → 45시간, 12시간 제한에 걸려 아무것도 남지 않을 상황). Turing(sm_75)은 bf16 텐서코어가 없어 bf16 autocast가 느린 경로로 떨어집니다(Ampere/Ada 예상 ~0.3 s/step). 수정 셋: ① `slurmutil.NO_BF16_GPUS`(TITANRTX·RTX2080Ti 등)를 ≥24 GB 티어의 모든 잡에서 제외(`LG_ALLOW_NO_BF16=1`로 되돌림; `python s1ctl.py partitions`에 사유 표시) — zero-shot·longbench도 같은 이유로 제외(B1 첫 task가 TITAN RTX에서 2.3 s/row = 19시간). ② head·dense가 30분마다(`ckpt_every_sec`) 그리고 SIGTERM 때 `head.resume.pt`/`dense.resume.pt`(fp32 master·optimizer·scheduler·RNG·위치)를 쓰고, 다시 제출하면 그 지점부터 이어서 학습(`[head] resumed from …`; 중단 없이 돈 것과 가중치 동일 — 테스트). sbatch에 `--signal=TERM@600`(제한 10분 전)을 넣어 신호를 받으면 저장 후 rc=3으로 나가므로 의존 잡은 대기 상태로 남고 같은 명령을 다시 내면 이어집니다. head 기본 제한 24 h, dense 48 h. ③ 로그 줄에 `(0.31 s/step, eta 4.9 h, gpu 11.2 GB)` 추가. `s4ctl.py t1/t2 --seeds 0`으로 한 seed의 체인만 다시 제출.
* (9/19, 4B head OOM) `s4_head_qwen3guard_4b_s0`가 첫 `opt.step()`의 `_foreach_sqrt`에서 CUDA OOM(47.4 GiB 중 46.5 GiB 사용). 층 12–35의 2.4B 파라미터를 fp32 master로 두면 가중치 9 + grad 9 + Adam 18 = 36 GiB인데 AdamW의 기본(foreach) 경로가 sqrt(v)를 전 파라미터에 대해 한꺼번에 만들어 +9 GiB. 수정: `make_optimizer`가 CUDA에서 `fused=True`(제자리 갱신, 임시 없음), `lgs4/__init__`이 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 재개 체크포인트 간격을 학습 파라미터 수에 비례(0.6B 30분, 4B ~4시간; 파일이 ~29 GB). 그래도 안 맞으면 `python s1ctl.py env-set LG_BB_MODE_qwen3guard_4b=lora`로 4B를 8B처럼 LoRA r=16으로(Table A2 각주). 예상 피크 ~41 GiB.
* (9/20) 0.6B head 3 seed 완료(58,120 step, val 윈도우 AUPRC 0.964/0.904/0.949) 뒤 router 3개가 **4시간 제한**에 죽음. Stage A가 학습 행 141k개(presence 66k + native 75k)의 모든 윈도우 2.1M개(5.5억 토큰)를 행 하나씩 전체 모델에 통과시켜 RTX 3090에서 targets만 5시간 이상, Stage B는 2 epoch의 trunk 통과가 또 4시간 — 계획서의 "targets 1 h + fit 40 min"이 틀렸습니다. 수정 셋: ① router는 head 학습 행의 seed 고정 균등 부분표본(`ROUTER_HP.router_rows=12000` presence + native_ratio 0.5 → 18k행, 그 행의 모든 윈도우 ≈ 0.4M)으로 학습(3k 파라미터 라우터에 충분; Table A2 각주), ② Stage A·B 모두 윈도우를 행 경계 없이 256개씩 묶어 forward(작은 L·native 행의 GPU 유휴 제거), ③ targets 캐시를 2,000행 단위 part 파일로 써서 죽어도 이어감. router 제한 12 h, valcheck 6 h. 죽은 체인은 `python s4ctl.py t1 --seeds 0,1,2 --from router --sbatch`로 router부터 다시(head 잡을 큐에 넣지 않음).
* (9/20, dense-FT 예산) dense s0가 RTX 3090에서 **15 s/step → 58,120 step = 240시간/seed**. 계획서의 "같은 step 수"는 dense step(행 8개 × 평균 2.8k 토큰 = 22.6k 토큰, 28/28 층)이 head step(윈도우 32개 × 256 = 8k 토큰, 19/28 층)보다 8배 넘게 비싸다는 걸 놓친 것. `DENSE_HP.budget`을 추가해 step 수를 head.pt에서 유도: `tokens`(기본, head와 같은 학습 토큰 수 4.76억 → ≈21k step), `flops`(같은 학습 FLOPs → ≈12k step), `steps`(옛 규칙). 로그 첫 줄에 유도식이 찍히고 `dense_ft.pt` meta에 기록됩니다(Table A2에 "matched token budget"으로). 같은 step의 행 8개를 길이순으로 묶어 padding을 줄이고(0.6B micro 24k 토큰), 예상 3090 ≈9 s/step → 21k step ≈ 52 h/seed(Ada ~25 h). 시작할 때 `dense.resume.pt`가 옛 예산(58,120)이면 거부하므로 지우고 다시.
* (9/20, 자동 재개) head/dense sbatch가 `--signal=B:TERM@1500 --requeue --open-mode=append`로 나가고, 배치 셸이 신호를 trainer에 전달(`trap` + 프로세스 그룹) → trainer가 체크포인트를 쓰고 rc=3 → 스크립트가 `scontrol requeue $SLURM_JOB_ID`로 **같은 잡 번호로 다시 큐에** 들어가 이어서 학습(router/valcheck 의존성 유지, 손으로 다시 낼 필요 없음). `tests/test_sbatch_signal.py`가 가짜 trainer로 신호 전달·rc=3·requeue 호출을 검증. 4B head(24 h 제한, 33 h 예상)는 이 경로로 두 번에 나눠 끝납니다. `python s4ctl.py status`가 진행 중 단계도 `running <step>/<total> eta <h>`로 보여 줍니다.
* `stack`(bigcode/the-stack-smol)은 gated: 데이터셋 페이지에서 약관 수락 + `huggingface-cli login` 후 `bash slurm/s1_download_login.sh --only stack`. 30개 언어 × 10k행이라 shard를 섞어(seed 0) 120k 문서에 모든 언어가 들어가게 했습니다.

```bash
pkill -f lgs1.download
bash slurm/s1_download_login.sh --only polyguard_qwen_7b,bingoguard_8b,gliguard,longsafetybench,pg19,stack,arxiv
```

## 0.11 S2 패키지 (9/17) — 데이터 v2 전반부: `s2ctl.py` + `lgs2/`

계획서 S2(배경 스크리닝 D4, needle 풀·분할·감사 D5, 합성 스크립트·dry-run D6)를 같은 패키지 안에 넣었습니다. 순서와 통과 기준은 `docs/S2_runbook.md`, 규칙·상수는 `lgs2/config.py`(도메인 10개·소스 14개·MinHash·분할 비율·합성 레시피 `RECIPE_V2` — 이 dict의 해시가 모든 합성 행의 `recipe_hash`).

```bash
python s2ctl.py haystack --sbatch                          # D4a 10 도메인 문서 풀 (Qwen3 토큰 예산 × 1.35)
python s2ctl.py screen --sbatch --dep <haystack jobid>     # D4b 512-토큰 chunk × LG3-8B·Qwen3Guard-8B(strict), 20 task
python s2ctl.py needles --sbatch                           # D5a needles/pool.jsonl.gz (그룹 = 정규화 해시 + MinHash 5-gram ≥0.8)
python s2ctl.py screen --summarize                         # D4  통과율·keep 목록·수동 확인 50건
python s2ctl.py embed --sbatch                             # D4  bge-m3 chunk 임베딩 (related 서브셋용)
python s2ctl.py splits --local                             # D5b 분할(공개 test 소스는 test만) + SafetyNIAH·외부 test 9종 overlap 감사 + gates
python s2ctl.py synth --dry-run --local                    # D6  ≈200행 + 스키마·규칙 검사 (twin·문단 경계·±10%·≤128 토큰)
python s2ctl.py status                                     # 6개 항목 DONE/PENDING
python s2ctl.py all --sbatch                               # 위 체인을 의존성으로 한 번에
```

모든 sbatch는 gres 1(QOS normal), 파티션은 GPU 메모리 티어(스크리닝·임베딩 24 GB, CPU 작업은 아무 GPU), array 동시 실행 제한 없음. S3(전량 합성·변형·토큰화·봉인)는 `lgs2.synth`의 `run_full`과 `validate_rows`를 그대로 씁니다.

## 0.12 S3 패키지 (9/17) — 데이터 v2 후반부 + baseline 장기 잡: `s3ctl.py` + `lgs3/`

계획서 S3(전량 합성 D6, 변형 10종 D7, 모델별 토큰화 D8, E0-4 게이트 7개·G0 봉인, 데이터 카드 D10, E0-5 tiny 파일럿, B1/B2 장기 잡 착수). 순서와 통과 기준은 `docs/S3_runbook.md`.

```bash
python s3ctl.py all --sbatch          # synth(4 잡) → variants(9) → tokspans(5×4) → gates·datacard·pilot
python s3ctl.py gates --seal          # 7개 게이트 PASS → lg_data_v2 읽기 전용 (G0)
python s3ctl.py b1-start --sbatch     # zero-shot 12종 × v2 test ≤28k (+SafetyNIAH, CD 2종) — 티어별 array, 동시 제한 없음
python s3ctl.py longbench --summarize --bench v2main
```

`lgs3/lgeval.py`는 LG 평가 러너의 핵심(route→head→verdict, hit@k, 계보 열, matched-FPR 집계)으로 S5 이후 학습된 체크포인트에도 그대로 씁니다. `lgs3/longbench.py`는 정책(native / trunc_keep_head / trunc_keep_tail / yarn / chunk_or:W:overlap)을 행마다 기록해 Table 3의 "장문 처리 정책" 열을 만듭니다.

## 0.13 S4 패키지 (9/18) — 학습: `s4ctl.py` + `lgs4/`

계획서 S4(T1 LG 0.6B×3 seed, T2 dense-FT 0.6B×3 seed, T3 LG 4B, E0-7 FLOPs 대조). 순서·하이퍼파라미터(Table A2)·통과 기준은 `docs/S4_runbook.md`. 봉인(`lg_data_v2/SEALED`) 전에는 학습 명령이 거부됩니다.

```bash
python s4ctl.py all --sbatch          # prep → head→router→valcheck ×3 seed (0.6B) + dense ×3 + 4B 체인 + E0-7
python s4ctl.py status                # run별 head/router/dense, val_report(hit@8@28k ≥ 0.95), E0-7 PASS
```

새 코드는 v0.4를 쓰지 않고 `lgs1/lgcore` 위에 만들었습니다: `lgs4/backbone.py`(동결 정책·학습 forward `run_layers`(= lgcore.manual_forward, grad ckpt 옵션)·체크포인트 I/O·`load_lg`), `lgs4/sampling.py`(윈도우 샘플링), `lgs4/head_train.py`, `lgs4/router_train.py`(targets 캐시 + BCE·listwise KL), `lgs4/dense_train.py`, `lgs4/valcheck.py`, `lgs4/flops.py`, `lgs4/lora.py`(peft 없이). `tests/test_s4_train.py`가 tiny 백본으로 prep→head→router→dense→valcheck→flops→재로드 동일성과 "학습 forward == lgcore 추론 경로, trunk에 grad 없음"을 검증합니다. `lgs3/lgeval.LGRunner.from_modules(*BB.load_lg(run_dir)...)`로 S5 평가가 바로 이어집니다.

## 0.14 S5 패키지 (9/20) — 평가: `s5ctl.py` + `lgs5/`

계획서 S5(E1-1·2·3 parity, E2-1·2 ≤28k, E2-7 3층 분해, B2 CD, B4 dense-FT, T4 LG 8B, G1 판정). 순서·기준은 `docs/S5_runbook.md`.

```bash
python s5ctl.py all --sbatch          # 준비된 체크포인트로 e1 + e2(≤28k) + e27 + CD 기본 config 잡 제출
python s5ctl.py status                # eval/v2/<exp>/<method>__<run>.rows.jsonl 행 수, s5_ 잡
python s5ctl.py tables                # reports/s5/table1_e1.md · table2_le28k.md · e27_decomposition_le28k.md (+ Fig 3·4 JSON)
```

* 모델 지정 `lg:<run>[:kK] | sliding | oracle | needle | dense | base:<guard>`(`lgs5/config.py`). LG rows는 라우팅된 윈도우의 head 확률을 라우터 순서로 담아 k=8·16이 k=32 한 번의 실행에서 유도됩니다(E6-1의 k sweep도 같은 rows).
* 임계값은 E0-3 규약: 기본 τ + calib benign-long (method, L)별 엄격 matched-FPR(5%·1%, 달성 FPR 기록). zero-shot·CD는 `lgs3.longbench` rows(`longbench/v2main`)를 같은 표에 합칩니다(기본 판정, `*`).
* E1은 needle 풀의 test 소스 전 행(7 벤치 + XSTest safe·OR-Bench-Hard FPR + WGTest adversarial/vanilla). Qwen3Guard base는 strict/loose 두 행.
* (9/20 저녁) B1 zero-shot ≤28k 첫 집계: dense 가드는 길이에 따라 recall이 무너지거나(LG3-8B 0.31→0.09, WildGuard 0.82→0.02, Nemotron 0.72→0.08) 유지하되 장문 benign FPR이 치솟습니다(PolyGuard 0.36, Qwen3Guard-8B 0.39 @28k) — 기본 판정 기준. 같은 규약(matched-FPR 5%)으로 비교하려고 `s5ctl.py b1-calib`(12종 + CD를 calib benign-long 3,000행에)와 longbench `v2calib` 벤치를 추가했고, 표에 `unscored@max L`(OOM·문맥 초과·파싱 실패 행 수) 열을 넣었습니다. gpt-oss는 48 GB 카드에서 ≥4k가 전부 unscored(dense MoE prefill OOM) → S6 전에 chunked prefill.
* (9/20 밤) LG 0.6B 3 seed의 E2 ≤28k 첫 행(기본 판정): R@1k 0.83 → R@28k 0.84–0.85, ε̂=0, hit@32@28k 0.973–0.977, k=8도 0.83 — 하지만 benign FPR@28k(τ=0.5)가 0.17–0.20으로 높음(oracle 윈도우만으로도 twin에서 0.15 → head 자체의 hard-negative 오탐 + 32 윈도우 max 집계 ~3–5 pt). matched-FPR 5% 행은 calib 행이 끝나야 채워집니다. 두 가지 고침: ① Table 2의 FPR을 twin / benign-long으로 나눠 표시(음성의 종류가 다름), ② **E2-7의 세 층이 서로 다른 음성으로 임계값을 잡고 있던 것**(needle·oracle = calib/main twin, LG = calib benign-long → 쉬운 음성을 쓴 LG가 유리, routing 손실 과소) → LG e2 rows에 calib/main 12,000행을 추가하고 E2-7은 세 층 모두 twin으로, Table 2는 benign-long만으로 임계값. 이미 만든 LG rows 파일은 잡이 끝난 뒤 `python s5ctl.py e2 --models lg:qwen3guard_0.6b_s0,lg:qwen3guard_0.6b_s1,lg:qwen3guard_0.6b_s2 --sbatch`를 한 번 더(todo=12,000, ~1 h/seed). 표의 recall·FPR은 판정이 있는 행만으로 계산하므로 unscored가 큰 행(granite 3221/3630, wildguard 2768/3080, gpt-oss 전부)의 recall은 처리된 행 안의 값 — 논문에는 unscored를 miss로 세는 effective recall도 같이.
* (9/20 밤, 배열 잡 목록 파일 충돌) `s3ctl._array_job`의 task 목록 파일 이름이 초 단위 stamp뿐이라(`longbench_0920_193847.txt`) 같은 초에 낸 두 티어가 서로의 목록을 덮어썼습니다 — b1-calib에서 24 GB 배열(12 task)이 40 GB 티어의 한 줄(llamaguard4_12b)을 읽어 task 0은 24 GB 카드에서 LG4를 돌리다 3,000행 전부 오류로 쓰고 나머지 11개는 빈 줄로 즉시 종료(v2calib 파일 1개만 생긴 이유). b1-start(9/19)의 40/48 GB 티어도 같은 초였을 가능성 → `s3_longbench_40gb` 잡이 LG4 대신 gpt-oss를 중복으로 돌렸는지 로그의 `longbench guard=` 줄로 확인. 이름에 잡 이름을 붙여(`longbench_<stamp>_<job>.txt`) 배열마다 고유하게 했고 `queued_longbench`의 정규식도 맞췄습니다. `s5ctl status`는 rows 파일마다 최신 로그를 읽어 `running n/todo  ms/row  eta`·`done`·`queued`를 보여 줍니다.
* (9/20 밤, 확인) 로그로 확정: b1-calib 24 GB task 0 = `guard=llamaguard4_12b`, b1-start의 `s3_longbench_40gb`(966381_0) = `guard=gpt_oss_safeguard_20b` — 즉 LG4는 B1(v2main·SafetyNIAH)에서 한 번도 돌지 않았고 gpt-oss v2main은 48 GB·40 GB 두 잡이 같은 파일에 썼습니다(40 GB 카드에선 CPU offload로 25 h째). 966381 취소, 표·summarize 로더에 `lgs1.lineage.dedupe_rows`(id당 한 행, 판정 있는 행 우선) 적용, LG4 v2main은 `s3ctl longbench --guards llamaguard4_12b --bench v2main --policy native --L ≤28k`로 따로 제출.
* (9/20 밤, FPR 손잡이) LG의 τ=0.5 FPR 0.17–0.20(twin+bl)을 다루는 도구 둘: ① Table 2에 **FPR@τ1k twin/bl**(1k의 matched-5% 임계값을 max L에 적용 — 길이에 따른 오탐 팽창; G1 (c)를 이 값으로 정의), JSON에 `fpr@tau1k`·`recall@tau1k_pos_mean` per L; ② `lgs5/aggregate.py`(E2-8) — 있는 rows의 `deep_probs`·`sel_scores`로 max / top2_mean / second_max(debounce) / lse_0.5 / router_gated_{1,3} × k∈{8,16,32}를 같은 규약(calib 임계값, FPR@τ1k, pos-min, ε̂)으로 오프라인 재평가 → `reports/s5/agg_ablation_le28k.md`(`tables`가 같이 만듦). max|k32 행이 Table 2의 lg_k32와 일치하는지 테스트. twin은 같은 풀의 길이 맞춘 benign 프롬프트(XSTest·OR-Bench-Hard 포함)라 twin FPR = hard-negative 과잉거부이며, 기본 판정 기준 LG 0.19는 Qwen3Guard-4B 0.24·8B 0.38·PolyGuard 0.36·CD 0.51보다 낮음 — 진짜 비교 대상은 같은 데이터로 학습한 dense-FT.
* (9/20 20:24, matched-FPR 첫 표) LG 0.6B k=32(3 seed): R@1k 0.89/0.89/0.92 → R@28k 0.85/0.83/0.85, pos-min 0.66, hit@32 0.975; **bl FPR(τ=0.5) 0.024/0.032/0.026** — 0.19는 전부 twin(0.17–0.20)이었음. 길이 팽창: 1k 임계값을 28k에 두면 bl FPR 0.084/0.130/0.102, ε̂ 0.044/0.058/0.064. E1: W-avg F1 0.807/0.816/0.806 vs base 0.6B strict 0.768·loose 0.819(−1 pt), XSTest FPR 0.07–0.11·OR-Hard 0.06 vs base 0.22/0.73(strict). E2-7: framing 손실 s0 5–7·s1 2–4·s2 7–9 pt. E2-8: second_max 붕괴(0.53–0.61), lse_0.5 발산, top2_mean 무효, router_gated_3이 bl 팽창 0.066/0.074/0.086·R@28k 0.847/0.858/0.850으로 가장 좋음(집계 선택은 calib/main 행에서 — `select_on_calib`, 재실행 뒤 채워짐). 다음 손잡이는 head: 토큰별 max pooling(framing) + 라우터 상위 benign 윈도우 hard-negative mining(팽창).
* (9/20 밤, head 설계 질문) "토큰별 max head vs landmark 토큰": 라우터가 이미 landmark 역할(트렁크 특징의 토큰 점수 → 윈도우 max, `pool_scores`)이고 head만 마지막 토큰 probe. landmark/CLS 토큰은 여전히 벡터 하나로 윈도우를 요약하는 MIL이라 위치 의존이 남지만, 토큰별 점수 + 윈도우 안 max는 needle 위치에 불변이고 합성 데이터의 `needle_tok_span`으로 토큰 단위 감독이 가능하며(Qwen3Guard-Stream과 같은 설계, 위치 표시도 공짜) FLOPs 변화 없음. 결정 전 확인용으로 `lgs5/framing.py`(oracle 행에서 needle 뒤 배경 토큰 수별 head 확률; `tables`가 같이 만듦)를 추가 — 128+ 구간에서 확률이 떨어지면 pooling 문제로 확정.
* (9/20 20:47, framing 진단 결과 — 가설 기각) needle 뒤 배경 토큰 수별 head 확률: 0–15개 **0.69–0.75**, 16–63개 0.81–0.86, 64–127개 0.83–0.86, 128+개 0.85–0.87 (3 seed, 모든 L 동일). 즉 "뒤 배경이 신호를 흐린다"는 반대였고, 마지막 토큰 probe는 needle 뒤에 **16토큰 이상**이 있어야 제 점수를 냄(needle 끝 = 창 끝이면 −13 pt). 창 단위로는 13%(2,650/20,610)가 그 구간. 그래서 framing.py에 행 단위(containing 윈도우 중 최대 tail로 묶은 recall, 128토큰 초과 needle 비율)를, e27에 framing 손실의 harmful-side(needle 창 점수 하락) / benign-side(twin 창 점수 상승 → 임계값 상승) 분해를 추가. 설계 결론은 이 둘을 보고: 행 단위 손실이 0–15 구간에 몰리면 stride 64(또는 창 뒤 16토큰 여유) 같은 기하 수정, benign-side가 크면 head 재학습(twin 창 hard negative), 둘 다 작으면 토큰별 head의 근거는 약함.
* (9/20 21:10, 진단 결과) 행 단위: containing 윈도우의 최대 tail이 0–15인 행이 10.4%(1,367/13,200; needle >128토큰은 0 — 즉 거의 전부 **문서 끝 depth 1.0**의 needle, 마지막 창이 needle 끝에서 끝남)이고 그 recall(τ=0.5) 0.63–0.66 vs 128+ 0.87–0.88 → Table 2의 pos-min 0.66이 바로 이 경우. 16–63: 3.8%(0.76–0.80), 64–127: 21%(0.84), 128+: 64%(0.87–0.88). E2-7 분해(28k): framing 손실 0.071/0.042/0.086 = harmful-side 0.015/0.010/0.025 + **benign-side 0.055/0.032/0.062**(needle 단독 임계값에서 twin 창 FPR 0.138/0.140/0.156 — 문서 안의 twin이 단독 twin보다 높게 찍힘). needle 단독 행은 전부 tail 0이므로 benign-side가 같은 probe-위치 효과인지 보려고 twin 창의 tail별 점수 표를 framing.py에 추가. 설계 함의: 마지막 토큰 probe는 뒤에 16토큰 이상 있어야 하므로 (i) 창 끝에 고정 suffix/landmark 토큰을 붙여 항상 같은 자리에서 읽거나(base 가드가 chat template의 assistant 접두로 판정하는 것과 같은 구조), (ii) 토큰별 점수 max(needle 안에서 읽음). 둘 다 head 재학습 필요; 기대 이득은 τ=0.5 recall ~2.7 pt(주로 depth 1.0)와 benign-side 3–6 pt의 일부.
* (9/20 21:10, LG4 버그 둘) b1-calib 20:13 제출은 zip을 풀기 전이라 목록 충돌이 그대로 재현(24 GB task 0 = LG4, 나머지 빈 줄) — 이번엔 코드 탓이 아니라 순서 탓. 그리고 LG4(`kind=vlm`)는 longbench가 processor를 `tok`으로 받아 `tok(text)`로 토큰을 세다 `ValueError: You have to specify text`(processor의 첫 인자는 images) → 3,000행 전부 오류. `ctok = processor.tokenizer`로 세고 `vlm_one`은 `apply_chat_template(tokenize=True)` 실패 시 렌더 후 `proc(text=…)`로 폴백. 테스트에 processor형 가짜 가드 추가.
* (9/21 00:10, e2 재실행 3개가 죽은 원인) `LineageMismatch: model_sha existing=0ad3… current=c53b…` — LG runner의 model_sha가 run 디렉터리 **전체** 지문(`*.pt` 모두)이었는데, 같은 디렉터리에서 도는 dense-FT가 `dense.resume.pt`를 30분마다 다시 쓰니 지문이 바뀌어 이어쓰기를 거부. model_sha를 실제로 읽는 파일만으로(LG = head.pt+router.pt, dense = dense_ft.pt) 정의하고(`lgs5.runners.lg_model_sha`), 이미 쓴 rows는 `s5ctl relineage`로 고침(router_sha가 같을 때만). 테스트: rows 뒤에 가짜 `dense.resume.pt`가 생겨도 resume 통과, relineage 갱신/거부. `s5ctl status`는 로그가 running인데 큐에 잡이 없으면 `STOPPED -> .err`로 표시. 또 점수가 이진인 baseline(GuardReasoner·gpt-oss·LG4)은 calib이 있어도 quantile이 0/1로 퇴화하므로 Table 2에서 기본 판정을 유지(`calib_binary`).
* (9/21 11:52, S5 ≤28k 거의 완성) LG e2 3 seed 44,400행(calib/main 포함) 완료, b1-calib 14종 완료. **matched-FPR로 놓은 baseline**: Qwen3Guard-8B native R 0.862→0.534(ε̂ 0.33), 1k 임계값에서 28k bl FPR 0.728; 4B 0.826→0.365; CD(1,144행) 0.870→0.773(ε̂ 0.12); LG3-8B 0.381→0.182. LG 0.85(ε̂ 0.04–0.06). E2-7 완성: routing 손실 L 평균 ~1/4/3 pt(28k: 0.0/5.4/3.5), framing 손실은 benign-side가 대부분이고 twin 창 점수도 tail과 함께 오르므로 probe-위치 효과. E2-8 calib 선택: 세 seed 모두 router_gated_3 → Table 2에 `lg_k32_rg3` 행(`derive_agg`)을 같은 rows에서 유도해 추가. 4B/8B hold는 해제(v1 설계로 E2-9 스케일 결과를 10/1 전에), head v2(창 끝 landmark 토큰)는 s0 ablation으로.
* (9/21 오후, head v2 옵션) `lgs1.lgcore.model.Head(d, probe="landmark")`: 학습되는 d차원 landmark 벡터를 트렁크 깊이(layer ell)에서 창의 마지막 content 토큰 **뒤에** 붙이고(`Head.inject`, causal이라 content 토큰의 상태는 그대로), deep layer가 거기서 창 전체를 읽은 뒤 probe가 그 자리를 읽음 — 모든 창을 같은 위치에서, 내용 전체를 뒤에 두고 판정(base 가드가 chat template 접미 뒤에서 판정하는 것과 같은 구조). 파라미터 +d, deep layer 위치 +1/창. `head_forward`·`lg_window_logits`가 inject를 거치므로 학습·router target·valcheck·S5 모두 자동 적용; head.pt에 `head_probe` 기록, `load_lg`가 복원. 기본값 `probe=last`는 바이트 단위로 v1과 동일. run 디렉터리 변형: `S4Paths.run_dir(bb, seed, variant)` → `runs/v2/qwen3guard_0.6b_lm_s0`; `s4ctl t1/head/router/valcheck --variant lm --probe landmark`가 체인 전 단계에 `--variant`를 넘기고 잡 이름에도 붙임(`s4_head_qwen3guard_0.6b_lm_s0`); `s4ctl status`는 변형 run도 나열. S5는 `lg:qwen3guard_0.6b_lm_s0`로 지정. 테스트(tiny): inject 위치·학습/추론 forward 일치·landmark gradient·변형 체인 head→router→LG forward·s4ctl dry-run.
* (9/22 16:05, head v2 결과 — 승격 안 함) `qwen3guard_0.6b_lm_s0`: E1 W-avg F1 0.816(v1 0.807)·OR-Hard 0.044(0.062)로 짧은 프롬프트는 낫고 twin 점수가 위치 무관하게 낮아져 benign-side framing 손실이 0.055→0.002로 사라졌지만, Table 2 R@28k 0.820(v1 0.850)·ε̂ 0.082(0.043)·bl 팽창 0.116(0.084)으로 긴 L이 나쁘고 tail 0–15 결손(0.729 vs 0.83)은 그대로. → 문서 끝 결손은 probe 위치가 아니라 **왼쪽 문맥 희석**(needle 앞 benign 배경 길이)이고 landmark 한 자리로는 회복이 안 됨(harmful-side 0.015→0.064). v1 유지, v2는 ablation 행. 다음 후보: b=64(희석 ↓) 또는 토큰 단위 감독 — S6 뒤.
* `tests/test_s5_eval.py`: tiny 백본으로 head→router→dense 학습 뒤 e1/e2/oracle/needle/dense rows → 재개(todo=0) → k 유도 → 가짜 base·CD rows까지 합친 Table 1·2·E2-7 생성.

## 0.16 S8·S9 패키지 (9/27) — E6 ablation 13축, hard-negative head, tail-pad, Table 4·Fig 10·Fig 1: `s8ctl.py` + `lgs8/`, `lgs4/hardneg.py`

* `python s8ctl.py arms | mine | run --arms a,b | --axis E6-4 | tables | status` — arm 32개(infer / router / head), arm마다 (mine →) head → router → 평가 afterok 체인, 평가는 부분집합(200쌍 × 8 L × 5 격자 위치 {0, .2, .5, .8, 1} + benign-long, `--no-calib-main`). `docs/S8_runbook.md`.
* lgcore: `LGConfig.router_ells`(라우터가 읽는 트렁크 깊이, concat), `trunk_windows_multi`, `Router(in_dim)`, `RouterMLP`, `make_router`; `LandmarkGuard._trunk`로 forward/stream/sliding/oracle이 한 경로. `load_lg`가 router 체크포인트의 pool/depth/arch를 복원.
* `lgs4.router_train`: `--pool`, `--router-ells`, `--arch mlp`, `--objective distill_listwise|distill|window_bce|token_bce`, head cfg의 b 사용. `lgs4.head_train`: `--b`, `--hard-neg DIR --hard-per-row`, `--init-from head.pt|dense_ft.pt`, `--max-L`, `--needle-frac`, `--domains`, `--no-twin`(`lgs4.sampling.make_row_filter`, `HardNegIndex`, `Unit.kind == "hard"`). `lgs4.hardneg`: 기준 run의 router 상위 8창 × head 확률 채굴(재개 가능).
* S5 spec 옵션 `lg:<run>[:kK][:sS][:tpN]`(stride 덮어쓰기, 중립 접미사 tail-pad; method `lg_k32_s256`/`lg_k32_tp32`, `split_lg_method`로 k′·rg3 유도가 접미사 보존), `lgs5.run --no-calib-main`.
* `lgs8.table4`(기준과 arm을 같은 행에서 `table2`로; Δ 열; Fig 10 데이터), `lgs8.fig1`(Table 2 × Table 5 이름 대응, Pareto, $/1M docs).
* (9/27 저녁) **cfg_hash 회귀 수정**: 위 S8 코드가 `router_ells: None`·`tail_pad: 0`을 항상 해시에 넣어 모든 LG rows의 cfg_hash가 바뀌고 LG3-8B e2 requeue가 재개를 거부함 → 두 키는 값이 있을 때만(`lgs5.runners.lg_cfg_dict`; 기본 구성은 9/27 이전과 같은 해시), 9/27 낮에 쓰인 S8 추론 arm 파일은 `write_rows`가 그날 해시(`lg_cfg_dict_20260927`)를 알아보고 그대로 이어 씀. **부분집합 depth**: test 격자가 0.0, 0.1, …, 1.0이라 `{0, .25, .5, .75, 1}`은 3 위치였음 → `lgs8.config.ABL_DEPTHS = {0, .2, .5, .8, 1}`(16,000 test/main행), 끝난 추론 arm은 0.2/0.8 행만 id로 이어 붙음; `lgs5.run`·`lgs3.longbench`가 빈 depth에 `[depths] WARNING`; Table 4는 덜 끝난 arm을 `partial n/N`으로 표시하고 기준을 같은 행에서 다시 잼. B1-long·B2 sweep의 `0,0.25,0.5,0.75,1`도 3 위치(0, .5, 1)로 보고(`s6ctl.SUB_LONG` 주석). 145 tests.
* (9/28) 첫 Table 4(0.6B, 3 depth분): tail-pad가 depth 1.0 결손을 지움(tp64 pos-min@128k 0.435 → 0.795), hard negative는 팽창을 못 고침(hardneg_ft infl 0.164 → 0.314), 팽창은 rg3/k↓로만 줄어듦, k64·kall은 더 나쁨, stride256은 28k −6.5 pt. Table 4에 `R@1k`·`G2(a)` 열과 arm별 `+rg3`/`+k8` 유도 행 추가; 유도 행은 run당 하나의 native pass에서만(`lgs5.tables.native_source_k` — k64 파일 때문에 lg_k8·rg3가 이중 유도되던 것). `docs/S8_runbook.md` §3. 146 tests.
* (9/28 저녁) Table 4 전량(router 축 15개 완료): ℓ=9·max/top4·linear distill+listwise가 이미 최선(다른 목적·용량은 ±1.6 pt), pool_mean 붕괴, dom1 −15 pt, notwin +16 pt는 twin FPR의 대가, data10/25가 비단조 → seed 분산. 더 깊은 라우터(ell12/16)는 R@128k가 정확히 동률(0.693) — matched 임계값이 head 천장에 붙는 포화 현상 → Table 4에 `tau@128k`, `twinFPR@28k` 열과 s1·s2 seed 행(잡음 바닥). 0.6B G2 (a) 전 arm FAIL → 8B+tp64 전량 격자 e2 제출. `docs/S8_runbook.md` §3.
* (9/29) Table 4 seed 행: 잡음 바닥 R@128k ±2 pt·pos-min ±5 pt; twinFPR 열로 notwin(0.695)·dom1(0.48)의 착시 확인; b256 R@28k +4.9; len4k·data50 +7 pt(step 수가 적은 레시피) → E6-10 `ep1`/`ep2` arm 추가. 전량 격자 tp64 0.6B: R@128k 0.816·pos-min 0.795(G2 (a) FAIL 확정). S7 비용 spec `:tpN`/`:sS`(`lg_k32_tp64+c16+h8`) + g2_cost/Fig 1 대응. 클러스터의 `~`는 /home3라 `~/lg_s1.zip`이 9/26 판이었음 → 새 zip을 그 자리에 복사해 둘 것.
* (9/29 저녁) tp64 비용 측정(0.6B 1019 ms·+8.3 %, 8B 13.2 s·+1.5 %: 공짜). `lgs5.tables.load_eval`이 S8 부분집합 파일(depth 격자 75 % 미만)을 Table 2/g2_check에서 제외(`subset_files`; 9/29 13:05 표가 arm 33개로 오염). E6-10 ep1/ep2 평가 중.
* (9/30) E6-10 ep1/ep2 결과: ep1 +1.6·ep2 −0.6 pt(잡음 안) → 과학습 가설 기각, 주 구성 후보 k32+tp64(+rg3) 유지. `python s8ctl.py table --arms ep1,ep2` — Table 4 JSON에서 base·seed 행 + 고른 arm만 출력(`--axis E6-10`, `--no-base`). `b256_tp64` arm(b256 위 tp64, 48 GB) 추가. 계획을 OSDI '27(12/8)로 변경: `docs/plan_osdi_2026-09-29.md`.
* (9/30 정오) **G2 (a) 첫 PASS**: tp64 8B 전량 — `lg_k32_tp64` pos-min@128k 0.890 ≥ R@1k 0.933 − 0.05, `_rg3` 0.895(+1.0 pt), k8/k16_tp64도 PASS(여유 < 1 pt, 200쌍 SE ≈ 2 pt). 주 구성 k32+tp64+rg3 확정(8B); 4B·0.6B s1/s2 tp64 전량 e2 제출(S9′ 2). ep1/ep2 판정으로 head 재학습 없음(S9′ 3 종결).
* (9/30 오후) `s8ctl status`: 로그 glob을 `<job>_<jobid>.out`로 한정(b256이 b256_tp64의 로그를 집어 "queued"로 보이던 것; s5ctl/s6ctl도 같이), 로그는 있는데 rows도 잡도 없으면 `FAILED before the first rows -> .err`. b256_tp64 평가 잡 1037593이 첫 행 전에 사라짐 → .err 확인 후 재제출. tp64 전량 e2 4B·0.6B s1/s2 제출(1037908–10).
* (9/30 16시) `LG_EXCLUDE=n44` → 모든 sbatch에 `#SBATCH --exclude=n44`(`python s1ctl.py env-set LG_EXCLUDE=n44`로 고정, 빈 값이면 없음). b256_tp64 평가가 ECC GPU 노드에서 두 번 연속 시작 직후 죽어서 추가. 로그 머리의 `[common] host=`가 노드 이름.
* (10/1) 주 구성 **b128+k32+tp64+rg3 확정**: 0.6B tp64 3 seed pos-min@128k 0.795/0.775/0.740(FAIL 재현), b256 비용 +18 %·growth 0.218(G2 (c) FAIL)이라 b256은 ablation만, b256_tp64 0.6B 0.832/0.815(+rg3 0.836/0.825). g2_check에 `n@128k`·`rows` 열(평가 중 파일의 예비 행 식별). 변형 9종 평가를 주 구성으로 제출(`s6ctl var --models lg:qwen3guard_0.6b_s0:tp64 --subsets <9종>`). ECC 노드는 n84였음(`LG_EXCLUDE=n84`).
* (10/1 저녁) 인수인계 도구 `handover.py`(`docs/handover.md`): `sizes`(범주별 크기), `pack --what code,results,checkpoints,data[,prep,legacy_*] | all`(zip/tar.gz + sha256 + 파일 목록 + HANDOVER.md), `git --remote <GitHub> [--push]`(.gitignore·`env/s1.env.example`(비밀값 제거)·커밋·origin; env 파일 추적·토큰 스테이징이면 중단). `.gitignore`·`env/s1.env.example` 패키지에 추가. tests 150.

## 0.15 S6·S7 패키지 (9/22) — 64k·128k, 희석·FPR·변형, 장문 baseline, 비용: `s6ctl.py` + `lgs6/`, `s7ctl.py` + `lgs7/`

계획서 S6(E2-1·2 64k·128k, E2-4, E2-6, E3-1·2·3, B1·B2 장문, B4·B5·B6·B7, T5 LG3-8B, T6 dense-FT 8B)과 S7 E4(latency·throughput·peak memory·breakdown·k sweep). S5 결과값에 의존하지 않는 코드라 S5 잡이 도는 동안 제출 가능(같은 rows 파일을 쓰는 e2 잡만 S5 e2가 끝난 뒤). 순서·기준은 `docs/S6_runbook.md`.

```bash
python s6ctl.py all --sbatch          # e2-long(LG·dense 3 정책·sliding) + var(needle_repeat·decoy·framed·ood_domain) + b1-long + b2-long + b1-var + b6
python s6ctl.py status                # e2 파일별 64k 이상 행 수·진행, var, longbench 장문 행 수, T5/T6
python s6ctl.py tables                # reports/s5/table2_allL.md(+Fig 3·4 all L) + reports/s6/{fig5_dilution,fig6_fpr,table_variants,table2_cascade,g2_check}.md
python s7ctl.py e4 --sbatch           # E4 전 방법(--exclusive), 다른 GPU는 --partition … --gpu l40s;  python s7ctl.py tables → reports/s7/table5_cost.md·g2_cost.md
```

* (9/22) **윈도우 스트리밍** `LandmarkGuard.forward(stream=True)`/`forward_stream`: 64창 단위 trunk → 라우터 → 러닝 top-k 버퍼 → head. 판정·점수·선택 창이 전체 보관 경로와 동일(`tests/test_lgcore_stream.py`), 메모리는 L과 무관(G2 (c)). `lgs5.run --stream auto`는 32k 초과 행에만 적용(S5 ≤28k rows는 바이트 단위로 이전과 같은 경로). `sliding_scan`·`forward_batch`(E4-2 배치)·`PhaseTimer`(E4-4)·noskip/pad 비용 스위치(E6-13·E5-5)도 같은 파일.
* (9/22) dense 창 밖 정책 `dense:<run>:head|tail|yarn`(method `dense_ft`/`dense_ft_tail`/`dense_ft_yarn`), `lgs4.backbone.apply_yarn`(transformers 5의 `rope_parameters`는 rope_theta를 같이 넣어야 함 — 빠뜨리면 `None ** x`; longbench의 yarn 정책도 이 함수로), 패딩 없는 배치는 attention mask 생략(128k [L,L] 마스크 32 GB 회피).
* (9/22) `lgs5.run --exp var --subsets …`: 변형 서브셋 rows(여러 needle 스팬 → `needle_windows` 합집합), 변형 축·`needle_share`·`n_tok_needle`이 rows에 실림(S5 main rows에도 이제 실림; 옛 rows는 `needle_tok_span` 길이로 대체).
* (9/22) `lgs3.longbench`: `--max-bl N`(L당 benign-long 상한), `--prefill-chunk N`(causal 가드의 chunked prefill, `lgs1/zeroshot/prefill.py`: 캐시에 N씩 prefill 후 `generate` — greedy 출력이 단일 prefill과 같음을 tiny로 검증), 변형 축을 `meta`로 전달, classifier 가드(Prompt Guard)의 chunk-OR 토크나이저. `s3ctl.queued_longbench`가 s5/s6 컨트롤러의 배열도 인식.
* (9/22) T5: `Backbone.tok`(llamaguard3_8b = llama3) → `S4Paths.for_tok/for_backbone` → `s4/prep_llama3/`(`s4ctl prep --tok llama3`, D8 `tok/llama3` 스팬과 대조); 네 trainer가 백본의 토크나이저 경로를 씀(qwen3는 경로 불변). `s6ctl t5`가 prep → head → router → valcheck 체인, `t6`는 8B head에 afterok로 dense-FT 8B.
* (9/25) 시간 한도 자동 이어달리기: eval·longbench 잡이 `--signal=B:TERM --requeue`로 나가고 `lgs1.interrupt.Stop`이 현재 행까지 쓰고 rc=3 → 같은 id로 requeue(`tests/test_interrupt.py`). E4 첫 표: LG/sliding@128k 0.35(= ℓ/N 하한, G2 (b) 구조적 FAIL), peak +40 %는 64창 chunk transient → `lg:<run>:k32:c8|c16|c32` 스윕(G2 (c) fallback). gpt-oss는 48 GB에서 4k부터 OOM(eager attention) → 1k·2k만. `docs/S6_runbook.md` §3.5.
* (9/24) S7 운영 수정: `s7ctl e4`가 주 구성 대신 noskip/pad/nostream에 breakdown·ksweep·throughput을 3중 제출하던 필터 버그(`E4_MAIN`, `e4_plan` 중복 방지); E4 rows가 A5000/RTX 6000 Ada로 갈려 S7 잡은 기본 `--partition RTX6000ADA`(`LG_S7_PARTITION`, `any` = 티어) — 표·G2는 GPU 디렉터리 단위; Prompt Guard chunk 비용 오류(분류기 토크나이저 fallback); 128k-native base 가드는 `--prefill-chunk 8192` 기본; `--exclusive` 노드는 GPU 8장이 다 보여 `device_map="auto"`가 base/cd 가드를 8-GPU로 쪼개던 것(0.6B yarn cuda:0/cuda:7 오류) → S7 잡은 첫 GPU만(`GPU_ENV`, `lgs7.cost._one_gpu`, 행에 `n_gpus_visible`), 이전 base/cd 행은 `--force` 재측. E2-7 표에 `LG rg3`·`routing loss (rg3)` 열. 테스트: chunked prefill 동일성은 float32로(bf16 chunk=1 드리프트 flake).
* (9/23) 첫 S6 표: 128k 하락(R 0.78, pos-min 0.43)은 라우터(hit 0.95)·head(oracle 평평)가 아니라 32창 max 집계의 임계값 팽창 + depth 1.0 결손 — `docs/S6_runbook.md` §3.5. dense YaRN 128k OOM([L×L] 마스크 28 GiB) → dense 32k 초과 행은 KV 캐시 chunked prefill(`dense_logits_chunked`, 단일 패스와 동일 검증). cascade는 calib 순위 밴드(@e05/10/20) + 64k·128k는 8B yarn 행. oracle·needle 파일도 relineage 필요(9/21 lg만 했었음).
* (9/22) `lgs6/tables.py`: Fig 5(Benign-Fill vs Needle-Repeat, needle 길이 버킷별 recall), Fig 6(benign-long in-domain/OOD/도메인별·twin·decoy 소스별·framed frame별 FPR, 기본·matched), `table_variants`(서브셋 × L × 축), B7 cascade 유도(`derive_cascade`, band 0.2–0.8) + Table 2 형식, G2 (a). `lgs7/cost.py`·`lgs7/tables.py`: E4 러너(방법·L·배치·k별 p50/p99·docs/s·tokens/s·peak MB·단계 ms, GPU/드라이버/버전 메타, OOM 기록) → Fig 7·9 JSON, Table 5, G2 (b)(c). `s1ctl.write_sbatch(exclusive=, partition=)`.

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
python s1ctl.py zeroshot --sbatch --dep <tests jobid>    # E0-6 12종 job array (티어별, 동시 실행 제한 없음; --parallel N으로만 제한)
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
