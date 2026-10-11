# AG S1 걷는 뼈대 — 실행 안내

S1(구축 계획 4.1절)의 결과물을 클러스터에서 처음 돌릴 때 필요한 명령만 모았다. 인터페이스와 결과 파일 형식은 `docs/SPEC.md`(v1.3)가 기준이다.

## 1. venv (D-001, 로그인 노드에서 한 번)

```bash
/home3/212aig08/.local/bin/python3.11 -m venv /home/212aig08/ag_venv/ag311
source /home/212aig08/ag_venv/ag311/bin/activate
pip install -r requirements.txt          # LG 테스트에 pandas·pyarrow·datasets가 필요하다(D-023)
pip install -e ".[reference,test]"       # ag 패키지 + lgs* (AG가 유해 경로로 import)
cp env/ag.env.example env/ag.env         # 경로 확인. env/ag.env는 커밋하지 않는다
python agctl.py doctor
```

## 2. 뼈대 실행 (CPU, mock, 가상 시계)

```bash
export AG_OUT=/home/212aig08/ag_results
python -m ag.harness.run --config configs/exp/skeleton.yaml      # raw/mock/<run_id>/ 두 개 (B0, AG)
python plots/fig09.py --exp mock --run-id <B0 run_id> --allow-mock  # $AG_OUT/figures/fig09.pdf
python plots/abstract_numbers.py                                   # $AG_OUT/numbers.json (7개 키, 지금은 null)
```

- mock run은 `latest.txt` 자격이 없다(SPEC 7.1). 그림 스크립트는 mock·가상 시계·dirty run을 기본으로 거부하고, 테스트에서만 `--allow-mock`을 쓴다.
- B0 run이 Record를 남기고 S와 보고용 hit@k를 정한다. AG run은 `lineage.parent_run_ids`로 B0 run을 가리킨다(SPEC 6.1).
- 같은 run을 이어 쓰려면 `--ts <처음 expand 시각>`을 다시 준다. resume 키가 다르면 exit 2로 끝난다(SPEC 7.2).

## 3. Slurm (헤더 검증만)

```bash
python agctl.py run --config configs/exp/skeleton.yaml --sbatch            # 스크립트 생성 + sbatch --test-only
python agctl.py run --config configs/exp/skeleton.yaml --sbatch --submit   # test-only 통과 뒤 제출
```

B0 wave와 방어 시스템 wave를 따로 만들고, 두 번째 wave는 `--dependency=afterok:<첫 wave job id>`로 제출한다. `%N` throttle은 쓰지 않는다(CLAUDE.md 4절). 생성 스크립트는 `slurm/ag_common.sh`를 source한다.

## 4. 테스트 (PR 전 두 명령, CLAUDE.md 3.4)

```bash
python -m pytest tests -x -q
pytest $(cat tests/manifests/cpu.txt)
python tools/check_test_classification.py     # 모든 test 모듈이 manifest 하나에만 있는지
```

GPU 테스트(`tests/gpu/`)는 CUDA가 없으면 skip된다. 그림 fixture(`tests/fixtures/results_mock/`)는 `python tools/make_mock_fixtures.py`로 다시 만든다.

## 5. S1에서 stub으로 둔 것

실제 Target·Judge, `sglang_client`(S1E), judge pool·scheduler(MockSystem 안의 단일 worker 모형), EngineAdapter replay(S3)·sglang(S3E), 집계 B·rollout(S2), 실모델 참조 경로(SDPA prefill, S2), LG 유해 경로 연결(S3), `engine_integration/`(README만).

## 6. S2 step 0: 실제 모델 B0 pilot (W0-grader, D-041)

T2(Qwen3-4B-Instruct-2507)로 8k 초안 요청 200건을 생성한다. 출력은 사람이 라벨해 grader 일치율(W0-grader)을 재는 데 쓴다. 규모는 CLAUDE.md 3.4 순서대로 키운다.

```bash
# 0) 로그인 노드: T2와 배경 문서 받기 (S0 다운로더, env/ag.env에 AG_SOURCES)
bash slurm/ag_download_login.sh --stage pilot

# 1) CPU 연기 시험: 작은 로컬 HF 모델(tiny-hf)로 같은 경로 (GPU·다운로드 불필요)
python -m ag.harness.run --config configs/exp/w0_grader_smoke.yaml

# 2) GPU 1요청 (계산 노드, srun으로 바로)
srun -p RTX6000ADA --qos=normal --gres=gpu:1 -c 8 -t 00:30:00 \
  bash -c 'source slurm/ag_common.sh && $PY -m ag.harness.run --config configs/exp/w0_grader_pilot.yaml --limit 1'

# 3) 200건 전량 (sbatch --test-only 뒤 제출)
python agctl.py run --config configs/exp/w0_grader_pilot.yaml --sbatch --submit

# 4) 라벨 시트 (run 디렉터리 안에 label_sheet.csv, label_sheet.jsonl)
python scripts/export_label_sheet.py --run-dir $AG_OUT/raw/W0-grader/<run_id>
```

- 데이터는 초안이다(D-041): Wikipedia 배경 + 정답이 필요 없는 과업 5종 + Open-Prompt-Injection 방식 payload 4종 × 무해한 주입 목표 12개, 위치 10·50·90 %. 실제 원천 5개는 S2 ②.
- 이 run의 `success.jsonl`은 `pending@1`이라 비어 있고(`attack_success = None`), `summary.json`의 ASR은 `n_unlabelled`로 남는다. 라벨과 grader 비교는 S2 ①.
- 자산이 없으면 `ag.harness.run`은 exit 4와 함께 받을 명령을 알려 준다. ledger(attention) 기록은 아직 없다(S2 step 3, E0.1).
- 라벨 시트: `label_sheet.csv`에서 =·+·-·@·TAB·CR로 시작하는 텍스트 칸(출력, 과업, 주입 문구 등)은 앞에 `'`가 붙는다(엑셀 수식 방지). 원문은 `label_sheet.jsonl`. 라벨을 채운 시트(또는 cp949 등으로 다시 저장해 읽을 수 없는 시트)는 다시 export해도 덮어쓰지 않는다(`--force`).
- `--limit 1` run은 job의 요청을 다 덮지 못한 partial run이라 `latest.txt`와 그림에서 빠진다(`summary._meta.partial`). 같은 `--ts`로 나머지를 이어 돌리면 완전한 run이 된다.
- 요청당 시간(D-025)은 `events.jsonl`의 PREFILL_DONE·TOKENS 시각과 FINISHED `payload.perf`(peak 메모리, prefill·decode ns)에 있다.
