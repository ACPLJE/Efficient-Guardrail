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

## 1b. 자산 다운로드 (S0, 로그인 노드만)

```bash
cp env/ag.env.example env/ag.env   # 이미 있으면 AG_SOURCES=/home/212aig08/ag_sources 한 줄만 추가
bash slurm/ag_download_login.sh --list          # 자산 목록 (stage, gated, verify 표시)
bash slurm/ag_download_login.sh --stage pilot   # T2 + bg_wiki_en (S2 step 0 pilot에 필요한 것만)
bash slurm/ag_download_login.sh                 # 인자 없으면 pilot + s2 (T1, grader, judge, 원천 데이터, Mooncake)
python agctl.py assets                          # 상태표 ($AG_SOURCES/download_status.json)
```

- nohup으로 돌고 로그는 `$AG_OUT/logs/ag_download_<MMDD_HHMMSS>.log`다. gated 모델(T1, Prompt Guard 2)은 승인 뒤 `HF_TOKEN`을 export하고 다시 돌린다. 날짜별 `$AG_SOURCES/ag_assets_<MMDD>.json`에 gated 상태가 남는다.
- `verify` 표시가 붙은 id는 확인하지 못한 값이다. 실패하면 `revision.json`의 `error`와 `alternatives`를 보고 알려 준다.
- 원천 데이터마다 `probe.json`(파일 목록과 첫 행 키)이 생긴다. S2 ②의 원천 adapter는 이것을 보고 만든다.

### W0-mooncake (CPU)

```bash
python scripts/mooncake_stats.py --name fast25 \
  --trace conversation=$AG_SOURCES/mooncake/repo/FAST25-release/traces/conversation_trace.jsonl \
  --trace toolagent=$AG_SOURCES/mooncake/repo/FAST25-release/traces/toolagent_trace.jsonl --out $AG_OUT
```

`$AG_OUT/raw/W0-mooncake/<run_id>/trace_stats.json`에 길이 분포, D-020 길이 등급, `hash_ids` 512토큰 단위 확인(F-11), 20분 구간 부하(D-022)가 나온다. trace 파일 이름은 clone 뒤 확인한다.

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
