# S1 산출물과 통과 기준 (계획서 §2 S1 표 ↔ 이 패키지)

| 계획서 항목 | 실행 | 산출물 | 통과 기준 | 어디서 확인 |
| --- | --- | --- | --- | --- |
| D1 경로 확인 | `python s1ctl.py assets [--seal]` | `$LG_OUT/reports/assets_check_<MMDD>.md/.json` | 항목 전부 OK; Round 3는 `--seal` 후 writable=no | `s1ctl.py status` |
| D2 쿼터 | `python s1ctl.py quota` → 후보 확인 → `--apply-move` | `$LG_OUT/reports/quota_<MMDD>.md/.json` | 여유 ≥150G (quota 출력에서 확인) | 보고서 첫 블록 |
| E0-1 등가성 | `python s1ctl.py tests` (또는 `--sbatch`) | `reports/pytest_e0.log/.xml/.status` | `tests/test_correctness.py` 전부 green (a)–(f) | `.status` = PASS |
| E0-2 계보 해시 | 같은 실행 | 같은 파일 | `tests/test_lineage.py` green; `python -m lgs1.lineage check` 불일치 시 exit 2 | `.log` |
| E0-3 집계 규약 | 같은 실행 | 같은 파일 | `tests/test_metrics.py` green (유일성, 엄격 matched-FPR, 도메인 hard-fail) | `.log` |
| D3 다운로드 | `bash slurm/s1_download_login.sh` (로그인 노드) | `$LG_SOURCES/<name>/revision.json`, `download_status.json` | failed=0; `verify`/`gated` 항목은 대체 소스 기록 | `s1ctl.py status` |
| E0-6 zero-shot | `python s1ctl.py zeroshot --sbatch` → `python s1ctl.py compare` | `$LG_OUT/zeroshot/<guard>/<bench>.rows.jsonl/.summary.json`, `reports/e06_reproduction.md` | 공개 수치 있는 행 전부 PASS(±1pt); 파싱 실패율 기록 | `e06_reproduction.md` |
| M1 확인 | `python s1ctl.py m1` → 파일 편집 | `reports/M1_decisions.md` | 4건 결정 + `DECIDED` | `s1ctl.py status` |

## run_all.sh 훅 (E0-1 통과 기준의 두 번째 절반)

v0.4 `run_all.sh`의 모든 STAGE 분기 앞에 다음 두 줄을 넣습니다(테스트가 깨지면 어떤 실험도 시작되지 않음):

```bash
( cd "$LG_S1_ROOT" && "$PY" -m pytest tests/ -x -q >/dev/null ) || { echo "[run_all] E0 tests failed; refusing to start STAGE=$STAGE"; exit 3; }
"$PY" -m lgs1.lineage check --rows "$ROWS" --model "$MODEL" --router "$ROUTER" --cfg "$CFG_JSON" --recipe "$RECIPE" --manifest "$MANIFEST" || exit 2
```

`$ROWS/$MODEL/$ROUTER/$CFG_JSON/$RECIPE/$MANIFEST`는 run_all.sh가 이미 갖고 있는 변수 이름으로 바꿉니다. 두 번째 줄이 resume·rescore 보호(E0-2)이고, 새 출력 경로면 기존 행이 0개라 항상 통과합니다.
