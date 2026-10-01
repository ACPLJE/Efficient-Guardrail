# 인수인계 패키징 — `handover.py` (10/1)

작업 전체는 다섯 덩어리로 나뉘고, 각 덩어리의 위치는 `env/s1.env`가 정의한다(모든 ctl과 같은 경로 해석).

| 범주 | 위치(env 키) | 내용 | 기본 포함 | 형식 |
| --- | --- | --- | --- | --- |
| `code` | `LG_S1_ROOT` = 이 패키지 | lgs1–lgs8, s*ctl.py, slurm/, docs/, tests/, requirements.txt. `env/s1.env`는 빼고 비밀값을 지운 `env/s1.env.example`을 대신 넣음 | 예 | zip |
| `code_v04` | `LG_V04_ROOT` | v0.4 코드(`.venv` 제외) | 예 | zip |
| `data` | `LG_DATA_V2`(= `LG_SOURCES`의 상위) | sources/·needles/·splits/·synth/·manifests/·tok/·screen/·related/ — S2/S3가 봉인한 데이터 | 예 | tar.gz |
| `results` | `LG_OUT` | eval/·cost/·longbench/ rows, reports/, logs/, slurm/, runs/v2/*/{meta.json, router.pt, hashes.json, *.log.jsonl, hardneg/} — **체크포인트(head.pt·dense_ft.pt)와 s4/prep은 제외** | 예 | tar.gz |
| `checkpoints` | `LG_OUT/runs/v2` | run 디렉터리 통째(head.pt·dense_ft.pt·router.pt…). `--runs main`(기본: qwen3guard_{0.6b,4b,8b}_s*, llamaguard3_8b_s*) / `--runs all`(S8 arm 35개까지; arm의 head.pt는 base로 가는 상대 심볼릭 링크 그대로) | 예 | tar.gz |
| `prep` | `LG_OUT/s4/prep` | 토큰 배열(`python s4ctl.py prep`으로 재생성 가능) | 아니오 | tar.gz |
| `legacy_*` | `LG_V04_RUNS`, `LG_PAPER_ROOT`, `LG_EXTERNAL_*`, `LG_DATA_V1` | 이전 라운드 | 아니오 | tar.gz |
| (hf, venv) | `HF_HOME`, `LG_PYTHON` | 싸지 않음 — HANDOVER.md에 모델 repo 목록과 `requirements.txt` 복원 명령 | — | — |

모든 아카이브에 `<name>.sha256`(검증: `sha256sum -c <name>.sha256`)과 `<name>.files.txt`(멤버 목록·크기)가 붙고, `--dest`에 `HANDOVER.md`(아카이브 표, 모델 목록, 새 머신 복원 순서)가 생긴다.

## 명령

```bash
cd /home/212aig08/lg_s1
python handover.py sizes                                   # 범주별 크기·파일 수(쓰는 것 없음) — 먼저 이걸로 디스크 계획
python handover.py pack --what code,results                # 작은 것부터: $LG_HOME/handover/{code.zip, results.tar.gz, ...}
python handover.py pack --what checkpoints                 # base run 체크포인트
python handover.py pack --what data                        # 데이터(수십 GB 가능; --dest로 다른 디스크 지정 가능)
python handover.py pack --what checkpoints --runs all      # S8 arm 체크포인트까지 (크면 생략)
python handover.py git --remote https://github.com/ACPLJE/Efficient-Guardrail.git          # .gitignore, env 예시, 커밋, origin
python handover.py git --remote https://github.com/ACPLJE/Efficient-Guardrail.git --push   # 푸시(토큰/SSH 필요)
```

`git`은 `env/s1.env`가 추적되거나 스테이지된 파일에 `hf_…`/`ghp_…` 토큰이 보이면 중단한다. HTTPS 푸시는 GitHub PAT(Settings → Developer settings → Fine-grained token, 해당 repo Contents: read/write)를 비밀번호로 넣거나 `git remote set-url origin git@github.com:ACPLJE/Efficient-Guardrail.git`로 SSH를 쓴다. 로그인 노드에서 GitHub로 나가는 HTTPS가 막혀 있으면 `code.zip`을 노트북에 풀고 거기서 같은 `git` 명령을 실행하면 된다(코드만 있으면 됨).

## 복원(새 머신)

`HANDOVER.md`의 순서: code 풀기 → `env/s1.env.example`을 `env/s1.env`로 복사해 경로 수정 → venv(`requirements.txt`) → data·results·checkpoints 풀기(아카이브 루트가 옛 디렉터리 이름이라 그 상위에 풀고 env의 경로를 맞춤) → 모델 다운로드(gated는 `HF_TOKEN`) → `python -m pytest tests -x -q`, `python s5ctl.py status`, `python s8ctl.py status`, 표는 `python s6ctl.py tables; python s8ctl.py tables`로 rows에서 다시 생성.
