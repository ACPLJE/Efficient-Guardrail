# S2 실행 순서 — 배경 스크리닝(D4), needle 풀·분할·감사(D5), 합성 dry-run(D6)

계획서 S2(9/20–9/22)의 5개 작업을 `s2ctl.py` 하나로 돌립니다. 모든 산출물은 `$LG_DATA_V2`(= `$LG_SOURCES`의 상위, 즉 `/home/212aig08/lg_data_v2`) 아래에 생기고, 로그·sbatch 스크립트는 `$LG_OUT`에 남습니다. E0-6(zero-shot 재현표)은 S1의 `s1ctl.py compare`가 그대로 담당합니다.

| 계획서 | 명령 | 어디서 | 산출물 · 통과 기준 |
| --- | --- | --- | --- |
| D4 배경 풀 | `python s2ctl.py haystack --sbatch` | 계산 노드(CPU 위주, gres 1) 수십 분 | `haystack/<domain>/docs.jsonl.gz` 10개 + `stats.json`; 도메인별 토큰이 예산×1.35 |
| D4 스크리닝 | `python s2ctl.py screen --sbatch --dep <haystack jobid>` | GPU 20 task(10 도메인 × LG3-8B·Qwen3Guard-8B), 동시 실행 제한 없음 | `screen/<domain>.<model>.rows.jsonl` 20개 |
| D4 집계 | `python s2ctl.py screen --summarize` | 로그인 노드 | `screen/screen_summary.json`(도메인별 통과율·점수 분포), `keep_<domain>.json`, `manual_review_50.md` |
| D4 임베딩 | `python s2ctl.py embed --sbatch` | GPU 10 task | `related/<domain>.emb.npy`(bge-m3, 스크리닝 통과 chunk만) |
| D5 needle 풀 | `python s2ctl.py needles --sbatch`(또는 `--local`) | CPU | `needles/pool.jsonl.gz`, `pool_stats.json`(소스×라벨×길이 버킷 표) |
| D5 분할·감사 | `python s2ctl.py splits --local` | 로그인 노드 ≈2분 | `splits/needles.json`, `haystack.json`, `audit.json`; gates PASS |
| D6 dry-run | `python s2ctl.py synth --dry-run --local` | 로그인 노드 ≈5분 | `synth_dryrun/…` ≈200행, `dryrun_report.json` validation PASS |
| 전부 | `python s2ctl.py all --sbatch` | haystack → screen array → (summarize·splits·dry-run) 자동 체인, needles 병렬, embed 마지막 | — |

## 1. 순서대로 (하나씩 확인하며)

```bash
cd /home/212aig08/lg_s1
python s2ctl.py status                          # 처음엔 전부 PENDING

# D4a 배경 풀 (소스가 없으면 [haystack] <domain>: FAILED FileNotFoundError → S1 download --only <source>)
python s2ctl.py haystack --sbatch               # 잡 id 기록
python s2ctl.py screen --sbatch --dep <haystack jobid>     # 20 task; 8B bf16 → 24 GB 티어 파티션 전부
python s2ctl.py needles --sbatch                # 배경과 무관하니 바로 병렬 제출

# 스크리닝 array가 끝나면 (squeue에 s2_screen 없음)
python s2ctl.py screen --summarize              # 통과율 표 + 수동 확인 파일
python s2ctl.py embed --sbatch                  # related 서브셋용 (S3에서 사용)
python s2ctl.py splits --local                  # 분할 + 외부 overlap 감사 + gates
python s2ctl.py synth --dry-run --local         # ≈200행 합성 + 스키마·규칙 검사
python s2ctl.py status
```

스크리닝이 끝나기 전에 dry-run만 먼저 보려면 `python s2ctl.py splits --local` 뒤 `python s2ctl.py synth --dry-run --allow-unscreened --local` (haystack 분할이 PROVISIONAL로 표시되고 status는 DONE이 되지 않음; 스크리닝 후 두 명령을 다시 돌리면 됨).

## 2. 무엇을 확인하나

**haystack/<domain>/stats.json** — `tokens`가 `collect_target`(예산 × 1.35)에 도달했는지. `reached_target: false`면 소스가 모자란 것(예: AgentDojo는 suite 데이터가 작아 3M에 못 미칠 수 있음 → 그대로 두고 데이터 카드에 실제 토큰을 적음).

**screen/screen_summary.json** — 도메인별 `pass_rate`, `kept_tokens ≥ budget_tokens`(`budget_met`), 모델별 `n_flag_chunks`·`n_unparsed`·`score_hist20`. `n_unparsed`가 chunk의 1%를 넘으면 템플릿 문제이니 `screen/<domain>.<model>.rows.jsonl`의 `raw`를 확인. `budget_met: false`인 도메인은 `haystack --domains <d>`를 `--budget-factor 2.0`으로 다시 만들고 그 도메인만 다시 스크리닝(`screen --sbatch --domains <d>`; 이미 판정한 chunk는 resume으로 건너뜀).

**screen/manual_review_50.md** — 탈락 문서 50개(도메인별 균등). 각 항목의 `verdict:` 줄에 agree/disagree를 적어 두면 데이터 카드의 스크리닝 정밀도 근거가 됩니다(자동으로 되살리지는 않음).

**needles/pool_stats.json** — `sources`에서 `status: ok`가 14개 전부인지(`missing`이면 S1 download). `n_unique`가 공개 수치와 5% 이상 다르면 로더 경고가 찍힘. `n_usable_main`(≤128 토큰)이 계획서 규모(test 200쌍·train 2,000쌍 등)를 넉넉히 넘는지.

**splits/audit.json** — `overlap.n_removed`(정확·근사 중복으로 train 풀에서 제거된 needle 수; 외부 세트별 내역), `gates`: `needle_groups_disjoint: true`, `test_source_rows_in_train_val_calib: 0`, `pass: true`. `haystack_screened: true`여야 확정본.

**synth_dryrun/dryrun_report.json** — `validation.pass: true`, `problems: []`. 검사 항목: 필드 존재, (split, subset, id) 유일성, `needle_char_span`이 needle 원문을 재현, |n_tok − L_target| ≤ 10%, 주 세트 needle ≤128 토큰, harmful 행마다 twin(같은 배경·depth·L, benign, 역링크), 배경 조각이 해당 분할 소속, 길이별 라벨 50:50.

## 3. 규칙이 코드에서 어떻게 고정돼 있나 (`lgs2/config.py`)

* 도메인 10개·역할·토큰 예산: `DOMAINS`. 학습 5(govreport·pg19·wildchat·stack·arxiv) + OOD 5(wiki_en·wiki_zh·pubmed·agentdojo·ledgar).
* 스크리닝: 512 토큰 chunk(Qwen3 토크나이저), LG3-8B + Qwen3Guard-8B strict, 하나라도 flag면 조각 폐기 — `SCREEN`.
* needle 소스 14개와 pool(train/test): `NEEDLE_SOURCES`. 공개 test 소스는 절대 train/val/calib에 들어가지 않음.
* group = 정규화 텍스트 해시 + MinHash(단어 5-gram, 128 perm, 32 band) Jaccard ≥ 0.8 — `MINHASH`.
* 분할: train 풀 group의 4% val, 4% calib(seed 20260917); 배경 조각은 test 30% / calib 15% / val 5% / train 50%, OOD는 test 100% — `SPLITS`.
* 합성 레시피(길이 그리드·위치·쌍 수·twin ±20%·길이 ±10%·문단 경계 삽입·배경 1분할) — `RECIPE_V2`. 이 dict의 해시가 모든 행의 `recipe_hash`.

## 4. 자주 나올 문제

| 증상 | 원인 · 조치 |
| --- | --- |
| `[haystack] wildchat: FAILED FileNotFoundError` | 스트리밍 소스가 없음 → 로그인 노드에서 `bash slurm/s1_download_login.sh --only wildchat` |
| `[needles] wildguardtrain MISSING` | gated 저장소 미수신 → `huggingface-cli login` 후 `--only wildguardmix` |
| screen task가 `[skip] … not in the HF cache` | LG3-8B/Qwen3Guard-8B 캐시 없음(S1 D3) 또는 refs/main 없음 → `python -m lgs1.download --fix-refs` |
| `haystack split is provisional` (synth) | `screen --summarize` 전 → `--allow-unscreened`로 dry-run만, 확정본은 스크리닝 후 |
| dry-run `label imbalance` | 해당 split의 benign needle 부족(twin 실패) → pool_stats의 usable 수 확인 |
| dry-run `|n_tok − L_target| > 10%` | 도메인 조각이 작거나 문단이 너무 김 → `haystack --max-piece-tokens 4096`으로 재생성 후 재검사 |
