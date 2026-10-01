# S3 실행 순서 — 전량 합성(D6), 변형 10종(D7), 모델별 토큰화(D8), 게이트·봉인(E0-4/G0), 데이터 카드(D10), 파일럿(E0-5), baseline 장기 잡(B1/B2)

전제: S2가 끝나 있어야 합니다 — `s2ctl.py status`의 6항목 DONE(스크리닝 완료·keep 목록·splits 확정·dry-run PASS). 모든 명령은 `/home/212aig08/lg_s1`에서, `--local`은 자동으로 venv 파이썬을 씁니다.

| 계획서 | 명령 | 자원 | 산출물 · 통과 기준 |
| --- | --- | --- | --- |
| D6 주 세트 합성 | `python s3ctl.py synth --sbatch` | 잡 4개(test·calib·val·train), CPU 위주, 수 시간 | `synth/<split>/{main,benign_long,main16k,native}/rows.jsonl.gz` + `manifests/*.manifest.json`; test 35,200+4,000행 등 계획서 §5.7 규모 |
| D7 변형 서브셋 | `python s3ctl.py variants --sbatch --dep <test synth jobid>` | array 9(24 GB 티어; related는 bge-m3) | `synth/test/<variant>/rows.jsonl.gz` 9개 ≈30,000행 |
| D8 모델별 토큰화 | `python s3ctl.py tokspans --sbatch --dep <synth+variants jobids>` | array 20(5 토크나이저 × 4 split) | `tok/<model>/<split>/<subset>/spans.jsonl.gz` + `report.json`(qwen3 ±10% = 1.0, 복원 100행 PASS) |
| E0-4 게이트 7개 | `python s3ctl.py gates` | 로그인 노드 | `splits/audit.json["gates_e04"]` 전부 PASS, `manifests/lg_data_v2.manifest.json` |
| G0 봉인 | `python s3ctl.py gates --seal` | 로그인 노드 | `lg_data_v2/SEALED`, 디렉터리 읽기 전용 |
| D10 데이터 카드 | `python s3ctl.py datacard` | 로그인 노드 | `lg_data_v2/DATACARD.md`, `reports/table_A1_draft.md` |
| E0-5 파일럿 | `python s3ctl.py pilot --sbatch` | GPU 1, 수 분 | `reports/e05_pilot.json` PASS(E1·E2·E3·B5·E7·E7-7 러너 스키마·계보 열) |
| B1 장기 잡 착수 | `python s3ctl.py b1-start --sbatch` | array 3개(24/40/48 GB 티어), 며칠 | `$LG_OUT/longbench/{v2main,safetyniah}/<guard>.<policy>.rows.jsonl` |
| 전부 | `python s3ctl.py all --sbatch` | synth → variants → tokspans → (gates·datacard·pilot) 의존성 체인 | 끝나면 `gates --seal` → `b1-start --sbatch` |

## 1. 순서대로

```bash
cd /home/212aig08/lg_s1
python s2ctl.py status                       # 6항목 DONE 확인 (특히 D5b splits 'screened', D6 dry-run PASS)
python s3ctl.py all --sbatch                 # 4 synth 잡 + variants array + tokspans array + finish 잡 (afterok 체인)
python s3ctl.py status                       # 진행 확인 (D6 8세트, D7 9종, D8 5개 토크나이저, 게이트, 파일럿)

# finish 잡(gates·datacard·pilot)이 끝난 뒤
python s3ctl.py gates --seal                 # 7개 게이트 PASS면 lg_data_v2를 읽기 전용으로 봉인 (G0)
python s3ctl.py b1-start --sbatch            # zero-shot 12종 v2 test ≤28k + SafetyNIAH (CD 2종 포함) 착수
python s3ctl.py longbench --summarize --bench v2main       # 언제든 중간 집계
python s3ctl.py longbench --summarize --bench safetyniah
```

### 일부만 다시 돌릴 때 (synth 한두 split이 rc=2로 죽어 tokspans가 `DependencyNeverSatisfied`가 된 경우)

`synth` 로그의 `len_bad=N`이 0이 아닌 split만 다시 만들고, 그 뒤 체인을 손으로 이어 줍니다. 실패한 세트는 manifest가 지워지므로 `status`의 D6가 `7/9`처럼 줄어듭니다.

```bash
scancel <tokspans jobid> <finish jobid>
python s3ctl.py synth --splits calib,train --sbatch          # 죽은 split만; 출력의 {'calib': J1, 'train': J2}
python s3ctl.py tokspans --sbatch --dep J1:J2:<variants jobid>   # 5 토크나이저 × 4 split 전부 (다른 split의 spans도 최신 rows로)
python s3ctl.py finish --sbatch --dep <tokspans jobid>       # gates + datacard + pilot
```

변형 하나가 rc=2로 죽었을 때(예: 9/18 framed)도 같습니다 — 그 변형만 다시 만들고 tokspans·finish를 다시 겁니다:

```bash
python s3ctl.py variants --variant framed --sbatch          # -> J3 (실패한 변형은 manifest가 없어 status D7에 빠져 있음)
python s3ctl.py tokspans --sbatch --dep <아직 큐에 있는 synth jobid>:J3
python s3ctl.py finish --sbatch --dep <tokspans jobid>
```

### E0-4 게이트가 FAIL일 때

`python s3ctl.py status`의 `E0-4 gates FAIL: 3=FAIL …` 뒤 숫자는 `splits/audit.json`의 `gates_e04`에 있습니다:

```bash
python - <<'EOF'
import json
g = json.load(open("/home/212aig08/lg_data_v2/splits/audit.json"))["gates_e04"]["gates"]
for k, v in g.items(): print(k, {a: b for a, b in v.items() if a not in ("intersections", "pass_rates")})
EOF
```

게이트 3 `duplicate_ids`(같은 id 행) → `python s3ctl.py dedupe --split <split> --subset <subset>`; 게이트 4는 native가 예외(9/19). finish 잡은 gates 실패 시 datacard·pilot을 건너뛰므로 고친 뒤 세 단계를 다시:

```bash
python s3ctl.py finish --cpu          # 로그인 노드에서 gates + datacard + pilot(CPU), 몇 분; GPU 큐로 보내려면 --sbatch
python s3ctl.py status
python s3ctl.py gates --seal
```

### CPU 파티션 (선택)

synth·tokspans·finish·prep은 GPU를 쓰지 않는데 QOS normal이 `--gres=gpu:1`을 요구해 GPU 큐에서 1시간 넘게 기다렸습니다(9/18). CPU 파티션이 QOS 없이/다른 QOS로 받아 주면:

```bash
sbatch --test-only -p cpu-max24,cpu-max16 -c 8 --mem=32G -t 01:00:00 --wrap hostname              # 통과 → 아래 첫 줄만
sbatch --test-only -p cpu-max24,cpu-max16 --qos=normal -c 8 --mem=32G -t 01:00:00 --wrap hostname # 이것만 통과 → LG_CPU_QOS=normal 추가
python s1ctl.py env-set LG_CPU_PARTITIONS=cpu-max24,cpu-max16
python s1ctl.py env-set LG_CPU_QOS=normal        # 두 번째 probe만 통과했을 때
```

이후 `s2ctl`/`s3ctl synth·tokspans·finish·pilot`/`s4ctl prep`는 그 파티션으로 `--gres` 없이 나가고(`[sbatch] …: CPU job -> partitions …` 로그), GPU 잡(variants·longbench·학습)은 그대로입니다. 비워 두면 예전처럼 8 GB GPU 티어.

64k·128k dense baseline 부분집합(계획서 S6: 128쌍 × 5 위치)은 나중에 이렇게 제출합니다(48 GB 티어 자동):

```bash
python s3ctl.py longbench --guards qwen3guard_gen_8b,llamaguard3_8b,gpt_oss_safeguard_20b --bench v2main --policy native \
    --L 65536,131072 --max-pairs 128 --depths 0,0.25,0.5,0.75,1 --sbatch
python s3ctl.py longbench --guards qwen3guard_gen_8b --bench v2main --policy yarn --L 65536,131072 --max-pairs 128 --depths 0,0.25,0.5,0.75,1 --sbatch
python s3ctl.py longbench --guards qwen3guard_gen_8b --bench v2main --policy chunk_or:512:256 --sbatch          # CD 기본 config 전량
```

## 2. 무엇을 확인하나

**synth** — `manifests/<split>.<subset>.manifest.json`의 `rows`가 레시피 규모와 같은지: test/main 35,200, test/benign_long 4,000, calib/main 16,000, calib/benign_long 4,000, val/main 7,200, val/benign_long 1,200, train/main 32,000, train/main16k 2,000, train/native ≈ train needle 수. 모자라면 로그(`$LG_OUT/logs/s3_synth_<split>_*.out`)에 `only N usable harmful needles` 같은 SystemExit이 있음.

**variants** — 9개 manifest; `related`가 없으면 `related/<domain>.emb.npy`(S2 embed) 부재. `fragmented` 행의 `piece_gap_tokens ≥ 256`.

**tokspans report.json** — qwen3: `len_within_tol = 1.0`, 모든 모델 `recon_pass = true`; `needle_fully_contained`는 주 세트 1.0(long_needle 서브셋만 < 1).

**gates** — 7개 전부 PASS여야 `--seal`이 동작. FAIL이면 항목별 카운트가 찍힘(예: `3 … rows_without_qwen3_span`은 tokspans 미완료).

**pilot** — `reports/e05_pilot.json`의 `pass: true`; 각 러너의 `problems: []`. 숫자(recall 등)는 랜덤 초기화 백본이라 의미 없음.

**longbench** — `--summarize`가 guard × policy × L 별 R_u·R_s·FPR·F1_unsafe·AUPRC·parse fail·budget×·truncation·ms를 표로. `error` 행(OOM·context 초과)은 `n_error`로 따로 세며 Table 3의 "판정 불가"가 됨.

## 3. 자원·시간 감

* synth: test 세트는 행당 최대 128k 토큰이라 텍스트 ≈ 8 GB gzip, 토큰 세기(Qwen3)가 지배 → 잡당 1–4시간.
* variants: related만 bge-m3 GPU, 나머지 CPU. tokspans: 토크나이저 5 × 4 split, 각 30분–2시간.
* B1 ≤28k: 12 가드 × 29,400행 → 8B 가드는 행당 1–3초 → 가드당 10–25시간(48 h 제한, resume 가능). LG4는 40 GB, gpt-oss는 48 GB 티어.
* SafetyNIAH: 30,400행(prompt-side 15,200) × 12 가드 + CD 2종. 32k-단어 행은 32k-ctx 가드에서 context 초과 → error 행으로 기록(LongGuard의 unparseable에 해당).
