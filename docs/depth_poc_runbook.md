# 깊이 비대칭 PoC 실행 안내 (E0.3 / E1.1)

Oct 11, 2026 · 계획서 `asymmetric-guard-osdi27-plan-2026-10-10.md` §2.2의 관찰 1(깊이), §8.2의 E0.3, §8.3의 E1.1을 돌리는 코드다. 코드는 `agdepth/` 패키지, 진입점은 `depthctl.py`다.

> **지위.** 이 코드는 구축 계획 S2b의 `ag/probes/depth_probe.py`가 아니라 그 전에 쓴 PoC다. 결과는 SPEC 7.1 트리(`$AG_OUT/raw/`) 밖에 쓰고, plots/와 latest.txt에 섞이지 않는다. 계획서·SPEC과 다른 점과 결정이 필요한 항목은 `docs/DECISIONS.md` D-041에 있다. 특히 (a) 코드 위치, (b) injection 양성 출처, (e) G0에 쓰는 게이트(`gate` 또는 `gate_plan`), (h) G0에 쓰는 framing·pooling은 강지의의 결정이 필요하다.

값의 출처 표기는 CLAUDE.md 3.5를 따른다. \[측정\] 우리가 잰 값, \[계산\] 산술, \[문헌\] 문서·소스에서 확인한 값, \[추정\] 어림값, \[해석\] 판단.

## 1. 무엇을 보이는가

**주장.** 같은 backbone, 같은 선형 probe로 보면 유해 콘텐츠 창은 얕은 층에서 정상 창과 갈린다. 반면 injection 창은 지시문이 많은 정상 창과 깊은 층에서야 갈린다.

**게이트(계획서 E0.3, 제안값).** ℓ95(injection 대 지시문이 많은 음성) ≥ 1.5 × ℓ95(유해).

보고서는 두 가지로 판정한다.

| 이름 | 유해 쪽 음성 | 비고 |
| --- | --- | --- |
| `gate_plan` | 일반 정상(plain) | 계획서 그대로 |
| `gate` | plain + harm_hard | PoC 제안 |

G0에 무엇을 쓸지는 D-041 (e)에서 정한다.

- 주 지표는 창 FPR 2×10⁻⁴의 TPR이고, 고정이다(계획서 §8.2, §9.1, SPEC 6.5).
- ℓ95는 그 TPR이 층별 최고값의 95 %에 처음 닿는 층이다.
- 음성이 기대 오탐 5개를 못 채우면 지표를 바꾸지 않고 판정을 INCONCLUSIVE로 낸다.
- `--primary-fpr`로 다른 FPR을 줄 수 있고, 보고서에 그 사실이 적힌다.

판정은 3종이다.

| 판정 | 뜻 |
| --- | --- |
| PASS | 비율이 1.5 이상 |
| FAIL | 비율이 1.5 미만 |
| INCONCLUSIVE | 곡선 최고값이 TPR 0.2(AUROC 0.7) 미만이거나, 그 FPR을 재기에 음성이 모자라다 |

비율은 ℓ95_inj / max(ℓ95_harm, 1)이다(D-041 (g)).

## 2. 실험 설계

### 2.1 창

창은 배경 문서의 연속 구간 사이에 조각 하나를 독립된 문단으로 끼운 것이다. 라벨은 끼운 조각의 종류다.

- 길이는 기본 256토큰(Qwen3 tokenizer 기준)이다.
- 조각 길이는 12–112토큰이다. LG v2의 needle 상한 128을 따랐다.
- 조각 위치는 창 안의 상대 깊이 U(0.05, 0.95)이고, 가까운 줄 경계나 문장 경계에 맞춘다. 조각이 창의 첫 토큰이 되는 일은 없다.
- 모든 클래스를 같은 절차로 만든다.

| 클래스 | 내용 | 출처 |
| --- | --- | --- |
| `harm` | 유해 요청 | LG v2 needle pool(`needles/pool.jsonl.gz`, label harmful). 없으면 S2 loader로 원천 파일을 메모리에서 읽는다 |
| `harm_hard` | 정상 요청. OR-Bench, XSTest처럼 과잉 거부를 부르는 문장 포함 | LG v2 needle pool, label benign (계획서에 없는 클래스, D-041 (e)) |
| `inj` | prompt injection payload | 공개 데이터: BIPIA 공격 지시문(text·code), deepset/prompt-injections(label 1), LLMail-Inject, Gandalf (D-041 (b)). LLMail-Inject는 **항상 test 전용**이고 게이트에서 빼고 `inj@heldout`으로 따로 보고한다(D-041 (c)) |
| `instr` | 사람에게 하는 지시문이 많은 정상 텍스트 | 템플릿 40종 + 배경 문서에서 명령문이 2개 이상이고 40 % 이상인 문단 |
| `notinject` | trigger 단어가 든 정상 prompt | NotInject (339개 \[문헌: GitHub 파일\]) |
| `plain` | 지시문이 없는 정상 문단 | 배경 문서의 문단 |

**배경.** LG v2 haystack의 조각을 먼저 쓰고, 없으면 S1에서 받은 스트림을 쓴다. haystack은 LG v2 안전 선별(`screen/keep_<domain>.json`)을 통과한 조각만 쓴다. 선별되지 않은 도메인은 경고를 남긴다.

### 2.2 대비와 probe

| task | 양성 | 학습 음성 | 평가 음성 세트 |
| --- | --- | --- | --- |
| `harm` | harm | plain, harm_hard | all, plain, hard(= harm_hard) |
| `inj` | inj | plain, instr, notinject | all, plain, **instr**, notinject, hard(= instr + notinject) |

probe는 다음과 같다.

- 층(0 = embedding)마다 train 창의 평균·표준편차로 표준화한 L2 logistic regression이다(L-BFGS).
- 표본 가중은 양성 0.5, 음성 클래스들이 나머지 0.5를 똑같이 나눈다.
- λ는 val AUROC로 고른다.
- seed마다 train 그룹의 85 %를 쓴다. seed 수는 quick 3, full 5다.

### 2.3 framing과 pooling

| 이름 | 뜻 |
| --- | --- |
| `raw` framing | 창만 넣는다. LG trunk가 보는 입력과 같다 |
| `task` framing | 정상 과업 16종 중 하나(창 id의 hash로 정한다) 뒤에 창을 문서로 넣은 채팅 입력이다. instruct 타깃은 자기 chat template을 쓴다. guard(Qwen3Guard, Llama Guard)는 template이 moderation prompt이므로 같은 계열의 중립 user 턴(ChatML, Llama 3 형식)을 쓴다 |
| `mean` pooling | 창 토큰 평균. raw에서는 attention sink인 첫 토큰을 뺀다 |
| `last` pooling | raw는 창의 마지막 토큰, task는 prompt의 마지막 토큰이다 |
| `span` pooling | 끼운 조각 토큰의 평균. 진단용이고 기본으로 같이 뽑는다 |

문서 안의 special token과 added token 문자열(`<|im_start|>`, Qwen3의 `<think>`, `<tool_call>` 등)은 평문으로 토큰화한다. backbone 계열에 따라 제어 토큰이 되는 일을 막기 위해서다.

### 2.4 누수 방지, 재현성, 통계

**분할.** 조각은 그룹 단위로, 배경은 **원 문서** 단위로 hash 분할한다(train/val/test = 0.6/0.1/0.3). 한 책의 여러 조각은 같은 분할에 들어간다. 같은 그룹 키는 클래스가 달라도 같은 분할로 간다. LLMail-Inject는 어느 모드에서나 test 전용이다. 다른 injection 원천은 위 비율로 나뉘므로, group 모드에서 게이트의 두 곡선은 모두 학습 분포 안의 값이다. `datacard.json`의 `audit`은 조각 id, 그룹, 조각 원문, 배경 조각, 원 문서가 분할을 넘나들지 않는지 확인한다. `--split-mode source`는 LG 공개 test 원천도 test 전용으로 둔다. 이 모드에서는 held-out 원천이 있는 클래스의 나머지 원천이 train/val에만 들어간다. 그래서 source 모드의 injection test 양성은 LLMail-Inject뿐이다.

**test 전용 원천의 추출.** group 모드에서 LLMail-Inject 창은 다른 injection 창과 따로 뽑는다. 같은 길이 분포를 따르고 개수도 `n_test`로 같다. id는 `test-inj-ho-…`이다. 그래서 게이트의 injection 양성은 LLMail이 있든 없든 `n_test`개를 그대로 갖고, 길이 맞춤도 유지된다. `datacard.json`의 `snippet_tokens`는 게이트 쪽 조각만 세고, LLMail 조각은 `snippet_tokens_heldout`에 따로 적는다. source 모드에서는 LLMail이 게이트의 injection test 양성이므로 따로 뽑지 않고 `snippet_tokens`에 그대로 센다. 어느 분할에 조각이 하나도 없는 클래스는 `datacard.shortfall`에 `got: 0`으로 남는다.

**길이 맞춤.** 모든 클래스를 두 양성 클래스의 길이 히스토그램 교집합으로 층화 추출한다. 한 클래스 안에서는 원천별 가중이 같다. 모자란 구간은 가까운 구간의 아직 쓰지 않은 조각부터 채우고, 그다음에만 재사용한다.

`instr`과 `plain`에는 앞 절반 문장 변형을 더한다. 변형은 클래스 규칙을 다시 만족할 때만 남긴다. 평균 길이가 양성과 20 % 넘게 다르면 경고를 남긴다.

**데이터 정체성.**

- `datacard.json`의 `data_manifest_sha`는 창 내용의 hash다.
- 특징, probe, 점수, lexical baseline, big pool 점수가 모두 이 값을 기록한다.
- 데이터를 다시 만들면 예전 특징은 거부하거나 다시 계산하고, 섞이지 않는다.

**lineage(CLAUDE.md 3.3).** build(`data/`), backbone 실행(`probes/<bb>/`), analyze(`report/`)가 각자 `lineage.json`을 `ag/harness/lineage.py`로 쓴다. `pipeline`과 따로 돌린 `extract`, `fit`, `score`는 같은 backbone 실행의 lineage를 공유한다. 이어 돌리기는 같은 run_id에서만 하고, resume 키(코드, 환경, 설정, 모델, 데이터, GPU)가 다르면 exit 2로 끝난다. 새로 시작하려면 `--fresh`를 주거나 다른 `--run` 이름을 쓴다.

**지표.** 층마다 다음을 잰다.

- AUROC
- TPR@FPR {10⁻², 10⁻³, 2×10⁻⁴, 5×10⁻⁵}. 판정 규칙은 `lgs1.metrics.strict_matched_fpr`과 같다.
- 달성 FPR, resolvable(기대 오탐 5개 이상)

`report/probe_results.jsonl`은 seed별 행이다. 필드는 SPEC 7.3과 같고 `framing`, `pooling`, `negative_set_poc`를 더한다. 모든 행에 run_id가 붙는다.

**bootstrap.** 그룹 단위로 복원 추출한다. 같은 가중을 모든 대비, 층, seed에 쓰므로 ℓ95 비율의 CI가 짝지어진다.

**대조군.**

- lexical baseline: hashed n-gram에 같은 probe를 쓴다.
- shuffled-label control: train·val 라벨과 가중을 모두 섞는다. 0.5 근처여야 한다.

## 3. 실행

### 3.0 준비 (한 번, AG venv)

```bash
cd /home/212aig08/guardrail/main/Efficient-Guardrail
source /home/212aig08/ag_venv/ag311/bin/activate        # D-001
pip install -r requirements.txt matplotlib torch transformers   # 그림이 필요 없으면 matplotlib 생략
cp env/ag.env.example env/ag.env                          # sbatch가 slurm/ag_common.sh로 읽는다
python depthctl.py smoke                                  # 작은 무작위 모델로 전 과정 점검 (CPU, 이 컨테이너에서 26초 [측정])
python -m pytest -q tests/test_agdepth.py                 # 36개
```

smoke는 배관만 확인한다. 보고서 맨 위에도 그렇게 적힌다. 2×10⁻⁴는 smoke 규모로는 잴 수 없어서 판정이 INCONCLUSIVE로 나오는 것이 정상이다.

이 컨테이너(Python 3.11.17, torch 2.14.1 CPU, transformers 5.19.0)에서 잰 값 \[측정\]:

| 명령 | 결과 |
| --- | --- |
| `pytest tests/test_agdepth.py` | 36 passed, 약 102초 |
| `python -m pytest tests -x -q` | 335 passed, 2 skipped |
| `pytest $(cat tests/manifests/cpu.txt)` | 335 passed, 1 skipped |
| `python tools/check_test_classification.py` | rc 0 |

### 3.1 다운로드 (로그인 노드만)

```bash
nohup python depthctl.py download > dl.log 2>&1 &            # BIPIA, NotInject(GitHub raw), deepset, LLMail-Inject, Gandalf, Llama-3.1-8B-Instruct, Qwen3-4B-Instruct-2507
# LG v2 needle pool / haystack가 아직 없으면 위 대신 S1 원천까지 한 번에 받는다(둘을 동시에 돌리지 않는다):
#   nohup python depthctl.py download --with-s1 > dl.log 2>&1 &
python depthctl.py doctor                                    # 다운로드가 끝난 뒤: 무엇이 있고, job이 어디에 쓰는지
```

- xet은 끈다(`HF_HUB_DISABLE_XET=1`). URL 파일은 전부 받았을 때만 저장한다.
- Llama-3.1-8B-Instruct는 gated다(F-10).
- BIPIA와 NotInject는 10/11에 GitHub에서 파일 형식을 확인했다 \[문헌\]. deepset, LLMail-Inject, Gandalf는 이 세션에서 HF Hub에 접근하지 못해 스키마를 확인하지 못했다. loader는 필드 이름을 넉넉하게 받으므로, 받은 뒤 `doctor`와 build 로그의 건수가 0이 아닌지 확인한다.

### 3.2 E0.3 (backbone 2개)

```bash
python depthctl.py run --run e03 --profile quick --backbones e03 --sbatch            # 스크립트 4개 생성 + sbatch --test-only
python depthctl.py run --run e03 --profile quick --backbones e03 --sbatch --submit   # 제출 (afterok로 연결)
```

- 스크립트는 `agctl`과 같은 머리말을 쓴다(`-p RTX6000ADA --qos=normal --gres=gpu:1`, `slurm/ag_common.sh`).
- 출력 루트는 `$AGDEPTH_OUT`, 없으면 `$AG_OUT`(셸 또는 `env/ag.env`)`/poc/agdepth`, 그것도 없으면 `$LG_OUT/agdepth`다. 제출할 때 정한 루트를 job에 `AGDEPTH_OUT`으로 넘긴다. 결과는 `<root>/e03/report/report.md`에 나온다.
- pooling 기본값은 mean, last, span이고 framing 기본값은 raw, task다. `--backbone`은 backbone 하나만 받는다. 여러 개는 `run --backbones`로 준다.
- 단계별 명령(`build`, `pipeline`, `extract`, `fit`, `score`, `analyze`)도 같은 `--sbatch [--submit]`를 받는다.
- extract와 score는 shard 단위로 이어서 돈다.

### 3.3 E1.1 (backbone 4개, seed 5개, 음성 30만)

```bash
python depthctl.py run --run e11 --profile full --backbones e11 --drop-feats --sbatch --submit
```

`--drop-feats`는 fit과 score가 끝나면 특징 memmap을 지운다. 다시 제출하면 probe가 현재 데이터와 맞는 framing은 extract와 fit을 건너뛴다. 원천 일반화는 `--split-mode source`로 다른 `--run` 이름에서 확인한다.

### 3.4 규모 \[추정\]

| | quick (E0.3) | full (E1.1) |
| --- | --- | --- |
| 학습·평가 창 | 5 × 2,400 + NotInject 최대 1,017 ≈ 1.3만 \[계산\] | 5 × 6,000 + NotInject 최대 1,017 ≈ 3.1만 \[계산\] |
| big pool | 클래스 3 × 3만 = 9만 \[계산\] | 클래스 3 × 10만 = 30만 \[계산\] |
| 특징 크기 (8B, framing×pooling 하나) | 33층 × 1.3만 × 4,096 × 2 B ≈ 3.5 GB \[계산\] | ≈ 8.4 GB \[계산\] (`--drop-feats` 권장) |
| GPU 시간 (8B 한 backbone) | 몇 시간 이내 \[추정\] | 수 시간 \[추정\] |

NotInject는 prompt가 339개이고 한 분할에서 한 조각을 최대 3번 쓰므로 train/val/test 창이 대략 600/100/300개에서 멈춘다. 그래서 `datacard.shortfall`에 notinject가 늘 나온다. big pool은 특징을 저장하지 않고 점수만 매긴다. GPU 시간은 RTX 6000 Ada에서 잰 값이 아니다. 첫 실행 로그의 `windows/s`로 바로잡는다.

## 4. 결과 읽기

`report/`에 다음 파일이 생긴다.

| 파일 | 내용 |
| --- | --- |
| `report.md` | 게이트 표(`gate`, `gate_plan`. 주 TPR@2×10⁻⁴, 보조 AUROC), 부가 비교, 대비별 ℓ95와 compute ratio(ℓ95/L), `inj@heldout`(LLMail) 대비, lexical baseline, 대조군, 데이터 표 |
| `probe_results.jsonl` | seed별 층별 값(run_id 포함) |
| `layer_metrics.csv` | seed 평균 곡선과 CI |
| `summary.json` | 기계가 읽는 요약. NaN은 문자열로 쓴다. `e11_backbones`에 backbone별 PASS 수가 있다(E1.1 기준: 2개 이상) |
| `fig_depth_<framing>_<pool>_{tpr,auroc}.png` | 층별 곡선. 점은 ℓ95, 띠는 95 % CI. 계획서 Fig. 3 초안이고 최종본은 `plots/fig03.py`의 몫이다 |
| `fig_l95_gap.png` | ℓ95 상대 깊이 비교 |
| `lineage.json` | 출처 기록 |

PASS를 주장하기 전에 다음을 확인한다.

1. **게이트 두 대비가 resolvable이고 곡선 최고값이 충분한가.** INCONCLUSIVE가 아니어야 한다.
2. **lexical baseline이 이미 최고값에 닿지 않는가.** 닿는 대비는 표면 특징으로 풀리는 것이라 ℓ95가 깊이에 대해 말해 주지 않는다.
3. **`span` pooling에서도 방향이 같은가.** 길이 경고가 있으면 이것을 먼저 본다. span은 기본으로 같이 뽑는다. 나중에 따로 더하면 해당 framing의 특징을 처음부터 다시 뽑는다.
4. **framing 두 개에서 같은 방향인가.**
5. **shuffled-label control이 0.5 근처인가.**
6. **`--split-mode source`에서도 유지되는가.**

E0.3이 미달이어도 진행을 멈출 사유는 아니다(계획서 §2.2). 그 경우 설계 1은 비용 절감으로만 쓴다.

## 5. 한계와 주의

- **injection 양성은 공개 데이터만 쓴다.** 직접 만든 공격 문구는 넣지 않았다. 계획서의 공격 6종 구성과 다르다(D-041 (b)). 원천별 창 수는 datacard의 `windows_by_source`에 있다. S2 payload나 Open-Prompt-Injection·AgentDojo의 공식 생성기 출력을 더하려면 jsonl(`text` 필드)로 `$LG_SOURCES/<이름>/`에 두고 `agdepth/config.py`의 `SNIP_SOURCES`에 한 줄 추가한다(`loader="load_text_label"`).
- **`instr` 템플릿은 40종이다.** test에는 학습하지 않은 템플릿만 들어가지만 다양성은 실제 문서보다 낮다. 지시문 밀도 등급과 "과업이 문맥 지시를 따르라는" 정상 등급은 아직 없다(D-041 (f)).
- **injection 여부는 창 혼자로 정해지지 않는다**(계획서 관찰 2). 이 실험은 창 단위 probe로 깊이만 본다.
- **TPR@FPR은 test 음성의 ROC에서 읽는다.** 층별 분리도를 재는 실험이라 운영점을 정하지 않는다(D-041 (i)).
- 마지막 층의 hidden state는 transformers가 final norm을 적용한 값이다. 모든 클래스가 같은 처리를 받는다.
