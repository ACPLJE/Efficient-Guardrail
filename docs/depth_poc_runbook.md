# 깊이 비대칭 PoC 실행 안내 (E0.3 / E1.1)

Oct 11, 2026 · 계획서 `asymmetric-guard-osdi27-plan-2026-10-10.md` §2.2의 관찰 1(깊이), §8.2의 E0.3과 §8.3의 E1.1을 돌리는 코드다. 코드는 `agdepth/` 패키지, 진입점은 `depthctl.py`다.

## 1. 무엇을 보이는가

**주장.** 같은 backbone, 같은 선형 probe로 판정하면 유해 콘텐츠 창은 얕은 층에서 정상 창과 갈린다. 반면 injection 창은 지시문이 많은 정상 창과 깊은 층에서야 갈린다.

**게이트(계획서 E0.3, 제안값).** ℓ95(injection 대 지시문이 많은 음성) ≥ 1.5 × ℓ95(유해 대 모든 정상).

- ℓ95는 층별 곡선이 최고값의 95 %에 처음 닿는 층이다.
- 주 지표는 낮은 창 FPR에서의 TPR이다. 계획서 기준은 2×10⁻⁴이다.
- 음성 수가 모자라면 기대 오탐이 5개 이상인 가장 작은 FPR을 자동으로 고르고, 보고서에 그 사실을 적는다.

결과는 판정 3종으로 나온다.

| 판정 | 뜻 |
| --- | --- |
| PASS | 비율이 1.5 이상 |
| FAIL | 비율이 1.5 미만 |
| INCONCLUSIVE | 곡선의 최고값이 너무 낮다(TPR < 0.2, AUROC < 0.7). 또는 그 FPR을 재기에 음성이 모자라다 |

## 2. 실험 설계

### 2.1 창

창은 배경 문서의 연속 구간 사이에 조각 하나를 독립된 문단으로 끼운 것이다. 라벨은 끼운 조각의 종류다.

- 길이는 기본 256토큰(Qwen3 tokenizer 기준)이다.
- 조각 길이는 12–112토큰이다. LG v2의 needle 상한 128을 따랐다.
- 조각을 넣는 위치는 창 안의 상대 깊이 U(0.05, 0.95)이고, 가까운 줄 경계나 문장 경계에 맞춘다.
- 모든 클래스를 같은 절차로 만든다. 그래서 클래스 사이에 체계적으로 다른 것은 조각뿐이다.

| 클래스 | 내용 | 출처 |
| --- | --- | --- |
| `harm` | 유해 요청 | LG v2 needle pool(`needles/pool.jsonl.gz`, label harmful). 없으면 S2 loader로 원천 파일을 메모리에서 읽는다 |
| `harm_hard` | 정상 요청(OR-Bench 등 과잉 거부 유발 문장 포함) | LG v2 needle pool, label benign |
| `inj` | prompt injection payload | **공개 벤치마크만 쓴다**: BIPIA 공격 지시문(text/code), deepset/prompt-injections(label 1), LLMail-Inject 제출물, Gandalf |
| `instr` | 사람에게 하는 지시문이 많은 정상 텍스트 | 내장 템플릿 40종(README, 메일, 레시피, 절차, 정책, 고객지원, 설명서, 시험, 여행, trigger 단어의 정상 용례) + 배경 문서에서 명령문이 2개 이상인 문단 |
| `notinject` | trigger 단어가 든 정상 prompt 339개 | NotInject |
| `plain` | 지시문이 없는 정상 문단 | 배경 문서의 문단 |

### 2.2 대비와 probe

| task | 양성 | 학습 음성 | 평가 음성 세트 |
| --- | --- | --- | --- |
| `harm` | harm | plain, harm_hard | all, plain, hard(=harm_hard) |
| `inj` | inj | plain, instr, notinject | all, plain, **instr**, notinject, hard(=instr+notinject) |

probe는 다음과 같다.

- 층마다(0 = embedding) train 창 전체의 평균·표준편차로 표준화한 L2 logistic regression이다.
- 학습은 full-batch L-BFGS로 한다.
- 표본 가중은 양성 0.5, 음성 클래스들이 나머지 0.5를 똑같이 나눈다. 쉬운 `plain`이 어려운 음성을 덮지 못하게 하기 위해서다.
- λ는 val AUROC로 고른다.
- seed마다 train 그룹의 85 %를 쓴다. 기본 seed 수는 quick 3개, full 5개다.

### 2.3 framing과 pooling

| 이름 | 뜻 |
| --- | --- |
| `raw` framing | 창만 넣고 special token은 넣지 않는다. LG trunk가 보는 입력과 같다 |
| `task` framing | 창을 backbone 자신의 chat template 안에 문서로 넣고, 정상 과업 16종 중 하나를 앞에 붙인다. 과업은 창 id의 hash로 정하므로 클래스와 무관하다. 서빙 중인 타깃이 보는 입력이다 |
| `mean` pooling | 창 토큰 평균. raw에서는 첫 토큰(attention sink)을 뺀다 |
| `last` pooling | raw는 창의 마지막 토큰, task는 prompt의 마지막 토큰(첫 응답 토큰을 내는 위치) |
| `span` pooling | 끼운 조각 토큰의 평균. 배경 토큰에 신호가 희석되는 효과를 빼고 보는 진단용이다 |

문서 안의 special-token 문자열(`<|im_start|>` 등)은 split_special_tokens=True로 평문으로 다룬다.

### 2.4 누수 방지와 통계

**분할.** 조각은 그룹 단위로, 배경은 문서 단위로 train/val/test에 hash 분할한다(0.6/0.1/0.3).

그룹의 정의는 원천마다 다르다.

| 원천 | 그룹 |
| --- | --- |
| LG pool | near-dup cluster |
| BIPIA | 공격 범주 |
| 내장 템플릿 | 템플릿 |
| 배경에서 캔 문단 | 원래 문서 |
| 나머지 | MinHash near-dup |

같은 그룹 키는 클래스가 달라도 같은 분할로 간다. `datacard.json`의 `audit`은 조각 id, 그룹, 조각 원문, 배경 문서가 분할을 넘나들지 않는지 확인한다.

**`--split-mode source`.** holdout 원천을 test 전용으로 둔다. holdout은 LG의 공개 test 원천과 LLMail-Inject이고, holdout 행이 하나라도 든 그룹은 통째로 test로 간다.

**길이 맞춤.** 모든 클래스를 두 양성 클래스의 길이 히스토그램 교집합으로 층화 추출한다. 한 클래스 안에서는 원천별 가중이 같다. 정상 `instr`과 `plain`은 앞 절반 문장만 남긴 변형(같은 그룹)을 더해 짧은 길이 구간을 채운다. 클래스 평균이 양성 평균과 20 % 넘게 다르면 경고를 남긴다.

**지표.** 층마다 다음을 잰다.

- AUROC
- TPR@FPR {10⁻², 10⁻³, 2×10⁻⁴, 5×10⁻⁵}. 판정 규칙은 `lgs1.metrics.strict_matched_fpr`과 같다(score > τ).
- 달성 FPR, resolvable 여부(기대 오탐 5개 이상)

ℓ95는 seed 평균 곡선에서 읽는다. AUROC의 ℓ95는 2·AUC−1로 바꿔 읽는다. 0.5라는 바닥 때문에 "최고값의 95 %"가 너무 쉽게 닿기 때문이다.

**bootstrap.** test(와 big pool)의 그룹을 복원 추출한다. 같은 가중을 모든 대비, 층, seed에 함께 쓰므로 두 대비의 ℓ95 비율에 짝지은 CI가 나온다.

**대조군.**

- **lexical baseline:** hashed unigram+bigram에 같은 probe를 쓴다. 모델 계산이 전혀 없는 특징이다.
- **shuffled-label control:** train과 val의 라벨을 모두 섞고 가중도 섞은 라벨로 다시 맞춘 뒤 test AUROC를 잰다. 0.5 근처여야 한다.

**낮은 FPR 영역.** 2×10⁻⁴와 5×10⁻⁵를 재려면 음성 창이 수만에서 수십만 개 필요하다. full profile은 test 조각과 test 배경으로 big pool을 만든다(클래스 3종 × 10만). 특징은 저장하지 않고, 학습이 끝난 probe로 바로 점수만 매긴다(`score`).

## 3. 실행

### 3.0 준비 (한 번)

```bash
cd /home/212aig08/<이 저장소>
$LG_PYTHON -m pip install -r requirements.txt matplotlib    # 그림이 필요 없으면 matplotlib 생략
python depthctl.py smoke                                   # CPU, 1분 이내: 작은 무작위 모델로 전 과정 점검
python -m pytest -q tests/test_agdepth.py                  # 15개, CPU 약 40초
```

smoke는 배관만 확인한다. placeholder 클래스와 무작위 모델로 돌리므로 결과 수치에는 뜻이 없다. 보고서 맨 위에도 그렇게 적힌다.

### 3.1 다운로드 (로그인 노드만)

```bash
python depthctl.py download              # BIPIA, NotInject(GitHub raw), deepset, LLMail-Inject, Gandalf, Llama-3.1-8B-Instruct, Qwen3-4B-Instruct-2507
python depthctl.py download --with-s1    # LG v2 needle pool / haystack를 아직 만들지 않았으면 S1 원천도 받는다
python depthctl.py doctor                # 무엇이 있고 무엇이 빠졌는지
```

- Llama-3.1-8B-Instruct는 gated다. HF 페이지에서 라이선스를 수락하고 `HF_TOKEN`을 설정한다.
- BIPIA와 NotInject의 파일 형식은 10월 11일에 GitHub에서 확인했다.
- deepset, LLMail-Inject, Gandalf는 HF Hub에서 받는다. 이 세션에서는 Hub에 접근할 수 없어 스키마를 확인하지 못했다(`verify=True`). loader는 필드 이름을 넉넉하게 받도록 짰다(`text|prompt`, `subject+body`, `label`). 받은 뒤 `doctor`와 build 로그에서 건수가 0이 아닌지 확인한다.

### 3.2 E0.3 (계획서 W0, backbone 2개)

```bash
python depthctl.py run --run e03 --profile quick --backbones e03 --sbatch
#   = build(CPU 잡) -> qwen3guard_0.6b, llama31_8b 각각 extract+fit(GPU 잡) -> analyze, afterok로 연결
#   --dry-run 을 붙이면 sbatch 스크립트만 쓰고 제출하지 않는다
```

결과는 `$AG_OUT/e03/report/report.md`에 나온다. `$AG_OUT`의 기본값은 `$LG_OUT/agdepth`다.

단계별로 나눠 돌려도 된다. 각 명령에 `--sbatch`를 붙일 수 있고, extract와 score는 shard 단위로 이어서 돈다.

```bash
python depthctl.py build    --run e03 --profile quick [--split-mode source] [--window 256 --snip-tok 12,112]
python depthctl.py pipeline --run e03 --backbone llama31_8b --framings raw,task --poolings mean,last
python depthctl.py analyze  --run e03 [--primary-fpr 1e-3] [--n-boot 1000]
```

### 3.3 E1.1 (backbone 4개, 낮은 FPR)

```bash
python depthctl.py run --run e11 --profile full --backbones e11 --drop-feats --sbatch
```

- full profile은 클래스당 train 4,000, val 500, test 1,500이고 big pool이 클래스 3종 × 10만이다. seed는 5개다.
- `--drop-feats`는 fit과 score가 끝나면 특징 memmap을 지운다. probe와 점수는 남는다.
- 일반화를 확인하려면 `--split-mode source`로 한 번 더 돌린다. LG 공개 test 원천과 LLMail-Inject가 test 전용이 된다.

### 3.4 규모 \[추정\]

| | quick (E0.3) | full (E1.1) |
| --- | --- | --- |
| 창 수 | 클래스 6 × 2,400 ≈ 1.4만 | 클래스 6 × 6,000 ≈ 3.6만 + big pool 30만 |
| 특징 크기, 8B, framing×pooling 하나 | 33층 × 1.4만 × 4096 × 2 B ≈ 3.9 GB | ≈ 9.7 GB (`--drop-feats` 권장) |
| GPU 시간, backbone 하나 | 8B 기준 1시간 미만 | 8B 기준 big pool 포함 수 시간 |

수치는 RTX 6000 Ada에서 잰 값이 아니다. 첫 실행 로그의 `windows/s`로 바로잡는다.

## 4. 결과 읽기

`report/`에 다음 파일이 생긴다.

| 파일 | 내용 |
| --- | --- |
| `report.md` | 게이트 표(지표 2종: 주 TPR과 AUROC), 부가 비교, 대비별 ℓ95와 최고값, lexical baseline, 대조군, 데이터 표 |
| `layer_metrics.csv` | backbone × framing × pooling × 대비 × 층의 모든 값 |
| `summary.json` | 기계가 읽는 요약 |
| `fig_depth_<framing>_<pool>_tpr.png`, `_auroc.png` | 층별 곡선(계획서 Fig. 3 초안). 점은 ℓ95, 띠는 95 % CI |
| `fig_l95_gap.png` | backbone마다 harm과 inj/instr의 ℓ95 상대 깊이와 CI |

PASS를 주장하기 전에 다음을 확인한다.

1. **곡선이 제대로 섰는가.** 게이트 두 대비의 최고값이 충분히 높고(INCONCLUSIVE가 아님), `layer_metrics.csv`의 `resolvable@<FPR>`가 True인지 본다.
2. **lexical baseline이 이미 최고값에 닿지 않는가.** 닿는 대비라면 표면 특징으로 풀리는 것이어서 그 ℓ95는 깊이에 대해 말해 주지 않는다. 특히 Gandalf류의 "ignore previous instructions" 문형이 그럴 수 있다.
3. **`span` pooling에서도 방향이 같은가.** 조각 길이 경고가 있으면 이것을 먼저 본다.
4. **framing 두 개에서 같은 방향인가.** raw는 LG가 쓰는 입력이고 task는 타깃이 쓰는 입력이다. 둘 다에서 차이가 나야 주장이 강해진다.
5. **shuffled-label control이 0.5 근처인가.**
6. **`--split-mode source`에서도 유지되는가.** 원천 일반화를 본다.

E0.3이 미달이어도 진행을 멈출 사유는 아니다(계획서 §2.2). 그 경우 설계 1은 유해 경로의 비용 절감으로만 쓴다.

## 5. 한계와 주의

- **injection 양성은 공개 데이터만 쓴다.** 직접 만든 공격 문구는 넣지 않았다. 계획서의 공격 6종 중 fake completion, combined, tool-knowledge가 얼마나 들어가는지는 원천 구성에 달려 있으므로, 원천별 창 수를 datacard의 `windows_by_source`에서 확인한다. 특정 공격 유형을 늘리려면 Open-Prompt-Injection이나 AgentDojo의 공식 생성기로 만든 payload를 jsonl(`text` 필드)로 `$LG_SOURCES/<이름>/`에 두고, `agdepth/config.py`의 `SNIP_SOURCES`에 한 줄 추가한다(`loader="load_text_label"`).
- **`instr` 템플릿은 40종이다.** test에는 학습하지 않은 템플릿만 들어가지만 다양성은 실제 문서보다 낮다. 배경에서 캔 문단(`mined_instr`)의 비율을 datacard에서 확인한다. stack(README), govreport, agentdojo 배경이 있으면 늘어난다.
- **injection 여부는 창 혼자로 정해지지 않는다**(계획서 관찰 2). 이 실험은 창 단위 probe로 깊이만 본다. 과업과 출력까지 보는 판정은 E0.1–E0.2에서 다룬다.
- **창과 조각 길이는 기준 tokenizer로 맞춘다.** 다른 backbone에서는 같은 텍스트가 토큰 수 기준으로 ±10 % 정도 다르다. `extract`의 `aux.npz`에 backbone별 토큰 수가 남는다.
- 마지막 층의 hidden state는 transformers가 final norm을 적용한 값이다. 모든 클래스가 같은 처리를 받는다.
