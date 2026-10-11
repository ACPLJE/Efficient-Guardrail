# tests/data

CPU 테스트가 읽는 작은 입력 파일을 둔다. 행 파일은 `*.jsonl` ignore 규칙(CLAUDE.md 3.2)에 걸리지 않도록 `.jsonl.txt` 접미사를 쓴다.

## Mooncake trace 형식 fixture (W0-mooncake, SPEC 11)

| 파일 | 행 수 | 내용 |
| --- | --- | --- |
| `mooncake_conversation_50.jsonl.txt` | 50 | conversation trace 흉내. 앞 행의 전체 블록 prefix를 이어 받는 multi-turn 행, 입력 32,768토큰 이상 행 5개(D-020 128k 등급 1개) |
| `mooncake_toolagent_50.jsonl.txt` | 50 | toolagent trace 흉내. 대부분의 행이 3블록짜리 공통 도구 prefix를 공유, 출력 3토큰 이하 행이 많음, 입력 32,768토큰 이상 행 3개 |

**이 두 파일은 실제 Mooncake trace가 아니다.** FAST25-release README에 적힌 형식 \[문헌\](행마다 `timestamp`(trace 시작부터의 ms), `input_length`, `output_length`, `hash_ids`)을 따르는 **합성 행**이다. 이 fixture를 만든 환경에서는 github.com에 접근할 수 없어 실제 trace를 받거나 잘라 올 수 없었다. 값의 분포는 실제 trace의 분포와 관계가 없으므로 이 파일에서 나온 수치를 논문이나 보고에 쓰지 않는다. 실제 trace의 집계는 로그인 노드에서 `bash slurm/ag_download_login.sh --only mooncake`로 받은 뒤 `scripts/mooncake_stats.py`로 낸다.

만든 규칙:

- `timestamp`는 0에서 시작해 엄격히 증가한다.
- `hash_ids`는 512토큰 블록 하나에 id 하나(A-12)라서 길이가 `ceil(input_length / 512)`이다. 단, 파일마다 **정확히 2개 행**은 일부러 어긋나게 했다(하나는 id 하나 많음, 하나는 하나 적음). 블록 단위 검사(`hash_ids_block_check`)가 불일치를 세는지 보는 용도다.
- prefix 공유는 앞 행의 전체 블록(`input_length // 512`개) id를 그대로 앞에 두는 방식으로 만들었다.
- 난수는 Python `random.Random(seed).random()`만 쓴다(Python 버전이 바뀌어도 같은 수열이 보장되는 유일한 메서드).

다시 만드는 명령(생성기는 `tests/test_ag_trace_loader.py`의 `synth_rows()`이고, 같은 파일의 `test_fixture_matches_generator`가 커밋된 파일과 생성기 출력이 바이트 단위로 같은지 확인한다):

```bash
PYTHONPATH=. python tests/test_ag_trace_loader.py --write
```
