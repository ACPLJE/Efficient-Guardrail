# engine_integration — 엔진 변경을 모으는 곳 (S1E·S3E에서 채운다)

엔진(SGLang 0.5.21, 2차 vLLM) 수정은 이 디렉터리의 patch 파일과 shim에만 둔다. 논문에 엔진 변경 LoC를 따로 적기 위해서다(CLAUDE.md 3.7, 구축 계획 5절).

- `sglang/ag-sglang-0.5.21.patch`: Triton decode 분할 고정 + `attn_lse` 반출. hunk마다 `# AG:` 표시.
- `sglang/{patch.sh, check_patch.sh, remove_patch.sh}`: dry-run 멱등.
- `sglang/shim/`: ledger·gate·scheduler hook. 적용 시 한 줄 로그를 남기고, 하네스가 그 줄로 `shim_active`를 확인한다(B0에서는 없어야 한다).

S1 시점에는 비어 있다. 하네스는 mock backend로만 돈다.
