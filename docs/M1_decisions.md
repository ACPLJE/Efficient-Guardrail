# M1 · 교수님 확인 사항 (S1, 9/19까지)

각 항목에 결정과 날짜를 적고, 결정된 항목은 `[DECIDED]`로 표시합니다. `python s1ctl.py status`가 이 파일에서 `DECIDED`를 찾습니다.

| # | 결정 사항 | 선택지 | 제안 기본값 | 영향 | 결정 |
| --- | --- | --- | --- | --- | --- |
| 1 | 백본 범위 | (a) 0.6B만 (b) 0.6B/4B/8B (c) 0.6B/4B/8B + Llama Guard 3-8B | (c) — 계열 교차(LG3-8B)가 있어야 "Qwen 전용 트릭" 리뷰를 막음 | T3–T5 학습 +58 GPU-h, E2-8·E1 확장 +11 GPU-h; 없으면 §11 축소 6 | [ ] |
| 2 | E5 서빙 범위 | (a) E5-1·E5-5만 (b) + E5-2·E5-4 (c) 전체(E5-3·E5-6 포함) | (b) — MLSys 헤드라인은 E4로 성립, E5-2·4는 Fig 11 완성용 | (c)는 S11 GPU C 12 h + 엔지니어링 2일 추가 | [ ] |
| 3 | 논문 제목 | (a) LIFEGUARD 약어 사용 (b) 약어 없이 "Landmark Guard: …" (c) 다른 약어 | (b) 기본, 약어 풀이가 확정되면 (a) | 익명화·artifact 이름·Fig 2 라벨 | [ ] |
| 4 | CD sweep 범위 | (a) 기본 config(Qwen3Guard-8B, W=512, overlap W/2)만 전량, 나머지 부분집합 (b) 전량 sweep | (a) — 전량 sweep은 ≈300 GPU-h 추가(계획서 §3 계산) | Table 2 CD 행 수, Fig 6 CD 선 4개의 신뢰구간 | [ ] |

## 결정 기록

- 날짜:
- 참석:
- 1 백본 범위:
- 2 E5 범위:
- 3 제목:
- 4 CD sweep:

(모두 결정되면 아래 줄의 대괄호 안을 DECIDED로 바꾸세요)

상태: [PENDING]
