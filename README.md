# Qwen3 Teacher Scaling for Cartridges

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3ae3ae/qwen3-teacher-scaling/blob/v0.2.0/notebooks/qwen3_teacher_scaling.ipynb)

공개 Cartridges Qwen3-4B 대화를 사용해 scoring teacher 크기를 비교한다. Student는 Qwen3-4B, teacher는 A: Qwen3-4B와 B: Qwen3-8B다. 학습·평가는 공개 LongHealth·MTOB 예제를 따른다.

- [실험 설계](EXPERIMENT_PLAN.md)
- [재현 정보·데이터 출처](docs/REPRODUCIBILITY.md)
- [Colab notebook](notebooks/qwen3_teacher_scaling.ipynb)

## 실행

1. **Open in Colab**으로 노트북을 연다.
2. 런타임 버전 `2026.07`(Python 3.12)과 BF16 GPU를 선택한다.
3. `CONDITION=A/B/all`, `BENCHMARK`, `PROFILE`을 선택하고 셀을 순서대로 실행한다.

노트북은 공개 데이터 다운로드 → teacher 재채점 → 카트리지 학습·평가를 수행한다. 기본 `smoke`는 32대화·최대 2 steps, `pilot`은 512대화·최대 16 steps, `main`은 전체 131,072대화와 seed 42·123·2026을 사용한다.

A·B를 따로 실행할 때 같은 `RUN_NAME`·benchmark·profile을 유지한다. 공개 대화, 초기 cache와 완료된 단계를 재사용한다. `MODE=evaluate`는 저장된 최종 checkpoint와 평가 RNG 상태로 재평가한다. 결과 표는 조건별 점수와 같은 seed의 B−A 차이를 보여준다.

`USE_DRIVE=True`이면 실행 결과를 Google Drive에 저장하고 새 런타임에서 복원한다. 중단된 학습을 재시도할 때는 새 `RUN_NAME`을 사용한다.

공개 데이터 버전의 GPU 실행 검증은 아직 남아 있다. 설치 셀에서 설정·데이터 연결 검사를 수행한다.
