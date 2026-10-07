# Qwen3 Teacher Scaling for Cartridges

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3ae3ae/qwen3-teacher-scaling/blob/v0.3.0/notebooks/qwen3_teacher_scaling.ipynb)

공개 Cartridges Qwen3-4B 대화를 사용해 scoring teacher 크기를 비교한다. Student는 Qwen3-4B, teacher는 A: Qwen3-4B와 B: Qwen3-8B다. LongHealth에서는 전체 환자 기록을 프롬프트에 넣는 ICL 기준선도 비교한다. 학습·평가는 공개 LongHealth·MTOB 예제를 따른다.

- [실험 설계](EXPERIMENT_PLAN.md)
- [재현 정보·데이터 출처](docs/REPRODUCIBILITY.md)
- [Colab notebook](notebooks/qwen3_teacher_scaling.ipynb)

## 실행

1. **Open in Colab**으로 노트북을 연다.
2. 런타임 버전 `2026.07`(Python 3.12)과 BF16 GPU를 선택한다.
3. `CONDITION=A/B/ICL/all`, `BENCHMARK`, `PROFILE`을 선택하고 셀을 순서대로 실행한다.

A·B는 공개 데이터 다운로드 → teacher 재채점 → 카트리지 학습·평가를 수행한다. 기본 `smoke`는 32대화·최대 2 steps, `pilot`은 512대화·최대 16 steps, `main`은 전체 131,072대화와 seed 42·123·2026을 사용한다.

`ICL`은 LongHealth 전체 기록을 준비해 Qwen3-4B로 평가한다. `smoke`는 2문항, `pilot`·`main`은 200문항이다. Qwen의 공식 YaRN 설정으로 문맥을 확장하며 A100 40GB 이상의 메모리를 권장한다. `all`은 LongHealth A·B·ICL, MTOB A·B를 실행한다.

조건을 따로 실행할 때 같은 `RUN_NAME`·benchmark·profile을 유지한다. 공개 대화, 초기 cache와 완료된 단계를 재사용한다. `MODE=evaluate`는 A·B의 최종 checkpoint와 평가 RNG 상태로 재평가하며, ICL은 같은 원문과 seed로 다시 평가한다. 결과 표는 조건별 점수와 같은 seed의 B−A·A−ICL·B−ICL 차이를 보여준다.

`USE_DRIVE=True`이면 실행 결과를 Google Drive에 저장하고 새 런타임에서 복원한다. 중단된 학습을 재시도할 때는 새 `RUN_NAME`을 사용한다.

공개 데이터와 ICL 버전의 GPU 실행 검증은 아직 남아 있다. 설치 셀에서 설정·데이터 연결 검사를 수행한다.
