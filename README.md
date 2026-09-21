# Qwen3 Teacher Scaling for Cartridges

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3ae3ae/qwen3-teacher-scaling/blob/v0.1.0/notebooks/qwen3_teacher_scaling.ipynb)

공개 Cartridges Qwen 예제를 기준으로 생성 모델과 scoring teacher 규모를 비교한다. Student는 Qwen3-4B이며, 평가는 LongHealth·MTOB를 사용한다.

- [실험 설계](EXPERIMENT_PLAN.md)
- [Colab notebook](notebooks/qwen3_teacher_scaling.ipynb)
- [재현 정보·변경 범위](docs/REPRODUCIBILITY.md)

## Colab 실행

1. 위의 **Open in Colab** 버튼으로 노트북을 연다.
2. Python 3.12 GPU 런타임을 선택한다.
3. 상단에서 `MODE`, `BENCHMARK`, `PROFILE`을 선택하고 셀을 순서대로 실행한다. 소스 준비 셀은 고정된 commit의 실험 소스를 자동으로 내려받는다.

기본 profile은 smoke다. Main은 공개 학습 예제의 전체 데이터 규모와 seed 42·123·2026을 사용한다. 필요한 Python 패키지와 원본 소스는 Colab 안에서 설치된다. Colab의 기본 OS를 사용한다.

노트북은 준비 → 합성·scoring → 학습·평가 순서로 실행한다. 결과와 checkpoint를 Google Drive에 저장할 수 있다. 완료된 단계는 재사용하며, 중단된 학습의 재시도에는 새 `RUN_NAME`을 사용한다.

`MODE=evaluate`는 기존 `RUN_NAME`·benchmark·profile·seed의 최종 checkpoint와 평가 RNG 상태를 불러와 재평가한다. 결과 표에는 조건·seed별 점수, 조건별 평균·표준편차, 동일 seed의 B−A 차이가 표시된다.

평가 데이터는 실행 시 공식 배포처에서 받는다. 데이터별 조건은 [평가 자료와 라이선스](docs/REPRODUCIBILITY.md#평가-자료와-라이선스)에 정리되어 있다.

검증: 코드·notebook 정적 검사 완료. 설정 일치·연결 검사는 Colab 설치 셀에서 실행한다. Colab GPU 전체 실행은 검증 전이다.
