# A pilot: LongHealth

2026-09-21, [v0.1.4](https://github.com/3ae3ae/qwen3-teacher-scaling/releases/tag/v0.1.4), seed 42. 생성 모델·scoring teacher·student는 모두 Qwen3-4B다.

## 결과

| 평가 | 정답 수 | 정확도 | 답변 태그 누락 |
|---|---:|---:|---:|
| 학습 전 카트리지 | 34/200 | 17.0% | 155 |
| A pilot 학습 후 | 44/200 | 22.0% | 124 |
| 비교 논문 Qwen3-4B | 71/200 | 35.5% | — |

학습 전후 차이는 **+5.0%p**, 논문 대비 차이는 **−13.5%p**다. 논문 수치는 [arXiv:2508.17032의 Table 1](https://arxiv.org/html/2508.17032)에 제시된 학습된 카트리지의 결과다.

태그가 누락된 124문항은 공개 구현의 규칙에 따라 첫 선택지로 채점했다. 최종 정답 44개 중 30개는 태그가 있는 응답, 14개는 이 기본 처리에서 나왔다. 평가 문항 ID는 200개 모두 고유하며 학습 전후의 문항 집합이 일치한다.

## 실행 조건

| 항목 | A pilot | 비교 논문 |
|---|---|---|
| Cartridge 길이 | 2,048 | 2,048 |
| 초기화 | 기본 gradient 문서 | 기본 gradient 문서 |
| Optimizer updates | 10 | 512 |
| Global batch | 32 | 128 |
| Packed length | 2,048 | 1,024 |

512개 자체 합성 대화와 2 epochs를 사용했다. 공개 pilot 설정의 16-step 상한 안에서 2 epochs가 끝나 실제 update는 10회였다. Epoch당 packed batch는 168개다. 원문 발췌와 seed prompt가 일치하는 기존 생성 결과의 첫 16개 batch를 재사용했으며, 파일 해시를 보존했다.

대화 생성과 학습·채점은 고정한 공개 구현을 사용한다. Soft targets는 모든 비교 조건에 공통인 Hugging Face teacher-forcing 경로로 계산한다. 전체 변경 범위는 [v0.1.4 재현 정보](https://github.com/3ae3ae/qwen3-teacher-scaling/blob/v0.1.4/docs/REPRODUCIBILITY.md#변경-범위)에 정리되어 있다.

실행 환경은 Colab 2026.07, A100 40GB, Python 3.12.13, PyTorch 2.6.0+cu124, Transformers 4.53.0이다. 학습 프로세스 시간은 초기·최종 평가를 포함해 약 33분 26초였고, 최대 CUDA 할당량은 21.24 GiB였다. 첫·마지막으로 기록된 학습 loss는 각각 0.6993과 0.5824다.

## 해석과 재현 정보

합성·scoring·학습·200문항 평가·checkpoint 저장까지 A pilot 실행을 완료했다. 정확도는 5.0%p 상승했으며, 논문과의 격차는 13.5%p다. 학습량 차이와 높은 태그 누락률을 고려할 때 이 결과의 용도는 실행 검증과 축소 학습의 예비 성능 확인이다. 논문 수준의 성능 재현 여부는 추가 검증이 필요하다.

- 실험 소스 commit: `4cb6fe9e54cd5cd324edf7a36de61b5c43935340`
- 실행 설정: `CONDITION=A`, `MODE=train`, `BENCHMARK=longhealth`, `PROFILE=pilot`, `SEEDS=42`
- [집계 결과·모델/데이터 revision·파일 해시](A_PILOT_RESULT.json)

초기·최종 checkpoint, 평가 RNG 상태, 합성 대화·soft targets, 문항별 prediction과 실행 로그를 보존하고 checkpoint 해시를 검증했다.
