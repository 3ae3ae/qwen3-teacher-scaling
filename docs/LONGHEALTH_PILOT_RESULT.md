# LongHealth A·B·ICL pilot 결과

2026-10-08, [v0.4.3](https://github.com/3ae3ae/qwen3-teacher-scaling/releases/tag/v0.4.3), protocol 2.2, seed 42. Student는 Qwen3-4B이며, A의 scoring teacher는 Qwen3-4B, B는 Qwen3-8B다. ICL은 전체 환자 기록을 프롬프트로 제공한 Qwen3-4B 평가다.

## 최종 결과

| 조건 | 정답 수 | 정확도 | 답변 태그 누락 | 태그 응답 정답 | 기본 처리 정답 |
|---|---:|---:|---:|---:|---:|
| A | 78/200 | 39.0% | 100 | 64 | 14 |
| B | 73/200 | 36.5% | 117 | 60 | 13 |
| ICL | 71/200 | 35.5% | 138 | 56 | 15 |

B−A는 **−2.5%p**, A−ICL은 **+3.5%p**, B−ICL은 **+1.0%p**다. A·B의 학습 전 카트리지 점수는 모두 35/200(17.5%)다. 최종 학습 후 상승은 A +21.5%p, B +19.0%p다.

원본 채점 함수는 `<answer>...</answer>`가 있으면 문자열 유사도로 가장 가까운 선택지를 고르고, 태그가 없으면 첫 선택지를 사용한다. 이 기본 처리로 얻은 정답 수를 표에 별도로 표시했다. 600개 최종 응답을 원본 채점 함수로 대조했으며 저장된 점수·추출 답변과 모두 일치했다. 세 조건의 문항 ID는 각각 200개 모두 고유하고, 문항과 정답이 일치한다.

이번 실행에서 8B teacher의 최종 정확도는 4B teacher보다 낮았다. A만 맞힌 문항은 25개, B만 맞힌 문항은 20개다. 시드 1개와 높은 답변 태그 누락률을 고려하여 teacher 규모의 효과를 해석해야 한다.

## 실행 조건

공개 Qwen3-4B 대화의 고정 파일·행 순서에서 첫 32,768개를 선택했다. A·B는 동일한 대화와 초기 카트리지를 사용하고, 각 teacher의 soft targets로 2 epochs를 학습했다. Epoch당 packed batch는 11,303개이며 실제 optimizer update는 조건별 706회다.

| 항목 | 설정 |
|---|---|
| Cartridge 길이 / packed length | 2,048 / 2,048 tokens |
| Global batch / optimizer / learning rate | 32 / Adam / 0.02 |
| 평가 | 첫 10명 환자, 200문항, 학습 전·128 updates마다·최종 |
| 생성 | Thinking 활성화, temperature 0.3, 출력 상한 512 tokens |
| 평가 batch | A·B 32, ICL 1 |
| ICL 입력 | 원문 113,634 tokens, 최대 전체 prompt 114,219 tokens |
| ICL 문맥 확장 | YaRN factor 4, 기본 길이 32,768, 한도 131,072 tokens |
| 실행 환경 | Colab A100 40GB, Python 3.12.13, PyTorch 2.6.0+cu124, Transformers 4.53.0 |

ICL은 전체 기록을 보존했으며 입력 잘림은 없다. B의 학습 config가 8B 재채점 파일을 사용하는 것을 확인했다. A·B의 최종 checkpoint SHA-256은 각 실행 summary와 일치했다.

## 학습 중 평가

| Optimizer update | A 정확도 | B 정확도 |
|---|---:|---:|
| 0 | 17.5% | 17.5% |
| 128 | 30.5% | 26.5% |
| 256 | 31.0% | 30.0% |
| 384 | 31.5% | 32.0% |
| 512 | 31.0% | 32.5% |
| 640 | 36.5% | 34.5% |
| 706 (최종) | 39.0% | 36.5% |

## 논문 수치 비교

[arXiv:2508.17032 Table 1](https://arxiv.org/html/2508.17032)의 Qwen3-4B 카트리지는 71/200(35.5%)이며, 이번 A는 **+3.5%p**다. 논문 설정은 512 updates·global batch 128·packed length 1,024이고, 이번 실행은 공개 구현의 global batch 32·packed length 2,048과 축소 대화 32,768개를 사용해 706 updates를 수행했다. 성능 비교에는 이 학습 조건 차이가 함께 반영된다.

## 실행 시간과 재현 정보

| 단계 | 프로세스 시간 |
|---|---:|
| 공개 대화·ICL 자료 준비 | 7분 48초 |
| 4B 재채점 | 1시간 48분 5초 |
| 8B 재채점 | 2시간 39분 33초 |
| A 학습·중간·최종 평가 | 4시간 13분 42초 |
| B 학습·중간·최종 평가 | 4시간 14분 27초 |
| ICL 평가 | 6시간 40분 56초 |
| 합계 | 19시간 44분 31초 |

시간은 단계별 프로세스 시간의 합이며, 환경 설치와 Drive 복사 시간은 별도로 발생한다.

- 실험 소스 commit: `5dc6dbe7fdf2dcb45275181587d53c391c936680`
- Cartridges commit: `ef34ba97a06049c34820506e2c283746284ae5f0`
- 실행: `all`·`train`·`longhealth`·`pilot`·seed 42
- Run: `qwen3-v22-pilot-001`
- [집계 결과·모델 revision·파일 해시·조건별 시간](LONGHEALTH_PILOT_RESULT.json)
- [데이터 출처와 실행 산출물](REPRODUCIBILITY.md)
