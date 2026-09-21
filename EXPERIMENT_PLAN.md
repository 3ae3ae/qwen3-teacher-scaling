# Qwen3 teacher 규모 비교

## 비교 조건

기준 구현은 [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges/tree/ef34ba97a06049c34820506e2c283746284ae5f0)의 Qwen benchmark 예제다. Student는 Qwen3-4B이며, 합성 모델 G와 분포 계산 teacher T를 비교한다.

| 조건 | G | T |
|---|---|---|
| A | Qwen3-4B | Qwen3-4B |
| B | Qwen3-4B | Qwen3-8B |
| C | Qwen3-8B | Qwen3-4B |
| D | Qwen3-8B | Qwen3-8B |

주 비교는 B−A다. 보조 비교는 D−C, C−A, D−A다. A/B와 C/D는 각각 같은 대화 token IDs를 사용한다. G4/G8에는 같은 발췌·seed prompt batch를 제공한다. 학습 seed는 42·123·2026이며 합성 데이터와 soft targets를 재사용한다.

## 공개 예제의 설정

| 항목 | LongHealth | MTOB |
|---|---|---|
| Corpus | 환자 1–10의 전체 기록 | LaTeX 문법서와 parallel training sentences |
| 평가 | 200문항 정확도 | Kalamang → English 50문장 corpus chrF |
| Cartridge 길이 | 2,048 | 4,096 |
| Epoch | 2 | 1 |
| Global batch / packed length | 32 / 2,048 | 32 / 2,048 |
| Optimizer / learning rate | Adam / 0.02 | Adam / 0.02 |
| 합성 batch / round | 32 / 1 | 32 / 1 |
| Thinking 확률 | 0.75 | 0.2 |
| 평가 batch / temperature | 32 / 0.3 | 16 / 0 |
| 평가 출력 상한 | 512 tokens | 128 tokens |

이 값은 공개 `*_train.py`와 `*_synthesize.py`의 config를 직접 불러온다. 초기화는 `KVFromText` 기본 gradient 문서, packing은 truncate, 분포 저장은 top-20과 원본 `flatten(0.99)`다. Backbone을 동결하고 Cartridge K·V를 학습한다. 첫 attention-sink token의 원본 설정을 유지한다.

질문·답변 생성 temperature는 각각 0.6·0.0, 출력 상한은 512·1,024 tokens다. 원본 resource sampler와 다섯 seed prompt 유형을 사용한다. 생성은 공개 Tokasaurus client와 `SelfStudySynthesizer`가 수행한다. 학습 중 128 steps마다 수행하는 평가와 최종 평가도 유지한다.

LongHealth는 원본 `<answer>` 추출과 선택지 매칭 규칙을 사용한다. 태그가 빠지면 첫 선택지로 처리하며 누락 수를 함께 기록한다. MTOB는 원본 `batch_score_with_answers`를 사용한다.

## 실행 규모

| Profile | 생성 모델당 대화 수 | 학습 |
|---|---:|---|
| smoke | 32 | 64-token cache, batch 1, 최대 2 steps |
| pilot | 512 | 원본 cache·batch 설정, 최대 16 steps |
| main | 131,072 | 공개 예제의 두 데이터 shard에 대응, 지정 epoch |

Smoke는 질문·답변 생성 상한을 64 tokens, 평가를 2문항으로 줄인다. LongHealth smoke는 환자 11·12를 사용한다. Main은 4조건 × 2 benchmarks × 3 seeds, 총 24회 학습이다.

## 분석

Benchmark별로 seed 평균·표준편차와 동일 seed의 paired 차이를 보고한다. 학습 전·중·후 평가 결과를 함께 보존한다. 생성 모델에 따른 답변 길이 차이를 파악하도록 target 수, packed batch 수, optimizer steps, 시간·메모리도 기록한다.

모든 조건의 soft targets는 저장된 답변에 대한 동일한 Hugging Face teacher-forcing 경로에서 계산한다. G=T인 경우 원본 생성 서버의 logprob와 차이를 측정하여 scoring 연결을 확인한다. 원본 sparse-target 처리에서 top-20 질량이 0.99에 못 미치면 첫 항목만 남는 동작을 유지하고 teacher별 미달 비율을 기록한다.

QASPER는 원문과 평가 정답이 공개되어 있다. 실험 대상은 완결된 공개 학습 예제가 있는 LongHealth·MTOB다.
