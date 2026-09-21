# Qwen3 teacher 규모 비교

## 비교 조건

기준 구현은 [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges/tree/ef34ba97a06049c34820506e2c283746284ae5f0)의 Qwen benchmark 예제다. 공개 Qwen3-4B 대화를 고정하고, 더 큰 teacher의 soft targets가 Qwen3-4B student의 카트리지 성능을 높이는지 비교한다.

| 조건 | Scoring teacher | Student |
|---|---|---|
| A | Qwen3-4B | Qwen3-4B |
| B | Qwen3-8B | Qwen3-4B |

A·B는 동일한 문맥·질문·답변 토큰과 초기 cache를 사용한다. Teacher별 soft targets는 같은 Hugging Face teacher-forcing 경로에서 계산한다. 학습 seed는 42·123·2026이다.

## 데이터와 학습·평가

각 benchmark의 공개 `*_train.py`에 지정된 Hugging Face 대화 저장소 두 개를 사용한다. 저장소당 65,536개이며 revision과 파일 순서를 [설정 파일](configs/experiment.json)에 고정한다. 공개 대화의 system prompt·질문·답변 token IDs를 보존한다.

재채점 입력은 저장된 system prompt와 질문에 Qwen chat template을 적용해 구성한다. Thinking 설정은 답변 첫 `<think>` 토큰 여부로 추정한다. 해당 설정과 원래 prompt token IDs는 공개 데이터에서 제공되지 않으므로 A의 공개 logprob 대비 차이를 진단 지표로 기록한다.

| 항목 | LongHealth | MTOB |
|---|---|---|
| 대화의 원문 자료 | 환자 1–10의 기록 | LaTeX 문법서와 parallel training sentences |
| 평가 | 200문항 정확도 | Kalamang → English 50문장 corpus chrF |
| Cartridge 길이 | 2,048 | 4,096 |
| Epoch | 2 | 1 |
| Global batch / packed length | 32 / 2,048 | 32 / 2,048 |
| Optimizer / learning rate | Adam / 0.02 | Adam / 0.02 |
| 평가 batch / temperature | 32 / 0.3 | 16 / 0 |
| 평가 출력 상한 | 512 tokens | 128 tokens |

학습·평가 config는 공개 `*_train.py`에서 직접 불러온다. 초기화는 `KVFromText` 기본 gradient 문서, packing은 truncate, soft targets는 top-20과 원본 `flatten(0.99)`를 사용한다. Backbone을 동결하고 카트리지 K·V를 학습하며 첫 attention-sink token의 원본 설정을 유지한다. 학습 전·128 steps마다·최종 평가를 수행한다.

LongHealth는 원본 `<answer>` 추출과 선택지 매칭 규칙을 사용한다. 태그가 빠지면 첫 선택지로 처리하며 누락 수를 기록한다. MTOB는 원본 `batch_score_with_answers`를 사용한다.

## 실행 규모

| Profile | 대화 선택 | 학습 | 평가 |
|---|---|---|---|
| smoke | 첫 32개 | 64-token cache, batch 1, 최대 2 steps | 2문항, batch 1 |
| pilot | 첫 512개 | 원본 cache·batch, 최대 16 steps | 전체 |
| main | 전체 131,072개 | 공개 예제의 epoch | 전체 |

대화 선택 순서는 설정의 저장소 순서 → 파일 순서 → 파일 내 행 순서다. Smoke·pilot은 이 순서의 앞부분을 사용하는 축소 실행이다. 학습 시 공개 구현의 seed 기반 shuffle·packing을 적용한다. Main 전체 비교는 2조건 × 2 benchmarks × 3 seeds, 총 12회 학습이다.

## 분석

Benchmark별로 seed 평균·표준편차와 같은 seed의 B−A 차이를 보고한다. 학습 전·중·후 prediction과 지표, target token 수, packed batch 수, 실제 optimizer updates, 시간·최대 GPU 메모리를 기록한다.

원본 sparse 처리에서 top-20 질량이 0.99에 못 미치면 첫 항목만 남는 동작을 유지하며 teacher별 미달 비율을 기록한다. A의 공개 logprob와 재채점 logprob의 평균·최대 절대 차이도 보존한다.
