# 재현 정보

## 고정한 소스

- 실험 코드: notebook의 `SOURCE_REVISION` Git commit
- Cartridges: `ef34ba97a06049c34820506e2c283746284ae5f0`
- 모델·평가 자료·공개 대화 revision과 파일 순서: [experiment.json](../configs/experiment.json)
- Python 의존성: [requirements.lock](../requirements.lock), PyTorch 2.6.0·Transformers 4.53.0

Colab 2026.07의 Python 3.12, 기본 OS와 CUDA 드라이버를 사용한다. 실제 Python·패키지·GPU·CUDA 버전을 실행 결과에 기록한다.

## 공개 대화

아래 저장소는 원본 Qwen 학습 예제에 지정된 데이터다. 각 저장소의 train split은 65,536개 대화다.

| Benchmark | 저장소 |
|---|---|
| LongHealth | [part 0](https://huggingface.co/datasets/hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0), [part 1](https://huggingface.co/datasets/hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-1) |
| MTOB | [part 0](https://huggingface.co/datasets/hazyresearch/m07d28_mtob_synthesize_qwen3-4b_n65536-0), [part 1](https://huggingface.co/datasets/hazyresearch/m07d28_mtob_synthesize_qwen3-4b_n65536-1) |

고정 revision의 parquet를 읽고 저장소·파일·행 순서로 선택한다. 대화 내용과 token IDs를 보존하며, 각 행의 출처와 원본 파일 SHA-256을 기록한다. A·B에는 동일한 선택 결과를 제공한다.

## 변경 범위

| 변경 | 목적 |
|---|---|
| Teacher 4B·8B의 공통 HF 재채점 | 같은 공개 답변에 대한 teacher 규모 비교 |
| 공개 대화·모델 revision과 파일 해시 고정 | 학습 입력의 식별과 재사용 |
| 초기 cache 공유 | A·B 초기값 통제 |
| Colab 폴더에 config·prediction·metric·checkpoint 저장 | 실행 결과 보존 |
| LongHealth 전체 기록의 ICL 평가 | 자료를 프롬프트로 제공한 기준선 비교 |

재채점 prompt는 공개 system prompt·질문과 고정 chat template으로 구성한다. Thinking 설정은 답변 첫 `<think>` 토큰으로 추정하며, 모호한 토큰 배열은 오류로 처리한다. 공개 logprob와 A 재채점 결과의 차이에는 입력 복원과 모델·연산 구현 차이가 반영될 수 있다.

ICL은 공개 `baseline_longhealth.py`의 resource·system prompt와 `ICLBaseline`·`_process_batch`를 사용한다. 모델·tokenizer는 Qwen3-4B의 고정 revision을 사용하고, thinking·temperature·출력 상한은 A·B의 평가 config와 맞춘다. 원본 ICL 예제의 2,048-token 출력 상한은 비교 조건의 512 tokens로 맞춘다. 모델 서버 대신 HF SDPA 추론을 사용하며 문항을 하나씩 처리한다.

ICL 문맥 한도는 131,072 tokens, YaRN은 `factor=4.0`, `original_max_position_embeddings=32768`이다. 전체 기록을 system prompt에 넣고, 1,024-token 청크 prefill과 일반 KV cache를 사용한다. 매 문항 뒤 cache를 공통 system prompt 길이로 되돌린다. 입력을 자르는 처리는 없으며 한도 초과는 오류로 기록한다.

[패치](../patches/cartridges.patch)는 tokenizer·metric revision 고정, Qwen 모델명 대소문자 호환, cache 복원 시 token 축 수정, step 상한 적용, LongHealth ICL 예제의 사용하지 않는 잘못된 import 제거를 포함한다. Smoke는 최대 2 steps를 사용하고, pilot·main은 공개 예제의 epoch를 완료한다. 손실 함수, sparse 확률 처리, packing과 평가 채점은 공개 구현을 따른다.

## 실행 산출물

- `study.json`, `resolved-config.json`: 실험 설정과 소스·환경 해시
- `prepare.summary.json`, `data/`: 공개 데이터 출처·해시와 선택한 대화
- `scores/`: teacher별 soft targets와 재채점 지표
- `initial-cache.pt`, `seed-*/`: 초기 cache, 조건별 checkpoint·prediction·metric·평가 RNG
- `prepare-icl.summary.json`, `icl-config.json`, `data/icl-*.txt`: ICL 전체 자료·system prompt·설정·입력 길이·해시
- `seed-*/ICL/`, `seed-*/icl.summary.json`: ICL prediction·지표·평가 RNG·시간·메모리

같은 `RUN_NAME`·benchmark·profile에서 완료된 단계를 재사용한다. `MODE=evaluate`는 A·B의 최종 checkpoint와 저장된 RNG로 재평가한다. ICL은 같은 원문과 seed로 다시 평가한다. 수치 비교에는 같은 release·입력·seed·GPU·환경을 사용하며, GPU 연산에 따른 수치 변동이 생길 수 있다.

마지막 셀은 전체 결과를 Drive에 복사하고 `drive.flush_and_unmount()`로 동기화한 후 `runtime.unassign()`을 호출한다. 자동 삭제는 모든 선택 조건의 실행과 결과 저장이 완료되고 Drive 백업을 사용할 때 수행한다.

Drive 연결 셀은 설치보다 앞에 있다. 연결을 승인한 런타임에서 전체 노트북을 실행하면 기존 mount를 사용한다. 새 런타임에서는 Drive 연결 셀을 다시 실행한다.

## 데이터 출처와 라이선스

| 자료 | 배포처 표기 |
|---|---|
| [LongHealth](https://github.com/kbressem/LongHealth/blob/479922279ce8fa528fe2cd87d854c6a2f8365d21/README.md#intended-usage) | Apache-2.0. README의 연구용 사용·예제 재게시 안내 적용. |
| [MTOB](https://github.com/lukemelas/mtob/blob/0f5c939dc4e7ba7b2092eb115811941b32325667/README.md#license) | 코드 MIT, 문법서·데이터·인간 baseline 출력 CC BY 4.0. 공식 배포본은 암호화 ZIP과 canary 사용. |
| HazyResearch 공개 대화 | 고정 revision의 dataset card에 별도 라이선스 미표기. 원문 자료의 출처·조건 함께 확인. |

평가 자료와 공개 대화는 Colab 실행 시 배포처에서 다운로드한다. 저장소의 Apache-2.0 라이선스는 이 프로젝트 코드에 적용된다. 대화·token IDs·prediction·로그에 포함된 원문과 정답에는 각 자료의 이용 조건이 적용된다.

## 검증

설치 셀의 검사는 공개 학습 설정 일치, tokenizer, parquet 선택·토큰 보존·prompt 구성, sparse 처리, cache 복원과 실행 경로를 확인한다. ICL에는 원본 평가 설정 일치와 작은 임의 모델의 prefill·문항 사이 cache 분리 검사를 포함한다. Colab CPU·Python 3.12.3·PyTorch 2.6.0·Transformers 4.53.0에서 설치와 15개 연결 검사를 통과했다.

[v0.4.3 LongHealth pilot](LONGHEALTH_PILOT_RESULT.md)은 Colab A100 40GB에서 A·B 각 706 updates와 ICL 전체 200문항 평가를 완료했다. Drive에서 내려받은 최종 prediction·metric·summary를 대조했고, A·B 최종 checkpoint 해시가 실행 summary와 일치했다. 세 조건의 문항 집합과 원본 채점 결과도 일치했다.

[v0.1.4 pilot 결과](A_PILOT_RESULT.md)는 자체 생성 대화 512개를 사용한 이전 프로토콜의 기록이다.
