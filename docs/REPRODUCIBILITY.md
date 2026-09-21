# 재현 정보

## 고정한 소스

- 실험 코드: notebook의 `SOURCE_REVISION`에 고정한 Git commit. 실행 시 `study.json`에 기록한다.
- Cartridges: `ef34ba97a06049c34820506e2c283746284ae5f0`
- Tokasaurus: `9ccbb92ed5c042c19b262db3791f4330dc3c87d2` — 공개 Cartridges 배포 예제가 참조하는 `geoff/cartridges` branch의 고정 commit.
- 모델·데이터 revision: [experiment.json](../configs/experiment.json)
- Python 의존성: [requirements.lock](../requirements.lock)

PyTorch 2.6.0·Transformers 4.53.0은 Tokasaurus의 고정 의존성이며 Cartridges의 선언 범위와 호환된다. Colab의 기본 OS와 CUDA 드라이버를 사용하고 실제 버전을 실행 결과에 기록한다.

## 변경 범위

| 변경 | 목적 |
|---|---|
| 두 생성 모델과 두 scoring teacher | teacher 크기의 2×2 비교 |
| 발췌 batch·대화·초기 cache 파일 공유 | 비교 조건의 입력과 초기값 통제 |
| HF teacher-forcing scoring | 다른 teacher가 동일 답변의 분포를 계산; A–D에 공통 적용 |
| 단일 GPU에서 batch 순차 처리와 서버 종료 | 생성과 학습을 순차 실행 |
| Colab 실행 폴더에 config·prediction·metric·checkpoint 저장 | W&B 계정 없이 결과 보존 |

합성 서버의 batch 크기는 32로 유지하며, KV 수용량은 설정 파일의 32,768 tokens를 사용한다. 발췌 목록의 seed는 82다. 원본 생성 서버의 sampling 결과는 저장한 대화 파일을 기준으로 공유한다.

[패치](../patches/cartridges.patch)는 tokenizer·metric revision 고정, MTOB tokenizer revision 전달, Qwen 모델명 대소문자 호환, cache 복원 시 token 축 수정, smoke·pilot의 step 상한 적용만 포함한다. 손실 함수, sparse 확률 처리, 데이터 패킹과 채점 규칙은 공개 구현을 유지한다.

## 실행 산출물

실행 폴더에 다음 산출물이 저장된다.

- 실행 config, 단계별 환경 정보, 데이터·코드·패키지 lockfile 해시
- 생성 대화 token IDs와 teacher별 soft targets
- 초기 cache 및 조건·seed별 최종 checkpoint
- 문항별 prediction, 지표, 평가 직전 RNG 상태

`experiment.py evaluate`는 저장된 checkpoint와 최종 평가 RNG 상태로 재평가한다. 같은 학습 입력을 재사용하려면 생성 대화·soft targets·초기 cache가 필요하다. 합성부터 새로 실행하면 GPU 연산과 sampling에 따른 변동이 생길 수 있다.

합성 서버의 GPU 메모리는 별도 프로세스에 속하므로 현재 합성 요약의 `peak_allocated_bytes`는 미측정(`null`)으로 기록한다. 다른 GPU 단계에서는 해당 실행 프로세스의 PyTorch 최대 할당량을 기록한다.

## 평가 자료와 라이선스

| 자료 | 배포처의 표기 |
|---|---|
| [LongHealth](https://github.com/kbressem/LongHealth/blob/479922279ce8fa528fe2cd87d854c6a2f8365d21/README.md#intended-usage) | 저장소 라이선스 Apache-2.0. README는 연구용 사용과 예제 재게시 자제를 안내한다. |
| [MTOB](https://github.com/lukemelas/mtob/blob/0f5c939dc4e7ba7b2092eb115811941b32325667/README.md#license) | 코드 MIT, 문법서·데이터·인간 baseline 출력은 CC BY 4.0. 공식 데이터는 오염 방지를 위해 암호화 ZIP과 canary를 사용한다. |

평가 데이터는 Colab에서 고정 revision의 공식 자료를 내려받는다. 각 데이터에는 해당 라이선스의 출처·저작권·변경 표시 조건이 적용된다.

`predictions-*.json`에는 평가 질문과 정답이 포함된다. 생성 대화·token IDs·실행 로그에도 원문이 남을 수 있으며, 이 자료에도 해당 데이터의 라이선스가 적용된다.

## 검증 범위

CPU 검사는 공개 예제와 main 설정의 일치, tokenizer·parquet·토큰 위치, 원본 sparse 처리, cache 복원, 데이터 문항 수·채점 함수와 notebook 구문을 확인한다. GPU 서버 시작, 실제 학습·추론, 전체 notebook 실행은 검증 전이다. Python 패키지 lockfile은 해결된 설치 목록이며 GPU 실행 검증 결과와 구분한다.
