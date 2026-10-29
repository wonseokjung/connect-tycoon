# CONNECT TYCOON

**시뮬레이터로 회사의 의사결정 모델을 강화학습시킵니다.**
**RL AI COMPANY DECISION MAKING MODEL** · 상황 · 질문 · 보기를 읽고 선택 · 확률 · 확신으로 답하는 모델

![모델이 회사를 운영하는 하루](docs/images/hero.gif)

모델은 빌딩 맨 위층에서 회사를 운영합니다. 판단 하나가 빛줄기가 되어 층으로 내려가고, 그 결과가 돈 · 고객 만족 · 사고로 돌아옵니다.
학습 버튼을 누르면 모델이 시뮬레이터에서 수백 번 회사를 운영하며 스스로 배우고, 확신이 없을 때는 사람에게 묻습니다.

AI 교육 브랜드 **[CONNECT AI LAB](https://www.youtube.com/@CONNECT-AI-LAB) × [AI CITY BUILDERS](https://www.aicitybuilders.com)**의 강화학습 수업에서 만든 오픈소스 · Apache-2.0

| | |
|---|---|
| 설명 (모션그래픽 · 단계별 안내) | [ezerai.xyz/models/rl](https://www.ezerai.xyz/models/rl) |
| 시뮬레이터 프로그램 | 이 저장소 · 아래 [설치](#설치) |
| 내 RL 모델 바로 올리기 (Colab · 1분 · 게임 · 실험실 · LINK-01 로봇) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/upload_brain_colab.ipynb) |
| Qwen으로 옮기기 (Colab) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/rl_decision_model_qwen_colab.ipynb) |
| 물류 두뇌 → Qwen (Colab ③) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/logistics_brain_qwen_colab.ipynb) |
| LINK-01 로봇 두뇌 → Qwen 의사결정 모델 (Colab ④ · GPU 약 1시간) · [로봇 웹 학습기](https://www.ezerai.xyz/robot-lab/world/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/link01_brain_qwen_colab.ipynb) |
| 첫 결과 (Hugging Face) | [WonseokJayJung/connectailab-rl-jay-v1](https://huggingface.co/WonseokJayJung/connectailab-rl-jay-v1) |

---

## 설치

필요한 것: **Python 3.9 이상** (3.12 권장) · git (없어도 됩니다). GPU는 필요 없습니다. 노트북 CPU로 학습합니다.

### Windows

1. **Python 설치** · [python.org/downloads](https://www.python.org/downloads/)에서 받아 설치합니다. 첫 화면 아래 **Add python.exe to PATH**를 꼭 체크하세요.
2. **코드 받기** · [Git](https://git-scm.com/download/win)이 있으면 아래 `git clone`, 없으면 이 페이지 위의 초록 **Code → Download ZIP**으로 받아 압축을 풉니다.
3. **명령 프롬프트(cmd)**를 열고 (시작 메뉴에서 `cmd`), 한 줄씩 실행합니다.

```bat
git clone https://github.com/wonseokjung/connect-tycoon
cd connect-tycoon
py -m venv .venv
.venv\Scripts\activate
pip install -e ".[game]"
connect-tycoon play
```

- ZIP으로 받았다면 `git clone` 줄은 건너뛰고, `cd` 뒤에 압축을 푼 폴더 경로를 적습니다 (예: `cd %USERPROFILE%\Downloads\connect-tycoon-main`).
- `py`가 없다고 나오면 `python -m venv .venv`로 바꿉니다.
- PowerShell을 쓴다면 `.venv\Scripts\activate` 대신 `.venv\Scripts\Activate.ps1`이고, 막히면 cmd를 쓰는 게 가장 간단합니다.

### Mac

터미널을 열고 한 줄씩 실행합니다. (Python이 없다면 [python.org](https://www.python.org/downloads/)에서 설치)

```bash
git clone https://github.com/wonseokjung/connect-tycoon
cd connect-tycoon
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[game]"
connect-tycoon play
```

### 다음에 켤 때 · 새 버전 받기

| | Windows (cmd) | Mac · Linux |
|---|---|---|
| 다시 켜기 | `cd connect-tycoon` → `.venv\Scripts\activate` → `connect-tycoon play` | `cd connect-tycoon` → `source .venv/bin/activate` → `connect-tycoon play` |
| 새 버전 받기 | 폴더 안에서 `git pull` (ZIP이면 다시 받기) | 폴더 안에서 `git pull` |
| 소리 끄기 (촬영) | `connect-tycoon play --no-sfx --no-music` | 같음 |

`connect-tycoon`을 찾을 수 없다고 나오면 가상환경을 켜지 않은 것입니다 (`activate` 줄). `python -m connect_tycoon play`로도 켤 수 있습니다.

---

## 처음 켜면

타이틀에서 **학습 시작**을 누르면, 화면 왼쪽 아래 안내(STEP 1~6)가 누를 곳을 형광 테두리로 가리킵니다.

| | 화면 | 할 일 |
|---|---|---|
| 준비 | 모델 준비 중 · 규칙 사장을 따라 배우는 중 (몇십 초) | 기다리기 |
| 1 | **v0**이 회사를 운영 · 판단이 LIVE에 흐름 (상황 → 행동 · 확신 · 보상) | 보기 |
| 2 | **보상** 막대 · 이익 · 고객 만족 · 위험 | 무엇이 중요한지 정하기 |
| 3 | **학습 → v1** | 누르기 · 회사를 60번 × 24달 운영하며 강화학습 (몇 분) |
| 4 | **학습 곡선** · 3번마다 시험용 달의 한 달 이익 | 보기 · 가장 좋았던 순간의 모델이 다음 버전 |
| 5 | **실제 데이터 시험** · 학습에 쓰지 않은 진짜 카드 거래 713건 | 몇 건을 잡았나 확인 |
| 6 | **brain.json 저장** · 다운로드 폴더에 | Colab에서 Qwen의 선생님으로 |

v1부터는 모델이 확신 75% 미만인 판단을 만나면 카드가 떠서 사람에게 묻습니다. 고른 답은 시뮬레이터가 바로 채점하고, 라벨이 되어 다음 학습에 들어갑니다.

| 조작 | |
|---|---|
| 학습 → v2… | 지금 모델에서 이어서 학습 (강화학습 + 사람 라벨) |
| 사람 라벨 · 묻기/끄기 | 확신이 낮은 판단을 사람에게 물을지 |
| brain.json 저장 | 다운로드 폴더에 저장 (같은 이름이 있으면 `-2`, `-3`…) |
| 이어하기 | 다시 켜면 타이틀의 **이어하기** · 다른 컴퓨터에선 저장한 brain.json을 **창에 끌어다 놓으면** 그 모델에서 이어서 학습 |
| 스페이스 · Esc · H | 일시정지 · 타이틀 · 처음 안내 다시 보기 |

## 학습은 이렇게 됩니다

| | |
|---|---|
| ![학습](docs/images/training.png) | **① 먼저 따라 배우기 (v0) → 강화학습**<br>규칙 사장의 판단을 따라 배운 뒤(지도학습), 한 달을 수백 번 운영하며 보상이 커진 판단을 더 자주 하도록 고칩니다 (PPO). |
| ![사람 확인](docs/images/review-refund.png) | **② 사람 확인 · 확신이 낮을 때만**<br>모델이 확신하지 못한 판단은 게임을 멈추고 사람에게 묻습니다. 사람이 고른 답은 라벨이 되어 지도학습으로 들어갑니다. |
| ![답 확인](docs/images/feedback.png) | **③ 바로 채점**<br>시뮬레이터가 선택지마다 한 달을 끝까지 다시 돌려, 내 답이 좋았는지 바로 알려 줍니다. |
| ![버전](docs/images/studio.png) | **④ 버전**<br>학습할 때마다 v1, v2… 새 모델이 회사를 맡고, 버전별 한 달 이익이 쌓입니다. |

## 보상은 결과에 줍니다

```
보상 = 이익 × 번 돈 + 고객 만족 × 만족 변화 − 위험 회피 × 사고
```

행동에 점수를 매기지 않습니다. 시뮬레이터가 행동의 결과를 계산하고, 사람은 무엇이 중요한지(세 가중치)만 정합니다.
예를 들어 사기 결제를 통과시키면 그 금액만큼 손해와 사고 1건, 정상 결제를 막으면 금액의 30% 손해와 만족 하락입니다.
손실만 줄이면 "다 막기"가 되므로, 모델은 손실 · 매출 · 만족 사이의 균형을 배웁니다.

## 공개 데이터로 맞춘 회사 · 실제 데이터로 시험

시뮬레이터의 숫자는 지어낸 값이 아니라 여러 업종의 공개 데이터(약 160만 건)에서 가져옵니다.
원본은 저장소에 넣지 않고, 한 번 요약한 **확률표**(`connect_tycoon/data/profiles/general.json`)만 들어 있습니다.
시뮬레이터는 이 확률표대로 손님 · 결제 · 환불 · 문의를 만들고(학습할 때 계속 사용), 결과는 규칙으로 계산합니다.

| 데이터 | 업종 | 쓰는 곳 |
|---|---|---|
| UCI Online Retail II | 소매 | 요일 수요 · 하루 흔들림 · 취소 요일 · 환불 금액 · 고객 등급 |
| Hotel Booking Demand | 숙박 | 요일 수요 · 하루 흔들림 · 취소 요일 · 고객 등급 |
| UCI Bike Sharing | 대여 서비스 | 요일 수요 · 하루 흔들림 |
| ULB Credit Card Fraud | 금융 | 사기 비율 · 사기 점수 · 결제 금액 |
| Olist | 배송 마켓 | 서비스 실패 뒤 만족 하락 |
| Bitext Customer Support | 고객센터 | 문의 유형 → 팀 |

그리고 학습이 끝날 때마다 **학습에 쓰지 않은 실제 기록**(`data/real/`)으로만 다시 채점합니다.

| 처음 보는 실제 카드 거래 713건 (사기 177건) | 손실 줄임 | 잡은 사기 | 실제 수요 재생 · 한 달 이익 |
|---|---|---|---|
| 규칙 | 68% | 138 | 305만 |
| 강의 회사에서 배운 모델 그대로 | 74% | 145 | 1,059만 |
| 일반 회사에서 이어서 학습 | **90%** | **161** | **1,243만** |

```bash
connect-tycoon backtest --brain my-model.json        # 실제 데이터 시험
connect-tycoon train --profile general               # 일반 회사(기본) · lecture는 강의 엔진
python scripts/calibrate.py --data ~/connect-data    # 공개 데이터로 프로필 다시 만들기 (pip install -e ".[data]")
```

순서는 **일반 회사 → 우리 회사**입니다. 한 회사의 실제 기록(주문 · 환불 · 문의 · 결제)을 같은 방법으로 넣으면 그 회사의 프로필이 되고,
일반 모델에서 이어서 학습해 그 회사 전용 의사결정 모델이 됩니다. 실제 데이터에 없는 값(악용 비율 · 기기 신호 · 비용)은 가정이며 프로필의 `assumed`에 적혀 있습니다.

## 언어 모델(Qwen)로 옮기기

시뮬레이터에서 강화학습한 작은 정책(brain.json)을 선생님으로, **Qwen3.5-0.8B**가 글로 된 문제를 읽고 판단하도록 학습합니다.
**강화학습 → 지도학습 → 다시 강화학습** 순서입니다.

| 단계 | 하는 일 |
|---|---|
| ① 문제집 | 판단마다 선택지를 하나씩 바꿔 그 달을 끝까지 다시 돌려, 선택지별 점수를 매깁니다 |
| ② 지도학습 | 점수가 가장 높은 보기 글자를 따라 배웁니다 (LoRA) |
| ③ 강화학습 | 보기 전체를 한 묶음으로 보고, 점수가 평균보다 높은 보기의 확률을 올립니다 (묶음 표준화 + KL) |
| ④ 채점 | 시험 문제 정답률 · Qwen을 회사 사장 자리에 앉혀 한 달 운영 · 실제 데이터 시험 |
| ⑤ 공개 | Hugging Face에 어댑터 + 모델 카드 |

문제 하나는 상황 · 질문 · 보기(A, B, C…)이고, 모델은 글자 하나로 답합니다. 보기 글자의 확률이 선택지별 확률, 가장 큰 값이 확신입니다.

**가장 쉬운 길 · Colab (GPU)** · 브라우저에서 시뮬레이터 → 지도학습 → 강화학습 → 내 Hugging Face 업로드까지 · T4 기준 약 1시간 30분

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/rl_decision_model_qwen_colab.ipynb)

**Mac (Apple Silicon · MLX)**

```bash
pip install -e ".[llm-mac]"
connect-tycoon train --profile general --name teacher --out teacher.json     # 또는 게임에서 저장한 brain.json
connect-tycoon llm data --teacher teacher.json --out llm_data --train-months 10
connect-tycoon llm sft  --data llm_data --out adapters/sft
connect-tycoon llm rl   --data llm_data --from adapters/sft --out adapters/rl
connect-tycoon llm eval --data llm_data --adapter adapters/sft --adapter adapters/rl --months 2 --save eval.json
connect-tycoon llm push --adapter adapters/rl --data llm_data --results eval.json --name jay --version v1 --about-file about_me.md
```

**Windows · Linux (NVIDIA GPU · PyTorch)** · [pytorch.org/get-started](https://pytorch.org/get-started/locally/)에서 내 CUDA에 맞는 설치 명령을 복사해 GPU용 PyTorch를 먼저 설치하고, 위 `llm` 명령에 `--backend torch`를 붙입니다.

```bat
pip install -e ".[llm]"
connect-tycoon llm sft --backend torch --data llm_data --out adapters/sft
```

GPU가 없는 Windows 노트북은 Colab을 쓰세요.

### Hugging Face에 올리기 · 이름 규칙

**게임에서 저장한 brain.json만 바로 올리기 (1분 · GPU 필요 없음)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/upload_brain_colab.ipynb)

- 이름은 **`connectailab-rl-이름-버전`** (예: `connectailab-rl-jay-v1`) · rl = 강화학습 의사결정 모델 · 선생님 정책은 `…-teacher`
- 처음 한 번 `hf auth login` (Write 토큰) 또는 환경변수 `HF_TOKEN` · 토큰은 코드에 적지 않습니다
- `about_me.md`에 나를 알리는 글을 자유롭게 쓰면 (사업 · 서비스 · 채널 · 연락처) 모델 카드 맨 위 '만든 사람'에 그대로 들어갑니다

```bash
connect-tycoon push --brain brain.json --name jay --version v1 --about-file about_me.md   # 시뮬레이터 모델 → connectailab-rl-jay-v1-teacher
connect-tycoon llm collect        # (테스트 중) connectailab-rl-로 시작하는 모델을 같은 시험지로 채점 · 연결(평균 · 다수결 · 가장 확신) 비교
```

첫 결과 · [WonseokJayJung/connectailab-rl-jay-v1](https://huggingface.co/WonseokJayJung/connectailab-rl-jay-v1) · 학습 전 한 달 −1,340만 → 지도학습 +635만 → 강화학습 +757만 · 실제 사기 거래 손실 37% → 71% 줄임

## 물류 실험실 두뇌 → Qwen

웹 학습 실험실(CONNECT AI LAB · 물류센터 하루)에서 내가 알고리즘(Q러닝 · DQN · PPO) · 파라미터 · 보상 · **연습 세상**(함정 카드 · 보상 토큰)을 정해 키운 brain.json도 Qwen 의사결정 모델이 됩니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/logistics_brain_qwen_colab.ipynb)

1. **같은 시뮬레이터** · `connect_tycoon/labenv.py`는 웹(TypeScript)과 걸음마다 똑같이 움직입니다 (난수 · 정렬 · 반올림까지). 웹에서 본 시험 점수가 파이썬에서도 그대로 나옵니다.
2. **문제집** · 두뇌가 하루를 운영하는 동안 10분마다 보기 7개를 하나씩 해 보고, 남은 하루를 두뇌로 끝까지 돌립니다 (서로 다른 미래 8개 평균 · 공식 보상).
3. **지도학습 = 내 두뇌 따라 배우기** · 정답은 두뇌의 답 · 문제는 물류센터장이 보는 화면 글 + 보기 A~G
4. **강화학습 = 확실한 차이만** · 두뇌의 답보다 20점 넘게 낫거나 나쁜 보기만 올리고 내립니다 (회사 버전과 같은 학습 코드 `train_torch` · `train_mlx`)
5. **채점** · Qwen을 물류센터장 자리에 앉혀 웹과 같은 시험 24일을 운영 (`labdata.day_report`) · 점수가 높은 쪽을 올립니다

실험으로 정한 이유 (예시 두뇌 · 시험 24일 하루 점수 · 두뇌 279 · 규칙 사장 86)
- 점수 1등을 정답으로 → 운 좋게 높게 나온 보기만 골라 두뇌보다 나빠짐 (계획으로 돌려도 250)
- 점수를 그대로 강화학습에 → 점수는 '나머지 하루는 두뇌가 운영한다'는 가정이라 임시 인력 호출이 늘 조금 좋아 보여, Qwen이 그것만 고르고 트럭을 대지 않음 (−805)
- 두뇌의 답 + 확실히 나은 보기로 고친 정답 → 미리 돌려 보는 계획으로는 336이지만, 화면 글로는 72%만 맞힐 수 있어 Qwen이 배우면 228
- 두뇌의 답 그대로 1회 → 늘 '급한 트럭 도킹'만 골라 210 (두뇌는 82%가 도킹, 18%가 초과근무)
- **두뇌의 답 + 드문 답(초과근무) 2번 반복 + 2회 학습 → 정답률 90% · 시험 24일 289점 · 두뇌(279)보다 조금 높음** (Mac MLX · Qwen3.5-0.8B · 학습 전 87점)

```bash
connect-tycoon llm data --teacher my-logistics-ppo.json --out lab_data     # 웹 실험실 brain.json이면 물류 문제집
connect-tycoon llm sft  --data lab_data --out adapters/sft
connect-tycoon llm rl   --data lab_data --from adapters/sft --out adapters/rl
connect-tycoon llm eval --data lab_data --adapter adapters/sft --adapter adapters/rl --save results.json   # 시험 24일 운영 점수 포함
connect-tycoon llm push --data lab_data --adapter adapters/rl --results results.json --name jay           # connectailab-rl-jay-v1-logistics-qwen
```

brain.json이 없으면 `connect_tycoon.lab.example()` (웹 기본값 PPO · 시험 279점 · 규칙 사장 86점)로 해 볼 수 있습니다.

## 모델이 답하는 모양

회사의 판단 하나는 요청 하나입니다. 상황(`state`)과 선택지(`criteria`)를 받아 `choice` · `probabilities` · `confidence`로 답합니다.
OpenRouter의 `typesafe/jev-1.13`과 같은 모양이라, 이 형식으로 답하는 모델이면 무엇이든 이 시뮬레이터에서 학습하고 시험할 수 있습니다. → [docs/JEV.md](docs/JEV.md)

## 개발자용

```python
import gymnasium as gym, connect_tycoon
env = gym.make("ConnectTycoon-v0")            # 한 달 = 에피소드, 판단 하나 = 스텝, action_mask 포함
```

```bash
connect-tycoon train --name my-model --profit 2      # 터미널에서 학습
connect-tycoon push --brain my-model.json --name jay --version v1   # Hugging Face · connectailab-rl-jay-v1-teacher
connect-tycoon record --brain my-model.json --mp4    # 모델이 운영하는 하루를 영상으로
```

Colab: [`notebooks/connect_tycoon_colab.ipynb`](notebooks/connect_tycoon_colab.ipynb) · 설계: [docs/DESIGN.md](docs/DESIGN.md)

## 다음

- 우리 회사 데이터 넣기 · CSV를 넣으면 프로필과 실제 데이터 시험이 그 회사 기준으로
- 질문 만들기 · 사람이 질문 · 행동 · 결과표를 직접 정해 자기 회사의 의사결정을 추가
- 산업별 시뮬레이터 · 쇼핑몰 다음으로 병원 · 물류 · 금융 같은 업종마다 하나씩
- 각자 키운 모델을 Hugging Face에 올리고 한곳에 모으기 (Connect AI)

## 만든 곳 · CONNECT AI LAB × AI CITY BUILDERS

**Transforming Lives Through AI Education.** AI를 쓰는 사람에서, AI로 직접 만들고 일하는 사람으로.
AI 멘토 제이(정원석)가 만드는 과정을 그대로 공개하고, 그 과정을 기초부터 따라 할 수 있는 교육으로 만듭니다.

| | |
|---|---|
| **[CONNECT AI LAB · 유튜브](https://www.youtube.com/@CONNECT-AI-LAB)** | 구독자 12만+ · 영상 300편+ 무료 강의. AI 1인 기업, 로컬 AI, 강화학습, 피지컬 AI, JEV까지 직접 해 보며 배웁니다. |
| **[AI CITY BUILDERS](https://www.aicitybuilders.com)** | AI BUSINESS UNIVERSITY · **AI 비즈니스의 정석.** AI 1인 기업부터 기존 기업의 AI 전환까지, 이북 → 영상 실습 → 비즈니스 적용 순서로 배웁니다. |
| **[Connect AI](https://www.ezerai.xyz)** | 독립된 두뇌, 연결되는 지능. 기업 의사결정을 위한 우리 모델 [C1](https://huggingface.co/WonseokJayJung/Connect-C1-0.8B-v0.1)을 멤버들과 함께 연구합니다. |

**이 저장소와 이어지는 수업**

- [강화학습 AI 과정](https://www.aicitybuilders.com/robot): 로봇부터 JEV까지. 상태 · 행동 · 보상으로 문제를 바꾸고, DQN · PPO로 두뇌를 학습시키고, 마지막 강 「강화학습 × JEV」에서 1인 기업 자동화에 적용합니다.
- [무료 강화학습 실험실](https://www.aicitybuilders.com/explore): 탐험하지 않으면 9점, 탐험하면 178점. 브라우저에서 바로 눌러 보는 강의 속 실험.
- 유튜브 재생목록 [AI 강화학습을 알면 인생이 보입니다](https://www.youtube.com/playlist?list=PLSDuOiEMMABU) · JEV 영상 [이론부터 무료 실습까지](https://www.youtube.com/watch?v=p4Mn6KFsSeM) · [내 PC에서 JEV 같은 AI 돌리기](https://www.youtube.com/watch?v=u5EL3dNwJpw)
- 어디서 시작할지 모르겠다면: [무료 AI 학습 길찾기](https://www.aicitybuilders.com/ask)

여기서 학습시킨 의사결정 모델을 각자 Hugging Face에 올리고, 한곳에 연결해 함께 논문으로 씁니다 (CONNECT AI LAB 대표 멤버십 공동 연구).

## 라이선스

코드 · 그림 · 소리는 [Apache-2.0](LICENSE). 글꼴은 [프리텐다드](https://github.com/orioncactus/pretendard)와 [갈무리](https://github.com/quiple/galmuri) (SIL OFL 1.1).
공개 데이터의 출처와 라이선스(CC BY 4.0 · DbCL · CDLA-Sharing · Olist는 CC BY-NC-SA로 요약 통계 세 개만 사용)는 [NOTICE](NOTICE)에 있습니다.

합성 시뮬레이터 위의 교육용 모형입니다. 실제 경영 성과를 보장하지 않습니다.
