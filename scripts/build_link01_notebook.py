"""notebooks/link01_decision_model_colab.ipynb 만들기 · LINK-01 로봇 두뇌(brain.json) → 언어 모델 지도학습 → 강화학습 → 같은 코스 20판 → Hugging Face."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import build_notebook as B

REPO = B.REPO
WORLD = "https://www.ezerai.xyz/robot-lab/world/"
CELLS = [
    ("md", f"""# LINK-01 로봇 의사결정 모델 만들기 (Colab)

**CONNECT AI LAB × AI CITY BUILDERS** · [CONNECT AI WORLD 로봇 웹 학습기]({WORLD}) · [CONNECT TYCOON](https://github.com/wonseokjung/connect-tycoon)

웹 학습기에서 내가 코스(장애물 · 목표 · 출발점)를 만들고 브라우저 안에서 PPO로 키운 **LINK-01 결정 두뇌(brain.json)** 를,
언어 모델에 옮겨 담아 **로봇 의사결정 AI 모델**로 만들고 Hugging Face에 올립니다.

1. **두뇌 불러오기** · 내 brain.json + 코스 파일 world.json (없으면 예시 두뇌 · 「장애물 피하기」)
2. **문제집** · 두뇌가 코스를 걷는 동안 0.5초마다, 두뇌가 본 숫자(목표 거리 · 방향 · 거리 센서 · 속도)를 글로 바꾼 문제 · 정답 = 두뇌의 답
3. **학습 전** · 아무것도 배우지 않은 언어 모델을 로봇의 결정 자리에 앉혀 봅니다
4. **지도학습** · 언어 모델이 내 두뇌의 답을 그대로 배웁니다 (LoRA)
5. **강화학습** · 두뇌의 답보다 확실히 나은 보기는 올리고, 확실히 나쁜 보기는 내립니다
6. **공개** · 같은 코스 20판 성공률이 더 높은 쪽을 내 Hugging Face에 `connectailab-rl-이름-버전-link01`

> 메뉴 **런타임 → 런타임 유형 변경 → GPU (T4 이상)**. 왼쪽 🔑(보안 비밀)에 `HF_TOKEN`(Write 권한)을 넣어 두면 마지막 셀에서 바로 올라갑니다.

**걷기에 대해** · 웹에서는 두뇌가 고른 방향이 미리 배운 걷기 신경망 → MuJoCo 물리로 걷습니다. 이 노트북은 로봇 모델 없이, 그 걷기를 실제로 재서 만든 **2D 걷기 근사**(명령마다 실제 속도 · 회전 · 흔들림)로 걷습니다. 넘어짐이 없어서 웹보다 성공률이 조금 후할 수 있습니다."""),
    ("code", f"""# 설치 (2~3분)
!pip -q install "connect-tycoon @ git+{REPO}" "transformers==5.18.0" peft accelerate huggingface_hub
!pip -q install flash-linear-attention || echo "(빠른 계산 모듈 없이 진행 · 조금 느릴 뿐 결과는 같습니다)"
!pip -q uninstall -y torchao   # Colab 기본 torchao(0.10)가 최신 peft와 맞지 않습니다 · 이 학습에는 필요 없습니다
import torch, connect_tycoon as ct
print("CONNECT TYCOON", ct.__version__, "· GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음 (런타임을 GPU로 바꾸세요)")""", None),
    ("md", f"""## 1 · 내 LINK-01 두뇌 불러오기

[로봇 웹 학습기]({WORLD})에서 학습을 마치고 **brain.json** 칸의 **받기**로 받은 파일(`link01-과제이름-mine-brain.json`)과, **world.json** 칸의 **코스 저장**으로 받은 `…world.json`입니다.
- **끌어다 놓기** · Colab 왼쪽 **📁(파일)** 창에 두 파일을 같이 끌어다 놓고 실행 (가장 확실)
- **파일 선택 버튼** · 파일 창에 없으면 이 셀 아래에 [파일 선택] 버튼이 뜹니다 (두 파일을 같이 골라도 됩니다)
- world.json이 없으면 그 과제의 **무작위 코스**(웹 공식 시험과 같은 코스들)로 진행합니다
- `USE_MY_BRAIN`을 끄면 예시 두뇌(「장애물 피하기」 · 내 코스 하나에서 20갱신)로 해 봅니다

**왜** · 두뇌가 배운 그 코스에서 문제를 만들고, 그 코스에서 의사결정 모델을 걷게 해야 비교가 됩니다.

✅ **정상이면** · 코스 지문이 두뇌의 trained_on에 있고, 2D 걷기로 다시 걸어 본 두뇌 성공률이 웹의 「이 코스 20판」 근처입니다 (예시 두뇌 100%)"""),
    ("code", """USE_MY_BRAIN = True                                    #@param {type:"boolean"}
UPLOAD_BUTTON = True                                   #@param {type:"boolean"}
import os
from connect_tycoon.link01 import sim, questions as Q
from connect_tycoon.link01.brain import WorldBrain, example, is_world

if USE_MY_BRAIN:
    from connect_tycoon.colab import find_brain, find_course   # 📁에 있으면 그것, 없으면 [파일 선택] 버튼
    path = find_brain(upload_button=UPLOAD_BUTTON)
    assert is_world(path), "LINK-01 로봇 두뇌(connect-world-brain/1)가 아닙니다 · 게임 두뇌는 Colab ①, 물류 두뇌는 Colab ③을 쓰세요"
    brain, WORLD = WorldBrain.from_file(path), find_course(os.path.dirname(os.path.abspath(path)))
else:
    brain, WORLD = example()
COURSE = None
if WORLD:
    task, COURSE, HASH = sim.load_course(WORLD)
    assert task == brain.task, f"코스는 「{sim.TASK_KO[task]}」, 두뇌는 「{sim.TASK_KO[brain.task]}」입니다 · 같은 과제의 world.json을 넣어 주세요"
    print(f"코스 {os.path.basename(WORLD)} · 지문 {HASH}", "· 이 두뇌가 배운 코스 ✓" if HASH in brain.trained_on else "· ⚠ 이 두뇌가 배운 코스가 아닙니다")
else:
    print("world.json 없음 · 그 과제의 무작위 코스(씨앗별로 다른 코스)로 진행합니다")
mine = sim.evaluate(brain.act, brain.task, COURSE, rays=brain.rays)
print(f"두뇌 {brain.name} · {sim.TASK_KO[brain.task]} · {brain.contract}")
print(f"  웹 공식 시험(무작위 코스 20판) {brain.test.get('score', 0):.0%} · 2D 걷기로 {'이 코스' if COURSE else '무작위 코스'} 20판 {mine['success']:.0%} (부딪힌 판 {mine['collide']:.0%})")""", None),
    ("md", """## 2 · 문제집 · 두뇌의 결정을 글 문제로 (약 3~6분)

**무엇을** · 두뇌가 코스를 걷는 동안 0.5초마다, 두뇌가 본 숫자를 사람이 읽는 글로 바꿉니다 (목표까지 1.84 m · 왼쪽 21° · 거리 센서 정면 0.84 m …).
보기는 **직진 · 왼쪽으로 · 오른쪽으로 · 제자리 좌회전 · 제자리 우회전** (물건 고르기는 머그컵 · 사과 · 랜턴 · 화분)이고, **정답은 내 두뇌의 답**입니다.
길을 조금 벗어난 상황도 배우도록 가끔(20%) 두뇌 대신 아무 보기로 걸어 봅니다. 그때도 정답은 두뇌의 답입니다.

**점수** · 강화학습에서 씁니다. 보기마다 '그렇게 하고 나머지 10초는 두뇌가 걸었으면 받은 보상'(흔들림 2번 평균 · 웹과 같은 공식 보상)입니다. 두뇌의 답보다 1점 넘게 낫거나 나쁜 보기만 반영합니다.

**정답률 상한** · 글에 적힌 숫자는 반올림되어 있어서, 두뇌에게 그 반올림한 숫자를 다시 줘도 같은 답을 내는 비율이 모델이 닿을 수 있는 정답률의 상한입니다.

✅ **정상이면** · 20판이면 학습 약 600~1,500문제 (과제마다 다름) · 정답률 상한 99% 안팎"""),
    ("code", """from connect_tycoon.llm import data as D
TRAIN_EPISODES = 20                                    # 늘릴수록 문제가 많아지고 지도학습이 길어집니다 (한 판 ≈ 30~80문제)
splits = {"train": range(10000, 10000 + TRAIN_EPISODES), "valid": range(20000, 20004), "test": range(30000, 30006)}
print(Q.build(brain, "link01_data", COURSE, splits=splits, log=None))
import json
meta = json.load(open("link01_data/meta.json"))
rows = D.load("link01_data", "train")
ex = rows[len(rows) // 3]
print(); print(ex["task"]["state"]); print(ex["task"]["question"])
for o, s in zip(ex["task"]["options"], ex["scores"]): print(f"  {o['label']} {o['description']:28s} 점수 {s:6.2f}")
print("정답(두뇌의 답)", ex["answer"], "· 정답률 상한", f"{meta['text_ceiling']['test']:.1%}")""", None),
    ("md", """## 3 · 학습 전 · 언어 모델을 로봇의 결정 자리에 앉혀 보기 (약 2~4분)

**무엇을** · 아무것도 배우지 않은 언어 모델에게 시험 문제를 풀게 하고(두뇌와 같은 답을 고르는 비율), 같은 코스를 20판 걷게 합니다 (씨앗 91001–91020 · 웹 공식 시험과 같은 씨앗).
**왜** · 학습 효과를 보려면 출발점을 먼저 재야 합니다.

✅ **정상이면** · 정답률 낮음 · 성공률이 두뇌보다 낮음"""),
    ("code", """from connect_tycoon.llm import train_torch as T
test = D.load("link01_data", "test")
results = {"두뇌 (선생님)": {"accuracy": 1.0, **{f"drive_{k}": v for k, v in mine.items() if k in ("success", "collide", "wrong", "ret", "time_s")}, "drive_runs": mine["n"]}}
base = T.Scorer()
results["학습 전"] = D.agreement(test, base.pick)
results["학습 전"].update(Q.drive_report(base.pick, brain, COURSE, log=None))
print(results["학습 전"])
del base; torch.cuda.empty_cache()""", None),
    ("md", """## 4 · 지도학습 · 내 두뇌의 답을 따라 배우기 (T4 예상 약 20~30분)

**무엇을** · 두뇌가 고른 보기 글자를 맞히도록 LoRA를 학습합니다 (모델 전체가 아니라 작은 덧붙임 층만).
**왜** · 두뇌(작은 신경망)가 숫자로 하던 판단을, 언어 모델이 글을 읽고 하도록 옮깁니다.
**드문 답은 여러 번** · 두뇌가 대부분 '직진'을 고르면, 그대로 배운 모델은 언제 꺾을지 배우기 전에 늘 직진만 고릅니다. 그래서 드문 답을 몇 번 반복해 보여 줍니다.

✅ **정상이면** · 정답률이 크게 오르고, 같은 코스 성공률이 두뇌 근처"""),
    ("code", """sft = T.sft("link01_data", "adapters/sft", epochs=2, lr=1e-4, batch=8)
results["지도학습"] = D.agreement(test, sft.pick)
results["지도학습"].update(Q.drive_report(sft.pick, brain, COURSE, log=None))
print(results["지도학습"])
del sft; torch.cuda.empty_cache()""", None),
    ("md", """## 5 · 강화학습 · 걸어 본 보상으로 고치기 (T4 예상 약 10~15분)

**무엇을** · 두뇌의 답보다 확실히(1점 넘게) 나은 보기의 확률은 올리고, 확실히 나쁜 보기(예: 장애물 쪽으로 꺾기)는 내립니다. 지도학습 모델에서 너무 멀어지지 않게 붙잡아 둡니다(KL).
**손실** · −Σ p(보기) · 장점(보기) + β · KL(p ‖ 지도학습 모델)

✅ **정상이면** · 성공률이 지도학습과 비슷하거나 오릅니다. 마지막 셀은 **같은 코스 20판 성공률이 더 높은 쪽**(같으면 정답률이 높은 쪽)을 올립니다"""),
    ("code", """rlm, hist = T.rl("link01_data", "adapters/sft", "adapters/rl", steps=100, batch=8, lr=5e-6, beta=0.2, max_rows=1500)
results["강화학습"] = D.agreement(test, rlm.pick)
results["강화학습"].update(Q.drive_report(rlm.pick, brain, COURSE, log=None))
print(f"{'':14s} 정답률  같은 코스 20판 성공  부딪힌 판")
for k, v in results.items(): print(f"{k:14s} {v['accuracy']:6.1%}  {v['drive_success']:8.0%}  {v['drive_collide']:12.0%}")
key = lambda k: (results[k]["drive_success"], results[k]["accuracy"])
BEST = "adapters/rl" if key("강화학습") >= key("지도학습") else "adapters/sft"
print("올릴 모델 ·", "강화학습" if BEST.endswith("rl") else "지도학습", "(같은 코스 성공률이 더 높은 쪽)")""", None),
    ("md", """## 6 · Hugging Face에 올리기

1. https://huggingface.co/settings/tokens 에서 **Write** 토큰을 만듭니다.
2. Colab 왼쪽 🔑(보안 비밀)에 이름 `HF_TOKEN`, 값에 토큰을 넣고 **노트북 액세스**를 켭니다.
3. **이름**을 영문으로 적습니다. 업로드 이름은 `connectailab-rl-이름-버전-link01` 입니다 (예: `connectailab-rl-jay-v1-link01`).
4. **나를 알리는 글**을 자유롭게 적으면 모델 카드 맨 위 '만든 사람'에 들어갑니다.

토큰은 코드에 적지 않습니다."""),
    ("code", """from google.colab import userdata
from connect_tycoon.link01.hub import push_decision_model
from connect_tycoon.llm.hub import make_author

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
assert HF_TOKEN, "왼쪽 🔑에 HF_TOKEN(Write 권한)을 넣고 노트북 액세스를 켜 주세요"

NAME = "jay"                                         #@param {type:"string"}
VERSION = "v1"                                       #@param {type:"string"}
PRIVATE = False                                      #@param {type:"boolean"}

# 나를 알리는 글 · 형식 없이 자유롭게 (모델 카드 맨 위 '만든 사람'에 그대로 들어갑니다)
ABOUT_ME = '''
'''
author = make_author(ABOUT_ME)

meta["chosen"] = "강화학습" if BEST.endswith("rl") else "지도학습"
url = push_decision_model(BEST, name=NAME, version=VERSION, results=results, meta=meta, token=HF_TOKEN, private=PRIVATE, author=author)
print("올라갔습니다 ·", url)""", None),
]


if __name__ == "__main__":
    B.CELLS = CELLS
    print(B.build(os.path.join(os.path.dirname(__file__), "..", "notebooks", "link01_decision_model_colab.ipynb")))
