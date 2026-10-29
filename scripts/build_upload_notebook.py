"""notebooks/upload_brain_colab.ipynb 만들기 · 게임에서 저장한 brain.json을 Hugging Face에 바로 올리기 (1분 · GPU 필요 없음)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import build_notebook as B

REPO = B.REPO
CELLS = [
    ("md", """# 내 RL 모델을 Hugging Face에 올리기 · 1분

**CONNECT AI LAB × AI CITY BUILDERS** · [CONNECT TYCOON](https://github.com/wonseokjung/connect-tycoon)

게임에서 **brain.json 저장**으로 내려받은 모델, 웹 **학습 실험실**에서 내가 파라미터를 정해 키운 두뇌, 또는 [CONNECT AI WORLD 로봇 웹 학습기](https://www.ezerai.xyz/robot-lab/world/)에서 키운 **LINK-01 로봇 두뇌**(brain.json)를 내 Hugging Face 계정에 올립니다. GPU가 필요 없습니다.

1. https://huggingface.co/settings/tokens 에서 **Write** 토큰을 만듭니다.
2. Colab 왼쪽 🔑(보안 비밀)에 이름 `HF_TOKEN`, 값에 토큰을 넣고 노트북 액세스를 켭니다. 토큰은 코드에 적지 않습니다.
3. 아래 셀을 위에서부터 차례로 실행합니다.

업로드 이름 · 게임 두뇌 `connectailab-rl-이름-버전-teacher` · LINK-01 로봇 두뇌 `connectailab-rl-이름-버전-link01` (예: `connectailab-rl-jay-v1-link01`).
Qwen까지 학습해 올리려면 [Qwen 강화학습 Colab](https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/rl_decision_model_qwen_colab.ipynb) · LINK-01은 [LINK-01 → Qwen Colab](https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/link01_brain_qwen_colab.ipynb)을 쓰세요."""),
    ("code", f"""# 설치 (30초)
!pip -q install "connect-tycoon @ git+{REPO}" huggingface_hub
import connect_tycoon as ct
print("CONNECT TYCOON", ct.__version__)""", None),
    ("md", """## 1 · brain.json 고르기

게임에서 저장한 파일(다운로드 폴더의 `회사이름-v2.json` 같은 이름) 또는 로봇 학습기에서 받은 `link01-과제이름-mine-brain.json`(예: `link01-avoid_obstacles-mine-brain.json`)을 넣는 방법은 둘 중 편한 것:

- **끌어다 놓기** · Colab 왼쪽 세로 막대의 **📁(파일)** 을 누르고 그 창에 끌어다 놓은 뒤 아래 셀 실행 (가장 확실)
- **파일 선택 버튼** · 그냥 아래 셀을 실행하면 [파일 선택] 버튼이 뜹니다

버튼이 안 보이는 브라우저(사파리 · 앱 속 브라우저 · 광고 차단 등)도 있습니다. 그때는 셀을 멈추고(■) 끌어다 놓은 뒤 다시 실행하세요.
**LINK-01 로봇 두뇌라면** 코스 파일(`…world.json`, 학습기 world.json 칸의 「코스 저장」)도 같이 끌어다 놓으세요. 모델 카드에 코스 지문이 들어가고, 그 코스에서 다시 걸어 본 성공률도 함께 적힙니다 (없어도 올라갑니다).

파일 없이 보기만 하려면 `USE_EXAMPLE`을 켜면 예시 모델을 씁니다."""),
    ("code", """USE_EXAMPLE = False                                   #@param {type:"boolean"}
UPLOAD_BUTTON = True                                  #@param {type:"boolean"}
import os
from connect_tycoon.colab import find_brain
if USE_EXAMPLE:
    ct.CompanyBrain.default().save("example_brain.json"); BRAIN = "example_brain.json"
else:
    BRAIN = find_brain(upload_button=UPLOAD_BUTTON)       # 📁에 있으면 그것, 없으면 [파일 선택] 버튼
from connect_tycoon.lab import LabBrain, is_lab, ALGO_NAME
from connect_tycoon.link01.brain import WorldBrain, is_world
WORLD = None
if is_world(BRAIN):                                     # LINK-01 로봇 두뇌 (CONNECT AI WORLD)
    from connect_tycoon.colab import find_course
    from connect_tycoon.link01 import sim
    b = WorldBrain.from_file(BRAIN)
    print("LINK-01 로봇 두뇌:", b.name, "·", sim.TASK_KO[b.task], "·", b.contract)
    print("  배운 코스", b.trained_on, "· 공식 시험", f"{b.test.get('score', 0):.0%}", "·", b.test.get("metric", ""))
    WORLD = find_course(os.path.dirname(os.path.abspath(BRAIN)))
    if WORLD:
        task, L, h = sim.load_course(WORLD)
        print("  코스 파일", os.path.basename(WORLD), "· 지문", h, "· 이 두뇌가 배운 코스" if h in b.trained_on else "· (이 두뇌가 배운 코스는 아님)")
elif is_lab(BRAIN):                                     # 웹 학습 실험실 두뇌
    b = LabBrain.from_file(BRAIN)
    print("학습 실험실 두뇌:", b.name, "·", ALGO_NAME.get(b.algo, b.algo), "· 시험 점수", round(b.test.get("score", 0)), "· 정시", f"{b.test.get('onTimeRate', 0):.0%}")
else:                                                   # 게임(CONNECT TYCOON) 두뇌
    b = ct.CompanyBrain.from_file(BRAIN)
    print("모델:", b.name, "· 회사의 가치", b.values, "· 기록된 성적", b.spec["company"].get("results"))""", None),
    ("md", """## 2 · 이름 · 나를 알리는 글 · 올리기

이름은 영문, 버전은 게임의 버전(v1, v2…)을 적습니다 (LINK-01도 v1부터 · 다시 올리면 v2, v3…). 나를 알리는 글은 형식 없이 자유롭게 (무엇을 하는 사람인지 · 사업 · 서비스 · 채널 · 연락처). 모델 카드 맨 위 '만든 사람'에 그대로 들어가고, 비워 두면 빠집니다.

✅ 정상이면 · 1분 안팎 (모델 카드에 넣을 성적을 시험 달로 계산합니다) · 마지막에 내 모델 주소가 나옵니다"""),
    ("code", """from google.colab import userdata
from connect_tycoon.hub import push_brain, push_lab_brain
from connect_tycoon.lab import is_lab
from connect_tycoon.link01.brain import is_world
from connect_tycoon.link01.hub import push_world_brain
from connect_tycoon.llm.hub import make_author

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
assert HF_TOKEN, "왼쪽 🔑에 HF_TOKEN(Write 권한)을 넣고 노트북 액세스를 켜 주세요"

NAME = "jay"                                         #@param {type:"string"}
VERSION = "v1"                                       #@param {type:"string"}

# 나를 알리는 글 · 따옴표 사이에 자유롭게
ABOUT_ME = '''
'''

if is_world(BRAIN):                                     # LINK-01 로봇 두뇌 → connectailab-rl-이름-버전-link01 (+ world.json)
    url = push_world_brain(BRAIN, version=VERSION, name=NAME, author=make_author(ABOUT_ME), token=HF_TOKEN, world=WORLD)
else:
    up = push_lab_brain if is_lab(BRAIN) else push_brain    # 실험실 두뇌 → connectailab-rl-이름-버전-logistics
    url = up(BRAIN, version=VERSION, name=NAME, author=make_author(ABOUT_ME), token=HF_TOKEN)
print("올라갔습니다 ·", url)""", None),
]


if __name__ == "__main__":
    B.CELLS = CELLS
    print(B.build(os.path.join(os.path.dirname(__file__), "..", "notebooks", "upload_brain_colab.ipynb")))
