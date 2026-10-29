"""LINK-01 의사결정 모델(또는 두뇌만)을 Hugging Face에 올립니다 · 이름 connectailab-rl-이름-버전-link01.

토큰은 Colab 🔑 HF_TOKEN 또는 환경변수 HF_TOKEN (코드에 적지 않습니다).
환경변수 CT_HF_DRY_RUN=폴더 를 주면 아무것도 올리지 않고, 올릴 파일을 그 폴더에 그대로 만듭니다 (미리 보기 · 시험용).
"""
import json
import os
import shutil

from ..llm.hub import BASE_MODEL, SYSTEM, YT, author_section, brand_header, repo_name
from . import sim
from .brain import WorldBrain

WORLD_URL = "https://www.ezerai.xyz/robot-lab/world/"
COLAB = "https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/link01_decision_model_colab.ipynb"


def _pct(x):
    return "—" if x is None else f"{x:.0%}"


def _curve(train):
    pts = train.get("curve") or []
    step = max(1, len(pts) // 8)
    rows = pts[::step] + ([pts[-1]] if pts and pts[-1] not in pts[::step] else [])
    return "\n".join(f"| {int(d):,} | {s:.0%} |" for d, s in rows)


def brain_card(spec, repo, author=None, course=None, check=None):
    """LINK-01 두뇌 모델 카드 · 과제 · 코스 지문 · trained_on · 파일에 적힌 시험 숫자 그대로 · (있으면) 2D 걷기로 다시 본 숫자"""
    b = WorldBrain(spec)
    t, tr = b.test, b.train
    W = tr.get("weights") or {}
    params = [("알고리즘", tr.get("algo") or spec.get("algo")), ("학습 갱신", tr.get("updates")), ("결정 수", f"{tr['decisions']:,}" if tr.get("decisions") else None),
              ("학습률", tr.get("lr")), ("감마 γ", tr.get("gamma")), ("GAE λ", tr.get("lam")), ("엔트로피", tr.get("ent")), ("클립", tr.get("clip")),
              ("병렬 판", tr.get("nEnvs")), ("숨은 층", tr.get("net")), ("센서", tr.get("sensors")), ("씨앗", tr.get("seed")),
              ("걸린 시간", f"{tr['sec']:.0f}초" if tr.get("sec") else None)]
    ptab = "\n".join(f"| {k} | {v} |" for k, v in params if v is not None)
    wtab = " · ".join(f"{k} {v:+g}" for k, v in W.items())
    courses = (tr.get("courses") or {}).get("list") or []
    on = ", ".join(f"`{h}`" for h in b.trained_on) or "—"
    curve = _curve(tr)
    chk = ""
    if check:
        chk = (f"\n이 저장소의 `world.json`(코스 `{check['course']}`)에서 2D 걷기 근사로 다시 걸어 본 20판 · 성공 {check['success']:.0%} · "
               f"부딪힌 판 {check['collide']:.0%} (넘어짐이 없는 근사라 웹 MuJoCo보다 조금 후할 수 있습니다)\n")
    return f"""---
license: apache-2.0
language: [ko]
library_name: numpy
pipeline_tag: reinforcement-learning
tags: [reinforcement-learning, deep-reinforcement-learning, ppo, robotics, mujoco, decision-making, link01, connect-ai-world, connect-ai-lab, connectailab]
---

{brand_header(COLAB)}

{author_section(author)}

# LINK-01 결정 두뇌 · {sim.TASK_KO[b.task]}

[CONNECT AI WORLD 로봇 웹 학습기]({WORLD_URL})에서 내가 코스를 만들고(장애물 · 목표 · 출발점), 브라우저 안에서 PPO로 강화학습한 **LINK-01 로봇의 결정 두뇌**입니다.
두뇌는 0.5초마다 **{" · ".join(b.actions)}** 중 하나를 고릅니다. 걷기는 따로 미리 배운 걷기 신경망(CPG → 모방 학습 MLP, 고정 · PPO 아님)이 하고, 이 파일에는 결정 두뇌만 들어 있습니다.

| | |
|---|---|
| 과제 | **{sim.TASK_KO[b.task]}** (`{b.task}`) |
| 약속(contract) | `{b.contract}` |
| 배운 코스 (trained_on) | {on}{" · " + ", ".join(c.get("name", "") for c in courses) if courses else ""} |
| 함께 올린 코스 | {f"`{course}` (world.json)" if course else "없음"} |
| 관찰 {len(b.observation)}개 | {" · ".join(b.observation)} |
| 신경망 | {" → ".join(map(str, b.sizes))} · {b.act_fn} |

## 시험 결과 (brain.json에 적힌 숫자 그대로)

| 시험 | 성공률 | 평균 보상 | 씨앗 | 지문 | 엔진 |
|---|---|---|---|---|---|
| {t.get("metric", "공식 시험")} | **{_pct(t.get("score"))}** | {t.get("ret", 0):.2f} | {min(t.get("seeds") or [0])}–{max(t.get("seeds") or [0])} | `{t.get("fingerprint", "—")}` | {t.get("engine", "—")} |

공식 시험 20판은 그 과제의 **무작위 코스**(씨앗 91001–91020)입니다. 내가 만든 코스 하나에서만 배운 두뇌는 무작위 코스에서 낮게 나올 수 있습니다.
{chk}
## 내가 정한 학습 설정

| 항목 | 값 |
|---|---|
{ptab}

보상 가중치 · {wtab or "기본값"}

## 학습 곡선 (검증 판 성공률)

| 결정 수 | 성공률 |
|---|---|
{curve}

## 쓰는 법

```python
from huggingface_hub import hf_hub_download
from connect_tycoon.link01.brain import WorldBrain     # pip install "connect-tycoon @ git+https://github.com/wonseokjung/connect-tycoon"
brain = WorldBrain.from_file(hf_hub_download("{repo}", "brain.json"))
brain.act(obs)        # 관찰 {len(b.observation)}개 (위 순서) → 선택지 번호
```

웹 학습기에서 **두뇌 올리기**로 이 brain.json을 다시 올리면 같은 시험을 그대로 다시 볼 수 있습니다. 의사결정 모델 만들기: [LINK-01 의사결정 모델 Colab]({COLAB})

## 만든 곳 · CONNECT AI LAB × AI CITY BUILDERS

- [CONNECT AI LAB 유튜브]({YT}) · AI 1인 기업 · 로컬 AI · 강화학습 무료 강의
- [AI CITY BUILDERS](https://www.aicitybuilders.com) · [강화학습 AI 과정](https://www.aicitybuilders.com/robot)
- [CONNECT AI WORLD]({WORLD_URL}) · LINK-01 로봇 웹 학습기

시뮬레이터 위의 교육용 모델입니다. 실제 로봇에 쓰기 전에 그 로봇 · 그 공간에서 다시 검증하세요.
"""


class _Dry:
    """CT_HF_DRY_RUN · 올릴 파일을 폴더에 만들기만 합니다"""

    def __init__(self, root):
        self.root = root

    def create_repo(self, repo, **kw):
        os.makedirs(os.path.join(self.root, repo), exist_ok=True)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kw):
        dst = os.path.join(self.root, repo_id, path_in_repo)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if isinstance(path_or_fileobj, (bytes, bytearray)):
            open(dst, "wb").write(path_or_fileobj)
        else:
            shutil.copyfile(path_or_fileobj, dst)

    def upload_folder(self, folder_path, path_in_repo, repo_id, allow_patterns=None, **kw):
        import fnmatch
        for f in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, f)) and (not allow_patterns or any(fnmatch.fnmatch(f, p) for p in allow_patterns)):
                self.upload_file(os.path.join(folder_path, f), os.path.join(path_in_repo, f) if path_in_repo not in ("", ".") else f, repo_id)


def _api(token):
    dry = os.environ.get("CT_HF_DRY_RUN")
    if dry:
        return _Dry(dry), "dry-run"
    from huggingface_hub import HfApi
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    return api, api.whoami()["name"]


def push_world_brain(path, repo=None, version="v1", name=None, author=None, token=None, world=None, private=False):
    """LINK-01 brain.json 올리기 · connectailab-rl-이름-버전-link01 · world.json을 주면 코스도 함께 (지문이 trained_on에 있는지 확인)"""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    b = WorldBrain(spec)
    course = check = None
    if world:
        task, L, h = sim.load_course(world)
        if task != b.task:
            raise ValueError(f"코스는 「{sim.TASK_KO[task]}」, 두뇌는 「{sim.TASK_KO[b.task]}」입니다 · 같은 과제의 world.json을 쓰세요")
        course = h
        check = {"course": h, **sim.evaluate(b.act, b.task, L, rays=b.rays)}
    api, user = _api(token)
    repo = repo or f"{user}/{repo_name(name or user, version or 'v1', 'link01')}"
    api.create_repo(repo, exist_ok=True, private=private)
    api.upload_file(path_or_fileobj=path, path_in_repo="brain.json", repo_id=repo)
    if world:
        api.upload_file(path_or_fileobj=world, path_in_repo="world.json", repo_id=repo)
    info = {"format": "connectailab/1", "repo": repo, "kind": "world-brain", "world": spec.get("world"), "task": b.task, "contract": b.contract,
            "trained_on": b.trained_on, "course": course, "algo": spec.get("algo"), "author": author, "test": b.test, "sim_check": check,
            "train": {k: b.train.get(k) for k in ("updates", "decisions", "seed", "weights", "sec")}}
    api.upload_file(path_or_fileobj=json.dumps(info, ensure_ascii=False, indent=1).encode("utf-8"), path_in_repo="connectailab.json", repo_id=repo)
    api.upload_file(path_or_fileobj=brain_card(spec, repo, author, course, check).encode("utf-8"), path_in_repo="README.md", repo_id=repo)
    return f"https://huggingface.co/{repo}" if user != "dry-run" else os.path.join(os.environ["CT_HF_DRY_RUN"], repo)


def decision_card(repo, results=None, meta=None, base=BASE_MODEL, author=None):
    """LINK-01 두뇌 → 의사결정 모델 카드 · 시험 문제 정답률(두뇌와 같은 답) · 같은 코스 20판 걷기"""
    res, m = results or {}, meta or {}
    sec = lambda v: "—" if v is None else f"{v:.1f}초"
    rows = [f"| {k} | {r.get('accuracy', 0):.1%} | {_pct(r.get('drive_success'))} | {_pct(r.get('drive_collide'))} | {sec(r.get('drive_time_s'))} |"
            for k, r in res.items()]
    table = "\n".join(["| 모델 | 시험 문제 정답률 (두뇌와 같은 답) | 같은 코스 20판 성공 | 부딪힌 판 | 성공한 판 평균 시간 |", "|---|---|---|---|---|"] + rows)
    t = m.get("teacher_test") or {}
    ceil = (m.get("text_ceiling") or {}).get("test")
    task = m.get("task", "")
    return f"""---
license: apache-2.0
base_model: {base}
library_name: peft
language: [ko]
pipeline_tag: reinforcement-learning
tags: [reinforcement-learning, deep-reinforcement-learning, ppo, robotics, decision-making, link01, connect-ai-world, connect-ai-lab, connectailab, lora]
---

{brand_header(COLAB)}

{author_section(author)}

# LINK-01 의사결정 모델 · {sim.TASK_KO.get(task, task)}

LINK-01 로봇의 지금 상황(목표 거리 · 방향 · 거리 센서 · 속도 · 직전 선택)을 글로 읽고, **다음 0.5초 동안 어떻게 걸을지** 보기 글자 하나로 답하는 의사결정 모델입니다.
[CONNECT AI WORLD 로봇 웹 학습기]({WORLD_URL})에서 사람이 코스를 만들고 PPO로 강화학습한 두뇌(brain.json)를 선생님으로 언어 모델을 학습했습니다 (LoRA).

## 만든 방법

1. **두뇌 키우기 (웹 학습기)** · 과제 `{task}` · 약속 `{m.get('contract', '')}` · 배운 코스 {", ".join(f"`{h}`" for h in (m.get('trained_on') or [])) or "—"} · 공식 시험 {_pct(t.get('score'))}
2. **문제집** · 두뇌가 코스(`{m.get('course', '')}`)를 걷는 동안 0.5초마다, 두뇌가 본 관찰을 글로 바꾼 문제 · 정답 = 두뇌의 답 (가끔 다른 보기로 걸어 길을 벗어난 상황도 넣음 · {m.get('explore', 0):.0%})
3. **지도학습** · 언어 모델이 내 두뇌의 답을 따라 배웁니다 (LoRA)
4. **강화학습** · 보기마다 '그렇게 하고 나머지는 두뇌가 걸었으면 받은 보상'으로, 두뇌의 답보다 확실히 나은 보기는 올리고 나쁜 보기는 내립니다 (KL로 붙잡아 둠)
5. **채점** · 의사결정 모델을 로봇의 결정 자리에 앉혀 같은 코스를 20판 걷습니다 (씨앗 91001–91020){f" · 올린 모델: {m['chosen']}" if m.get('chosen') else ""}

문제 수: {json.dumps(m.get('counts', {}), ensure_ascii=False)} · 글로 옮겨도 두뇌가 같은 답을 내는 비율(정답률 상한) {_pct(ceil)}

## 성적

{table}

걷기 채점은 {m.get('sim', '2D 걷기 근사')}로 했습니다. 로봇 모델 · 걷기 신경망 없이 실제 걷기 속도를 잰 표로 걷기 때문에 넘어짐이 없고, 웹 MuJoCo 결과와 조금 다를 수 있습니다.

## 쓰는 법

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("{base}")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{base}"), "{repo}")
```

지시문(system): `{SYSTEM}`

```python
from connect_tycoon.link01 import questions as Q
Q.task_of("{task}", obs)      # 로봇 관찰 → 위 형식의 문제 (상황 · 질문 · 보기)
```

## 만든 곳 · CONNECT AI LAB × AI CITY BUILDERS

- [CONNECT AI LAB 유튜브]({YT}) · AI 1인 기업 · 로컬 AI · 강화학습 무료 강의
- [AI CITY BUILDERS](https://www.aicitybuilders.com) · [강화학습 AI 과정](https://www.aicitybuilders.com/robot)
- [CONNECT TYCOON](https://github.com/wonseokjung/connect-tycoon) · 학습 코드

시뮬레이터 위의 연구 · 교육용 모델입니다. 실제 로봇에 쓰기 전에 그 로봇 · 그 공간에서 다시 검증하세요.
"""


def push_decision_model(adapter_dir, repo=None, name=None, version="v1", results=None, meta=None, token=None, private=False, author=None, brain=None, world=None):
    """의사결정 모델(LoRA 어댑터) 올리기 · connectailab-rl-이름-버전-link01 · brain(spec) · world(world.json 경로)를 주면 같은 곳에 함께"""
    api, user = _api(token)
    repo = repo or f"{user}/{repo_name(name or user, version or 'v1', 'link01')}"
    api.create_repo(repo, exist_ok=True, private=private)
    api.upload_folder(folder_path=adapter_dir, path_in_repo=".", repo_id=repo,
                      allow_patterns=["adapter_config.json", "adapter_model.safetensors", "*.json"])
    m = meta or {}
    info = {"format": "connectailab/1", "repo": repo, "base_model": BASE_MODEL, "backend": "torch", "adapter": ".", "author": author,
            "kind": "world-decision", "world": m.get("world"), "task": m.get("task"), "contract": m.get("contract"), "course": m.get("course"),
            "trained_on": m.get("trained_on"), "teacher": m.get("teacher"), "teacher_test": m.get("teacher_test"), "results": results}
    api.upload_file(path_or_fileobj=json.dumps(info, ensure_ascii=False, indent=1).encode("utf-8"), path_in_repo="connectailab.json", repo_id=repo)
    if brain:
        api.upload_file(path_or_fileobj=json.dumps(brain, ensure_ascii=False).encode("utf-8"), path_in_repo="brain.json", repo_id=repo)
    if world:
        api.upload_file(path_or_fileobj=world, path_in_repo="world.json", repo_id=repo)
    api.upload_file(path_or_fileobj=decision_card(repo, results, m, author=author).encode("utf-8"), path_in_repo="README.md", repo_id=repo)
    return f"https://huggingface.co/{repo}" if user != "dry-run" else os.path.join(os.environ["CT_HF_DRY_RUN"], repo)


push_qwen = push_decision_model      # 예전 이름
