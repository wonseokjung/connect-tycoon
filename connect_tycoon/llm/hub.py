"""⑤ 공개 · 학습한 언어 모델 어댑터를 모델 카드와 함께 Hugging Face에 올립니다 (토큰은 환경변수 HF_TOKEN)."""
import json
import os

from .format import BASE_MODEL, SYSTEM

YT = "https://www.youtube.com/@CONNECT-AI-LAB"
RAW = "https://raw.githubusercontent.com/wonseokjung/connect-tycoon/main/"
COLAB = "https://colab.research.google.com/github/wonseokjung/connect-tycoon/blob/main/notebooks/rl_decision_model_qwen_colab.ipynb"


def _badge(label, msg, logo, url):
    from urllib.parse import quote
    img = (f"https://img.shields.io/badge/{quote(label)}-{quote(msg)}-d0f991?style=for-the-badge&logo={logo}"
           f"&logoColor=d0f991&labelColor=050807&color=d0f991")
    return f'<a href="{url}"><img src="{img}" alt="{label}"></a>'


def brand_header(colab=COLAB):
    """모델 카드 맨 위 · 움직이는 배너와 브랜드 배지 (그림은 GitHub에 있어 누가 올린 카드든 똑같이 보입니다)."""
    badges = " ".join([_badge("Colab", "직접 학습하기", "googlecolab", colab),
                       _badge("GitHub", "CONNECT TYCOON", "github", "https://github.com/wonseokjung/connect-tycoon"),
                       _badge("YouTube", "CONNECT AI LAB", "youtube", YT),
                       _badge("AI CITY BUILDERS", "강화학습 AI", "bookstack", "https://www.aicitybuilders.com/robot")])
    return (f'<p align="center"><img src="{RAW}docs/brand/banner.gif" alt="RL AI COMPANY DECISION MAKING MODEL" width="100%"></p>\n\n'
            f'<p align="center">{badges}</p>\n')


TESTING = "모델 연결(Connect AI) 실험은 테스트 중입니다."


def make_author(about=""):
    """자기소개 · 정해진 칸 없이 자유롭게 쓴 글 (마크다운). 모델 카드 맨 위 '만든 사람'과 connectailab.json에 들어갑니다."""
    text = "\n".join(line.rstrip() for line in str(about or "").strip().splitlines())[:3000]
    return {"about": text} if text else None


def author_section(author):
    """'만든 사람' 섹션 · 쓴 글을 그대로 보여 줍니다 (자기 사업 · 서비스 · 채널 홍보)."""
    if not author or not author.get("about"):
        return ""
    return f"## 만든 사람\n\n{author['about']}\n\n<sub>CONNECT AI LAB 공동 연구 참여 모델 · {TESTING}</sub>\n"


def simulator_gif():
    return (f'<p align="center"><img src="{RAW}docs/images/hero.gif" alt="모델이 회사를 운영하는 하루" width="88%"></p>\n'
            '<p align="center"><sub>시뮬레이터 속 회사 · 맨 위층의 모델이 판단하면 빛줄기가 그 층으로 내려가고, 결과가 돈 · 만족 · 사고로 돌아옵니다</sub></p>\n')


def card(repo, results=None, meta=None, backend="mlx", base=BASE_MODEL, results_image=None, author=None):
    res = results or {}
    rows = []
    for name, r in res.items():
        line = f"| {name} | {r.get('accuracy', 0):.1%} | {r.get('regret', 0):.2f} |"
        if "profit" in r: line += f" {r['profit']:.0f}만원 |"
        else: line += " — |"
        if "card" in r: line += f" {r['card']['saved']:.0%} ({r['card']['caught']}/{r['card']['fraud']}) |"
        else: line += " — |"
        rows.append(line)
    table = "\n".join(["| 모델 | 시험 문제 정답률 | 후회 (작을수록 좋음) | 시뮬레이터 한 달 이익 | 실제 카드 거래 · 손실 줄임 |",
                       "|---|---|---|---|---|"] + rows) if rows else "학습 뒤 `connect-tycoon llm eval` 결과가 여기에 들어갑니다."
    m = meta or {}
    load = (f'from huggingface_hub import snapshot_download\nfrom mlx_lm import load\n'
            f'model, tok = load("{base}", adapter_path=snapshot_download("{repo}") + "/adapter")   # Mac · MLX' if backend == "mlx" else
            f'from transformers import AutoModelForCausalLM, AutoTokenizer\nfrom peft import PeftModel\n'
            f'tok = AutoTokenizer.from_pretrained("{base}")\nmodel = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{base}"), "{repo}")')
    return f"""---
license: apache-2.0
base_model: {base}
library_name: {"mlx" if backend == "mlx" else "peft"}
language: [ko]
pipeline_tag: reinforcement-learning
tags: [reinforcement-learning, deep-reinforcement-learning, grpo, decision-making, gymnasium, connect-tycoon, connect-ai-lab, connectailab, lora, qwen3.5]
---

{brand_header()}

{author_section(author)}

# RL AI COMPANY DECISION MAKING MODEL

회사의 판단 하나(상황 · 질문 · 보기)를 읽고 **보기 글자 하나**로 답하는 의사결정 모델입니다. 보기 글자의 확률이 선택지별 확률, 가장 큰 값이 확신입니다.
[CONNECT TYCOON](https://github.com/wonseokjung/connect-tycoon) 시뮬레이터에서 만든 학습 데이터로 `{base}`에 LoRA를 학습했습니다.

{simulator_gif()}

## 만든 방법

1. **시뮬레이터 강화학습** · 작은 정책이 회사를 수천 달 운영하며 배웁니다 (PPO). 이 정책이 선생님입니다.
2. **문제집** · 선생님이 운영하는 달마다, 판단 하나의 선택지를 바꿔 그 달을 끝까지 다시 돌려 선택지별 점수를 매깁니다.
3. **지도학습** · 점수가 가장 높은 보기를 따라 배웁니다.
4. **강화학습** · 보기 전체를 한 묶음으로 보고, 시뮬레이터 점수가 평균보다 높은 보기의 확률을 올립니다 (묶음 표준화 장점 + 출발 모델과의 KL).

회사 분포: `{m.get('profile', 'general')}` (공개 데이터로 맞춘 일반 회사) · 문제 수: {json.dumps(m.get('counts', {}), ensure_ascii=False)}

## 성적

{f'<p align="center"><img src="{results_image}" alt="학습 단계별 성적" width="100%"></p>' if results_image else ''}

{table}

첫 실험 결과입니다. 회사 운영은 처음 보는 2달 평균이라 표본이 작고, 선생님 정책(시뮬레이터에서 PPO로 배운 작은 모델)보다는 아직 아래입니다.
{m.get("note", "")}

## 쓰는 법

```python
{load}
```

지시문(system): `{SYSTEM}`
문제(user): `{{"state": "...", "question": "...", "options": [{{"label": "A", "key": "...", "description": "..."}}]}}` · 답: 글자 하나

## 만든 곳 · CONNECT AI LAB × AI CITY BUILDERS

- [CONNECT AI LAB 유튜브]({YT}) · AI 1인 기업 · 로컬 AI · 강화학습 무료 강의
- [AI CITY BUILDERS](https://www.aicitybuilders.com) · AI 비즈니스의 정석 · [강화학습 AI 과정](https://www.aicitybuilders.com/robot)
- [Connect AI](https://www.ezerai.xyz) · 기업 의사결정 모델 C1 공동 연구 · 모델 연결은 테스트 중

합성 회사 시뮬레이터 위의 연구 · 교육용 모델입니다. 실제 경영 판단에 쓰기 전에 그 회사의 기록으로 다시 검증하세요.
"""


LAB_URL = "https://www.ezerai.xyz/models/rl"


def lab_card(repo, results=None, meta=None, backend="torch", base=BASE_MODEL, author=None):
    """물류 실험실 두뇌 → Qwen 모델 카드 · 같은 시험 24일(웹 실험실과 같은 날)에서 Qwen이 물류센터를 운영한 점수를 보여 줍니다."""
    res, m = results or {}, meta or {}
    rows = []
    for name, r in res.items():
        day = f"{r['day_score']:.1f}" if "day_score" in r else "—"
        ot = f"{r['on_time']:.0%}" if "on_time" in r else "—"
        rows.append(f"| {name} | {r.get('accuracy', 0):.1%} | {r.get('regret', 0):.2f} | {day} | {ot} |")
    if m.get("teacher_test") is not None:
        rows.append(f"| 두뇌 · 선생님 ({m.get('algo', '').upper()}) | — | — | {m['teacher_test']:.1f} | — |")
    if (m.get("baselines") or {}).get("rule") is not None:
        rows.append(f"| 규칙 사장 (자리가 있으면 먼저 온 트럭) | — | — | {m['baselines']['rule']:.1f} | — |")
    table = "\n".join(["| 모델 | 시험 문제 정답률 | 후회 (작을수록 좋음) | 시험 24일 하루 점수 | 제시간 출발 |", "|---|---|---|---|---|"] + rows)
    cfg = m.get("config") or {}
    knobs = " · ".join(f"{k} {cfg[k]}" for k in ("gamma", "lr", "episodes", "dream", "curiosity", "coin", "goal") if k in cfg)
    load = (f'from transformers import AutoModelForCausalLM, AutoTokenizer\nfrom peft import PeftModel\n'
            f'tok = AutoTokenizer.from_pretrained("{base}")\nmodel = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{base}"), "{repo}")'
            if backend != "mlx" else
            f'from huggingface_hub import snapshot_download\nfrom mlx_lm import load\n'
            f'model, tok = load("{base}", adapter_path=snapshot_download("{repo}") + "/adapter")   # Mac · MLX')
    return f"""---
license: apache-2.0
base_model: {base}
library_name: {"mlx" if backend == "mlx" else "peft"}
language: [ko]
pipeline_tag: reinforcement-learning
tags: [reinforcement-learning, deep-reinforcement-learning, {m.get('algo', 'ppo')}, decision-making, logistics, connect-ai-lab, connectailab, lora, qwen3.5]
---

{brand_header()}

{author_section(author)}

# RL AI DECISION MAKING MODEL · 물류센터

물류센터의 지금 상황(대기 트럭 · 빈 도크 · 팔레트 · 함정)을 읽고, **10분 동안 무엇을 할지** 보기 7개 중 글자 하나로 답하는 의사결정 모델입니다.
보기 글자의 확률이 선택지별 확률, 가장 큰 값이 확신입니다.

[CONNECT AI LAB 학습 실험실]({LAB_URL})에서 사람이 직접 알고리즘과 파라미터를 골라 강화학습한 두뇌(brain.json)를, `{base}`에 옮겨 담았습니다.

## 만든 방법

1. **두뇌 키우기 (웹 실험실)** · {m.get('algo', '').upper()} · {knobs}
   소나기 · 지게차 고장 · 결근 · VIP 긴급 · 도로 정체 · 시스템 다운 · 피크 폭주가 중간중간 터지는 어려운 하루 하나에서 배웠습니다.
2. **문제집** · 두뇌가 하루를 운영하는 동안 10분마다 보기 7개를 하나씩 해 보고, 그 하루를 끝까지 다시 돌려 점수를 매깁니다 (서로 다른 미래 {m.get('futures', 8)}개 평균 · 공식 보상).
3. **지도학습** · Qwen이 내 두뇌의 답을 따라 배웁니다 (LoRA).
4. **강화학습** · 두뇌의 답보다 확실히(20점 넘게) 나은 보기는 올리고 나쁜 보기는 내립니다 (지도학습 모델과의 KL로 붙잡아 둠).
5. **채점** · Qwen을 물류센터장 자리에 앉혀, 웹 실험실과 똑같은 시험 24일을 운영합니다. 점수가 더 높은 쪽을 올렸습니다{f" · 올린 모델: {m['chosen']}" if m.get('chosen') else ""}.

문제 수: {json.dumps(m.get('counts', {}), ensure_ascii=False)} · 환경 `{m.get('env', 'logistics-day/2')}`

## 성적

{table}

## 쓰는 법

```python
{load}
```

지시문(system): `{SYSTEM}`
문제(user): `{{"state": "물류센터 하루 · 오전 10:20 ...", "question": "지금부터 10분 동안 물류센터에서 무엇을 할까요?", "options": [{{"label": "A", ...}}]}}` · 답: 글자 하나

```python
from connect_tycoon.llm import labdata as L
L.task(obs)          # 시뮬레이터 관측 16개 → 위 형식의 문제
L.day_report(pick)   # pick(문제) → 글자 · 시험 24일 운영 점수
```

## 만든 곳 · CONNECT AI LAB × AI CITY BUILDERS

- [CONNECT AI LAB 유튜브]({YT}) · AI 1인 기업 · 로컬 AI · 강화학습 무료 강의
- [AI CITY BUILDERS](https://www.aicitybuilders.com) · AI 비즈니스의 정석 · [강화학습 AI 과정](https://www.aicitybuilders.com/robot)
- [CONNECT TYCOON](https://github.com/wonseokjung/connect-tycoon) · 시뮬레이터 · 학습 코드

합성 물류센터 시뮬레이터 위의 연구 · 교육용 모델입니다. 실제 운영에 쓰기 전에 그 센터의 기록으로 다시 검증하세요.
"""


def repo_name(name, version="v1", kind=None):
    """업로드 이름 규칙 · connectailab-rl-이름-버전 (예: connectailab-rl-jay-v1 · 선생님 정책은 connectailab-rl-jay-v1-teacher).
    rl = 강화학습 의사결정 모델 · 같은 connectailab이라도 다른 종류의 모델과 섞이지 않게 합니다."""
    import re
    clean = lambda s: re.sub(r"[^a-z0-9.]+", "-", str(s).lower()).strip("-.")
    n, v = clean(name), clean(version) or "v1"
    if not n:
        raise ValueError("이름은 영문 · 숫자로 적어 주세요 (예: jay, minji, team3)")
    return f"connectailab-rl-{n}-{v}" + (f"-{clean(kind)}" if kind else "")


def push(adapter_dir, repo=None, results=None, meta=None, backend="mlx", token=None, private=False, results_png=None, author=None):
    from huggingface_hub import HfApi
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    repo = repo or f"{api.whoami()['name']}/{repo_name(api.whoami()['name'])}"
    api.create_repo(repo, exist_ok=True, private=private)
    api.upload_folder(folder_path=adapter_dir, path_in_repo="adapter" if backend == "mlx" else ".", repo_id=repo,
                      allow_patterns=["adapters.safetensors", "adapter_config.json", "adapter_model.safetensors", "*.json"])
    img = None
    if results_png:                                           # 성적 그래프는 저장소 안에 (모델마다 다르므로)
        api.upload_file(path_or_fileobj=results_png, path_in_repo="assets/results.png", repo_id=repo); img = "assets/results.png"
    m = meta or {}
    lab = m.get("kind") == "lab"
    info = {"format": "connectailab/1", "repo": repo, "base_model": BASE_MODEL, "backend": backend,          # 모아서 비교할 때 읽는 정보
            "adapter": "adapter" if backend == "mlx" else ".", "author": author,
            "values": m.get("values"), "profile": m.get("profile"), "results": results}
    if lab:
        info.update(kind="lab-qwen", industry=m.get("industry"), env=m.get("env"), algo=m.get("algo"), teacher=m.get("teacher"),
                    teacher_test=m.get("teacher_test"), baselines=m.get("baselines"), config=m.get("config"))
    api.upload_file(path_or_fileobj=json.dumps(info, ensure_ascii=False, indent=1).encode("utf-8"), path_in_repo="connectailab.json", repo_id=repo)
    text = lab_card(repo, results, m, backend, author=author) if lab else card(repo, results, meta, backend, results_image=img, author=author)
    api.upload_file(path_or_fileobj=text.encode("utf-8"), path_in_repo="README.md", repo_id=repo)
    return f"https://huggingface.co/{repo}"
