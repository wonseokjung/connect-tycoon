"""언어 모델 단계 · 문제 형식 · 문제집 · 장점 · 모델 카드 (MLX · PyTorch 없이 도는 부분)."""
import json
import os

import connect_tycoon as ct
from connect_tycoon.company import DEPTS
from connect_tycoon.llm import data as D
from connect_tycoon.llm.format import LETTERS, SYSTEM, messages, task
from connect_tycoon.llm.hub import card
from connect_tycoon.llm.train_mlx import advantages


def _month():
    co = ct.make_company(profile="general")
    return co, D.rows_for_month(co, ct.RulePolicy(), 22000, stride=4)


def test_task_format_matches_c1():
    co, rows = _month()
    seen = set()
    for r in rows:
        t = r["task"]
        acts = DEPTS[r["dept"]]["actions"]
        assert [o["label"] for o in t["options"]] == list(LETTERS[:len(acts)])
        assert [o["key"] for o in t["options"]] == DEPTS[r["dept"]]["keys"]
        assert t["state"] and t["question"]
        if r["label"] == "simulator":                                         # 낮의 일: 시뮬레이터 점수 최고 (반올림 동점 허용)
            assert r["scores"][LETTERS.index(r["answer"])] == max(r["scores"])
        else:                                                                 # 아침 판단: 선생님의 답
            assert r["dept"] in ("staffing", "inventory", "marketing") and r["answer"] == r["teacher"]
        seen.add(r["dept"])
    assert {"cs", "refund", "security"} <= seen
    m = messages(rows[0]["task"], rows[0]["answer"])
    assert m[0]["content"] == SYSTEM and json.loads(m[1]["content"]) == rows[0]["task"] and m[2]["content"] in LETTERS


def test_refund_state_reads_like_the_card():
    co = ct.make_company(profile="general")
    ev = {"ctx": {"day": 2, "weekday": "수", "sat": 0.7, "human_seats_left": 3}, "reason": "미배송", "tier": "단골", "amount": 9.0,
          "fraud_score": 0.31, "billing_left": 0.5}
    s = task("refund", ev, co.values)["state"]
    assert "9.0만원" in s and "미배송" in s and "아직 안 왔어요" in s and "악용 의심 0.31" in s


def test_advantages_center_each_task():
    rows = [{"scores": [1.0, 2.0, 3.0]}, {"scores": [5.0, 5.1]}]
    adv = advantages(rows)
    assert all(abs(sum(a)) < 1e-9 for a in adv)
    assert abs(adv[0][2]) > abs(adv[1][1])                     # 차이가 큰 문제가 더 크게 반영됩니다


def test_build_writes_splits_and_meta(tmp_path):
    co = ct.make_company(profile="general")
    out = str(tmp_path / "llm")
    counts = D.build(co, ct.RulePolicy(), out, splits={"train": range(20000, 20001), "valid": range(21000, 21001), "test": range(22000, 22001)},
                     stride=6, log=None)
    assert all(counts[k] > 0 for k in ("train", "valid", "test"))
    meta = json.load(open(os.path.join(out, "meta.json"), encoding="utf-8"))
    assert meta["profile"] == "general" and meta["counts"] == counts
    first = json.loads(open(os.path.join(out, "sft", "train.jsonl"), encoding="utf-8").readline())
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant"]
    assert D.agreement(D.load(out, "test"), lambda t: "A")["n"] == counts["test"]


def test_model_card_has_results_and_brand():
    c = card("me/x", {"강화학습": {"accuracy": 0.61, "regret": 1.2, "profit": 900.0, "card": {"saved": 0.9, "caught": 160, "fraud": 177}}})
    assert "RL AI COMPANY DECISION MAKING MODEL" in c and "61.0%" in c and "CONNECT AI LAB" in c and "base_model: Qwen/Qwen3.5-0.8B" in c


def test_upload_name_rule():
    from connect_tycoon.llm.hub import repo_name
    assert repo_name("Jay", "v1") == "connectailab-rl-jay-v1"
    assert repo_name("Min Ji", "V2") == "connectailab-rl-min-ji-v2"
    assert repo_name("team3", "v1", "teacher") == "connectailab-rl-team3-v1-teacher"


def test_author_intro_in_card():
    from connect_tycoon.llm.hub import card, make_author
    about = "빵집을 운영하며 AI로 주문을 자동화하고 있어요.\n- 가게: https://example.com\n- 유튜브 @bakery"
    a = make_author(about)
    c = card("me/connectailab-rl-jay-v1", author=a)
    assert c.index("## 만든 사람") < c.index("# RL AI COMPANY DECISION MAKING MODEL")
    assert "빵집을 운영하며" in c and "https://example.com" in c and "테스트 중" in c
    assert "connectailab" in c.split("---")[1]
    assert make_author("   ") is None and "## 만든 사람" not in card("me/x")


def test_collect_scoring_and_connection():
    from connect_tycoon.llm import collect as C
    assert C.kind_of(["adapter/adapters.safetensors"]) == "mlx" and C.kind_of(["adapter_model.safetensors"]) == "peft"
    assert C.kind_of(["brain.json"]) == "brain" and C.kind_of(["README.md"]) == "other"
    rows = [{"scores": [1.0, 0.0, 0.5]}, {"scores": [0.0, 2.0, 1.0]}]
    a = [[0.8, 0.1, 0.1], [0.6, 0.3, 0.1]]           # 1번 맞음 · 2번 틀림
    b = [[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]]           # 1번 틀림 · 2번 맞음
    ma, mb = C.metrics(a, rows), C.metrics(b, rows)
    assert ma["accuracy"] == 0.5 and mb["accuracy"] == 0.5 and abs(ma["regret"] - 1.0) < 1e-9
    assert C.metrics(C.ensemble([a, b], "confident"), rows)["accuracy"] == 1.0    # 문제마다 더 확신하는 쪽
    assert len(C.ensemble([a, b], "mean")) == 2 and len(C.ensemble([a, b], "vote")[0]) == 3
    board = C.leaderboard({"a": {"probs": a, **ma}, "b": {"probs": b, **mb}}, rows)
    assert [x["type"] for x in board].count("ensemble") == 3 and board[0]["type"] == "ensemble" and board[0]["accuracy"] == 1.0   # 연결이 하나보다 낫다
    assert "| 1 |" in C.table(board)


def test_notebook_cells_are_valid_python():
    """Colab 셀이 문법 오류 없이 실행되도록 (! 명령 · % 매직 줄은 빼고 확인)."""
    import ast
    import glob
    import json
    import os
    for path in glob.glob(os.path.join(os.path.dirname(__file__), "..", "notebooks", "*.ipynb")):
        for cell in json.load(open(path, encoding="utf-8"))["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            ast.parse("\n".join(line[:len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith(("!", "%")) else line
                                 for line in src.splitlines()))


def test_lab_brain_from_web_simulator():
    """웹 학습 실험실 두뇌(connect-lab-brain/1) · 표 · 신경망 둘 다 행동을 고르고, 모델 카드에 시험 점수가 들어감."""
    from connect_tycoon.hub import lab_card
    from connect_tycoon.lab import LabBrain, is_lab
    base = {"format": "connect-lab-brain/1", "env": "logistics-day/1", "name": "jay", "algo": "q", "actions": ["대기", "도킹"], "observation": ["a", "b"],
            "test": {"score": 300, "onTimeRate": 0.9, "perDay": {"onTime": 20, "late": 2, "left": 0, "overtime": 3}}, "baselines": {"rule": 238, "random": -150},
            "config": {"gamma": 0.95, "reward": {"onTime": 12}}, "curve": [{"ep": 20, "real": 100, "valid": 120, "shaped": 100}]}
    q = LabBrain({**base, "model": {"type": "q-table", "table": {"k1": [0.1, 0.9]}}})
    assert is_lab(q.spec) and q.act(key="k1") == 1 and q.act(key="없음") == 1
    m = LabBrain({**base, "algo": "dqn", "model": {"type": "mlp", "sizes": [2, 2], "W": [[1, 0, 0, 1]], "b": [[0, 0]], "head": "q"}})
    assert m.act(obs=[0.2, 0.8]) == 1 and m.act(obs=[0.9, 0.1]) == 0
    card = lab_card(base, "me/connectailab-rl-jay-v1-logistics")
    assert "pipeline_tag: reinforcement-learning" in card and "**300**" in card and "| 규칙 사장" in card and "238" in card


def test_python_lab_env_matches_web():
    """파이썬 물류센터 하루가 웹(TypeScript)과 걸음마다 같음 · 예시 두뇌의 웹 시험 점수 · 규칙 사장 점수가 그대로 나옴."""
    from connect_tycoon import labenv as E
    from connect_tycoon.lab import example
    b = example()
    assert b.spec["env"] == E.ENV_ID
    assert abs(E.evaluate(b.policy())["score"] - b.test["score"]) < 1e-9
    assert abs(E.evaluate(E.rule_policy)["score"] - b.spec["baselines"]["rule"]) < 1e-9


def test_lab_brain_to_qwen_problems(tmp_path):
    """물류 두뇌 → Qwen 문제집 · 보기 7개 · 점수 · 정답(두뇌 + 첨삭) · 기존 지도학습 형식 그대로 · Qwen을 센터장으로 앉히는 정책."""
    from connect_tycoon import labenv as E
    from connect_tycoon.lab import example
    from connect_tycoon.llm import data as D
    from connect_tycoon.llm import labdata as L
    from connect_tycoon.llm.hub import lab_card, repo_name
    b = example()
    counts = L.build(b, str(tmp_path), splits={"train": range(30000, 30001), "test": range(32000, 32001)}, futures=1, log=None)
    rows = D.load(str(tmp_path), "train")
    assert counts["train"] == len(rows) > 30
    r = rows[0]
    assert len(r["task"]["options"]) == len(r["scores"]) == 7 and r["label"] == "lab" and r["dept"] == "logistics"
    assert "물류센터" in r["task"]["state"] and r["answer"] in "ABCDEFG"
    assert all(r["answer"] == r["teacher"] for r in rows)                                                       # 지도학습 정답 = 두뇌의 답
    fake = [{"answer": "C"}] * 80 + [{"answer": "D"}] * 20
    assert sum(r["answer"] == "D" for r in L.balanced(fake)) == 40                                             # 드문 답(초과근무)은 √4 = 2번
    assert L.label([0, 5, 30, 0, 0, 0, 0], chosen=1, margin=10) == 2 and L.label([0, 5, 12, 0, 0, 0, 0], chosen=1, margin=10) == 1
    task = rows[0]["task"]
    adv = L.advantages([{"scores": [0, 5, 30, 0, 0, -40, 99], "teacher": "B", "temp": 3, "task": task}])[0]
    assert adv[1] == 0 and adv[0] == 0 and adv[2] > 0 and adv[5] < 0 and adv[6] == 0                                          # 20점 안쪽은 0 · 임시 인력 근무 중 재호출은 올리지 않음
    from connect_tycoon.llm.train_mlx import advantages
    assert advantages([{"scores": [0, 50, 0, 0, 0, 0, 0], "teacher": "A", "dept": "logistics", "task": task}])[0][1] > 0
    line = json.loads(open(tmp_path / "sft" / "train.jsonl", encoding="utf-8").readline())
    assert line["messages"][2]["content"] in "ABCDEFG"
    assert D.agreement(rows, lambda t: "B")["n"] == len(rows)
    total, acts, _ = E.run_day(L.as_policy(lambda t: "B"), 90001)
    assert set(acts) == {1}
    meta = json.load(open(tmp_path / "meta.json", encoding="utf-8"))
    assert meta["kind"] == "lab" and meta["teacher_test"] == b.test["score"]
    md = lab_card("me/x", {"지도학습": {"accuracy": 0.8, "regret": 1.0, "day_score": 300.0, "on_time": 0.8}}, meta)
    assert "시험 24일 하루 점수" in md and "300.0" in md and "규칙 사장" in md
    assert repo_name("Jay", "v1", "logistics-qwen") == "connectailab-rl-jay-v1-logistics-qwen"


def test_security_state_treats_probability_signals_as_off():
    """실제 카드 백테스트는 new_device·overseas를 평균 확률(예: 0.32 · 0.2)로 채웁니다 → 글로는 '평소 기기 · 국내'여야 합니다
    (참/거짓으로 읽으면 713건 전부 '새 기기 · 해외'가 되어 글 읽는 모델이 전부 차단합니다). 시뮬레이터의 1.0/0.0은 그대로."""
    ctx = {"day": 0, "weekday": "월", "sat": 0.8, "human_seats_left": 2}
    base = {"ctx": ctx, "amount": 22.0, "fraud_score": 0.05, "velocity": 0.36}
    s = task("security", {**base, "new_device": 0.32, "overseas": 0.2})["state"]
    assert "평소 기기 · 국내" in s and "새 기기" not in s and "해외" not in s
    s = task("security", {**base, "new_device": 1.0, "overseas": 0.0})["state"]
    assert "새 기기" in s and "해외" not in s
    s = task("security", {**base, "new_device": 1.0, "overseas": 1.0})["state"]
    assert "새 기기 · 해외" in s
