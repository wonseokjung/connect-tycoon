"""LINK-01 로봇 두뇌 (CONNECT AI WORLD) · 읽기 · 코스 지문 · 무작위 코스 · 문제집 · 올리기 (Qwen 없이 도는 부분)."""
import json
import os

import pytest

from connect_tycoon.link01 import questions as Q
from connect_tycoon.link01 import sim
from connect_tycoon.link01.brain import WorldBrain, example, is_world
from connect_tycoon.llm.format import LETTERS

TASKS = ["avoid_obstacles", "choose_object", "goto_target", "follow_route"]


@pytest.mark.parametrize("task", TASKS)
def test_world_brain_and_course_from_web(task):
    """웹 학습기에서 내려받은 brain.json · world.json 그대로 · 코스 지문이 웹과 같고 두뇌의 trained_on에 있음"""
    b, wj = example(task)
    assert is_world(b.spec) and b.task == task and b.contract.startswith(f"link01-decision/{task}/")
    t, L, h = sim.load_course(wj)
    assert t == task and h == json.load(open(wj))["hash"] and h in b.trained_on
    w = sim.make_world(task, L, 91001, b.rays)
    o = w.observe()
    assert len(o) == b.sizes[0] == len(b.observation) and 0 <= b.act(o) < len(b.actions)
    assert abs(sum(b.probs(o)) - 1) < 1e-9


def test_world_brain_rejects_other_files(tmp_path):
    b, _ = example("avoid_obstacles")
    for bad, msg in [({**b.spec, "world": "taxi"}, "LINK-01"), ({**b.spec, "format": "connect-lab-brain/1"}, "connect-world-brain"),
                     ({**b.spec, "observation": b.observation[:-1]}, "observation")]:
        with pytest.raises(ValueError, match=msg):
            WorldBrain(bad)
    assert not is_world({"format": "connect-lab-brain/1"}) and not is_world(str(tmp_path / "없음.json"))


def test_random_courses_match_web():
    """웹 rng.ts(mulberry32)와 같은 수열 · 씨앗 91001의 무작위 코스가 웹 공식 시험과 같음 (값은 Node에서 뽑은 것)"""
    r = sim.Rand(91001)
    assert [r(), r(), r()] == [0.4968885185662657, 0.7447812077589333, 0.10897741396911442]
    L = sim.random_layout("goto_target", sim.Rand(91001))
    assert L["start"] == [0.29373744931072, -0.4692271032370627, -0.019550014428001194] and L["goal"] == [-1.2312650837004186, -0.7572957172989846]
    L = sim.random_layout("avoid_obstacles", sim.Rand(91001))
    assert [o[0] for o in L["obst"]] == [3, 4, 2, 6, 7] and L["obst"][0][1] == -0.3718820770410822


def test_brain_walks_its_course_in_2d():
    """2D 걷기 근사에서도 두뇌가 자기 코스를 걷습니다 (웹 MuJoCo 이 코스 20판 100%)"""
    b, wj = example("avoid_obstacles")
    _, L, _ = sim.load_course(wj)
    assert sim.evaluate(b.act, b.task, L, seeds=sim.TEST_SEEDS[:5], rays=b.rays)["success"] >= 0.8


def test_questions_and_problem_set(tmp_path):
    from connect_tycoon.llm import data as D
    from connect_tycoon.llm.train_mlx import advantages
    b, wj = example("avoid_obstacles")
    _, L, h = sim.load_course(wj)
    counts = Q.build(b, str(tmp_path), L, splits={"train": range(10000, 10001), "test": range(30000, 30001)}, futures=1, horizon=5, log=None)
    rows = D.load(str(tmp_path), "train")
    meta = json.load(open(tmp_path / "meta.json"))
    assert counts["train"] == len(rows) > 20 and meta["course"] == h and meta["task"] == "avoid_obstacles"
    r = rows[0]
    assert [o["label"] for o in r["task"]["options"]] == list(LETTERS[:5]) and r["task"]["options"][0]["key"] == "straight"
    assert "목표(화분)까지" in r["task"]["state"] and "거리 센서" in r["task"]["state"] and r["dept"] == "link01"
    assert all(x["answer"] == x["teacher"] == LETTERS[b.act(x["obs"])] for x in rows)          # 정답 = 두뇌의 답
    assert meta["text_ceiling"]["train"] >= 0.9                                              # 글로 옮겨도 두뇌가 같은 답
    adv = advantages(rows)                                                                   # LINK-01 문제는 두뇌의 답 기준
    assert all(a[LETTERS.index(x["teacher"])] == 0 for a, x in zip(adv, rows))
    sft = [json.loads(line) for line in open(tmp_path / "sft" / "train.jsonl", encoding="utf-8")]
    assert sft[0]["messages"][2]["content"] in "ABCDE" and len(sft) >= len(rows)
    pol = Q.as_policy(lambda t: "A", b.task)                                                 # Qwen 자리 · 늘 직진
    assert pol(rows[0]["obs"]) == 0


@pytest.mark.parametrize("task", ["choose_object", "follow_route"])
def test_question_text_other_tasks(task):
    b, wj = example(task)
    _, L, _ = sim.load_course(wj)
    t = Q.task_of(task, sim.make_world(task, L, 91001, b.rays).observe())
    assert len(t["options"]) == len(b.actions)
    assert ("단서 · 랜턴" in t["state"]) if task == "choose_object" else ("다음 경유점" in t["state"] and "거리 센서" in t["state"])


def test_find_brain_accepts_link01(tmp_path):
    from connect_tycoon.colab import find_brain, find_course
    b, wj = example("goto_target")
    json.dump(b.spec, open(tmp_path / "link01-goto_target-mine-brain.json", "w"), ensure_ascii=False)
    assert find_course(str(tmp_path)) is None
    json.dump(json.load(open(wj)), open(tmp_path / "course.world.json", "w"), ensure_ascii=False)
    assert find_brain(str(tmp_path), upload_button=False).endswith("mine-brain.json")
    assert find_course(str(tmp_path)).endswith("course.world.json")


def test_push_world_brain_dry_run(tmp_path, monkeypatch):
    from connect_tycoon.link01.hub import push_decision_model, push_world_brain
    monkeypatch.setenv("CT_HF_DRY_RUN", str(tmp_path / "hf"))
    b, wj = example("avoid_obstacles")
    p = tmp_path / "brain.json"
    json.dump(b.spec, open(p, "w"), ensure_ascii=False)
    out = push_world_brain(str(p), name="Min Ji", version="v1", world=wj)
    assert out.endswith("connectailab-rl-min-ji-v1-link01") and sorted(os.listdir(out)) == ["README.md", "brain.json", "connectailab.json", "world.json"]
    card, info = open(os.path.join(out, "README.md"), encoding="utf-8").read(), json.load(open(os.path.join(out, "connectailab.json")))
    assert "pipeline_tag: reinforcement-learning" in card and b.contract in card and "**75%**" in card and "18b2956e" in card
    assert info["trained_on"] == ["18b2956e"] and info["test"]["score"] == b.test["score"] and info["sim_check"]["n"] == 20
    ad = tmp_path / "adapter"
    ad.mkdir()
    (ad / "adapter_config.json").write_text("{}")
    out = push_decision_model(str(ad), name="jay", version="v2", brain=b.spec, world=wj, results={"지도학습": {"accuracy": 0.9, "drive_success": 0.95, "drive_collide": 0.0, "drive_time_s": 27.5}},
                    meta={"task": "avoid_obstacles", "world": "link01-decision", "course": "18b2956e"})
    assert out.endswith("connectailab-rl-jay-v2-link01") and "brain.json" in os.listdir(out) and "world.json" in os.listdir(out) and "| 지도학습 | 90.0% | 95% | 0% | 27.5초 |" in open(os.path.join(out, "README.md"), encoding="utf-8").read()
