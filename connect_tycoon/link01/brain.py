"""LINK-01 결정 두뇌 (connect-world-brain/1 · world link01-decision) · CONNECT AI WORLD 로봇 웹 학습기에서 내려받은 brain.json.

두뇌는 0.5초마다 방향(또는 물건) 하나만 고릅니다. 걷기는 고정된 걷기 신경망(CPG → MLP, PPO 아님)이 하고, 두뇌 파일에는 없습니다.

    b = WorldBrain.from_file("link01-avoid_obstacles-mine-brain.json")
    b.act(obs)            # 관찰(observation 순서 그대로) → 선택지 번호
    b.probs(obs)          # 선택지별 확률
"""
import json
import os

import numpy as np

FORMAT, WORLD = "connect-world-brain/1", "link01-decision"
MAX_BYTES, MAX_WIDTH, MAX_LAYERS = 3 * 1024 * 1024, 256, 4
# 과제별 관찰 수 (웹 tasks.ts) · 목표로 가기 · 경로 따라가기는 장애물 코스에서 배우면 거리 센서 11개가 붙습니다 (+11)
OBS = {"goto_target": (10, 21), "choose_object": (26,), "avoid_obstacles": (21,), "follow_route": (16, 27)}
N_ACT = {"goto_target": 5, "choose_object": 4, "avoid_obstacles": 5, "follow_route": 5}


def _load(path_or_dict):
    d = path_or_dict
    if isinstance(d, str):
        if os.path.getsize(d) > MAX_BYTES:
            raise ValueError("파일이 너무 큽니다 (3MB 까지)")
        with open(d, encoding="utf-8") as f:
            d = json.load(f)
    return d


def is_world(path_or_dict):
    try:
        d = _load(path_or_dict)
    except Exception:
        return False
    return isinstance(d, dict) and d.get("format") == FORMAT and d.get("world") == WORLD


class WorldBrain:
    def __init__(self, spec):
        if spec.get("format") != FORMAT:
            raise ValueError("CONNECT AI WORLD 두뇌(connect-world-brain/1)가 아닙니다")
        if spec.get("world") != WORLD:
            raise ValueError(f"LINK-01 두뇌가 아닙니다 (world {str(spec.get('world'))[:40]}) · 지금은 LINK-01 로봇 세상만 읽습니다")
        task = spec.get("task")
        if task not in OBS:
            raise ValueError(f"어느 과제의 두뇌인지 모르겠습니다 (task {task})")
        m = spec.get("model") or {}
        if m.get("type") != "mlp" or m.get("out", "logits") != "logits" or m.get("act", "tanh") not in ("tanh", "relu"):
            raise ValueError("model 이 신경망(mlp · tanh · logits)이 아닙니다")
        sizes = m.get("sizes") or []
        if not (2 <= len(sizes) <= MAX_LAYERS + 1) or any(not isinstance(n, int) or n < 1 for n in sizes) or max(sizes[1:-1] or [1]) > MAX_WIDTH:
            raise ValueError("신경망 크기(sizes)가 이상합니다")
        if sizes[0] not in OBS[task] or sizes[-1] != N_ACT[task]:
            raise ValueError(f"관찰 {sizes[0]} · 선택지 {sizes[-1]} 이 과제({task})와 맞지 않습니다")
        obs, acts = spec.get("observation") or [], spec.get("actions") or []
        if len(obs) != sizes[0] or len(acts) != sizes[-1]:
            raise ValueError("observation · actions 이름 수가 신경망 크기와 다릅니다 · 웹에서 내려받은 파일을 그대로 쓰세요")
        if len(m.get("W", [])) != len(sizes) - 1 or len(m.get("b", [])) != len(sizes) - 1:
            raise ValueError("W · b 의 층 수가 sizes 와 맞지 않습니다")
        # W[l] 은 입력 쪽 먼저 펼친 배열 (W[i * 출력 + j]) · 웹 nn.ts 와 같은 순서 · 계산은 float64
        self.W = [np.asarray(w, dtype=np.float64).reshape(sizes[i], sizes[i + 1]) for i, w in enumerate(m["W"])]
        self.b = [np.asarray(b, dtype=np.float64).reshape(sizes[i + 1]) for i, b in enumerate(m["b"])]
        if not all(np.isfinite(a).all() and np.abs(a).max() <= 1e6 for a in self.W + self.b):
            raise ValueError("두뇌 안에 숫자가 아닌 값(또는 너무 큰 값)이 있습니다")
        self.spec, self.task, self.sizes, self.act_fn = spec, task, sizes, m.get("act", "tanh")
        self.rays = sizes[0] == 21 if task == "goto_target" else sizes[0] == 27 if task == "follow_route" else task == "avoid_obstacles"

    @classmethod
    def from_file(cls, path):
        return cls(_load(path))

    name = property(lambda self: self.spec.get("name", "LINK-01 두뇌"))
    contract = property(lambda self: self.spec.get("contract", ""))
    observation = property(lambda self: self.spec.get("observation", []))
    actions = property(lambda self: self.spec.get("actions", []))
    trained_on = property(lambda self: self.spec.get("trained_on", []))
    test = property(lambda self: self.spec.get("test") or {})
    train = property(lambda self: self.spec.get("train") or {})

    def logits(self, obs):
        h = np.asarray(obs, dtype=np.float32).astype(np.float64)     # 웹 관찰은 Float32Array
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h) if self.act_fn == "tanh" else np.maximum(h, 0)
        return h

    def probs(self, obs):
        z = self.logits(obs)
        e = np.exp(z - z.max())
        return e / e.sum()

    def act(self, obs, world=None):
        return int(np.argmax(self.logits(obs)))


def example(task="avoid_obstacles"):
    """예시 두뇌 · 웹 학습기 e2e 시험에서 만든 「장애물 피하기」 두뇌와 그 코스(world.json). 내 brain.json이 없을 때 써 봅니다."""
    d = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "link01")
    return WorldBrain.from_file(os.path.join(d, f"{task}-brain.json")), os.path.join(d, f"{task}.world.json")
