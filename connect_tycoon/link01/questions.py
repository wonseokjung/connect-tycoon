"""② 문제집 (LINK-01) · 로봇 웹 학습기에서 키운 결정 두뇌(brain.json) → Qwen이 배울 문제집.

두뇌가 코스를 걷는 동안 0.5초마다, 두뇌가 보던 관찰(목표 거리 · 방향 · 거리 센서 · 속도 · 직전 선택)을 사람이 읽는 글로 바꾸고
보기 5개(직진 · 왼쪽으로 · 오른쪽으로 · 제자리 좌회전 · 제자리 우회전) 또는 물건 4개 중 하나를 고르는 문제로 만듭니다.

정답(지도학습) · 두뇌의 답. Qwen은 먼저 내 두뇌를 그대로 배웁니다.
점수(강화학습) · 보기마다 '그렇게 하고 나머지는 두뇌가 걸었으면 받은 보상'(2D 걷기 · 서로 다른 흔들림 몇 번의 평균 · 공식 보상).
장점 · 두뇌의 답보다 확실히(dead 넘게) 낫거나 나쁜 보기만 · 그 사이는 0 (물류 실험실과 같은 방식)

길을 조금씩 벗어난 상황도 배우도록, 문제집을 만들 때는 가끔(explore) 두뇌 대신 아무 보기나 골라 걷습니다. 정답은 그때도 두뇌의 답입니다.

    sft/train.jsonl   {"messages": [system, user(문제), assistant(정답 글자)]}     ← train_torch.sft 그대로
    rl/train.jsonl    {"task": 문제, "scores": [보기별 점수], "answer", "teacher", "dept": "link01", ...}
"""
import json
import math
import os

from ..llm.format import LETTERS, messages
from . import sim

GOAL = {"goto_target": "머그컵(목표)까지 걸어가기", "choose_object": "단서가 가리키는 물건으로 가기 · 다른 물건에 닿으면 실패",
        "avoid_obstacles": "장애물에 부딪히지 않고 화분(목표)까지", "follow_route": "경유점을 순서대로 지나 끝(랜턴)까지"}
QUESTION = {"nav": "LINK-01이 다음 0.5초 동안 어떻게 걸을까요?", "choose": "LINK-01이 지금 어느 물건 쪽으로 걸어갈까요?"}
NAV_OPTIONS = [("straight", "직진 · 앞으로 걷기"), ("veer_left", "왼쪽으로 · 걸으면서 왼쪽으로 꺾기"),
               ("veer_right", "오른쪽으로 · 걸으면서 오른쪽으로 꺾기"), ("turn_left", "제자리 좌회전 · 멈춰서 왼쪽으로 돌기"),
               ("turn_right", "제자리 우회전 · 멈춰서 오른쪽으로 돌기")]
TERM_KO = {"progress": "가까워진 1 m마다", "reach": "도착", "collision": "부딪힘", "time": "0.5초마다", "wrong": "다른 물건",
           "switch": "고른 물건 바꾸기", "offroute": "경로에서 벗어난 1 m마다", "gate": "경유점 하나 지날 때"}
PROGRESS_KO = {"follow_route": "경로를 따라 나아간 1 m마다", "choose_object": "단서 물건에 가까워질수록 (1 m마다)"}
SPLITS = {"train": range(10000, 10040), "valid": range(20000, 20004), "test": range(30000, 30006)}


def _deg(s, c):
    return math.degrees(math.atan2(s, c))


def _dir(deg):
    d = round(deg)
    return "정면" if abs(d) < 3 else f"왼쪽 {abs(d)}°" if d > 0 else f"오른쪽 {abs(d)}°"


def _dist(f, cap=4.0):
    return f"{cap:g} m 이상" if f >= 0.9995 else f"{f * cap:.2f} m"


def _motion(v, w):
    sp, turn = v * 0.2, math.degrees(w * 0.6)
    t = "회전 없음" if abs(turn) < 3 else f"{'왼쪽' if turn > 0 else '오른쪽'}으로 {abs(turn):.0f}°/초 회전 중"
    return f"지금 움직임 · 앞으로 {sp:.2f} m/초 · {t}"


def _rays(fs):
    parts = []
    for a, f in zip(range(-100, 101, 20), fs):
        side = "정면" if a == 0 else f"{'왼쪽' if a > 0 else '오른쪽'} {abs(a)}°"
        parts.append(f"{side} {'트임' if f <= 0.0005 else f'{(1 - f) * sim.RAY_RANGE:.2f} m'}")
    return "거리 센서 (1.2 m 안에 막힌 곳까지 거리) · " + " · ".join(parts)


def _last(onehot, names):
    k = [i for i, x in enumerate(onehot) if x > 0.5]
    return f"직전 선택 · {names[k[0]]}" if k else "직전 선택 없음 (첫 결정)"


def _wp(name, fb, fs):
    def ax(f, pos, neg):
        return f"{pos if f >= 0 else neg} {'2 m 이상' if abs(f) >= 0.9995 else f'{abs(f) * 2:.2f} m'}"
    return f"{name} · {ax(fb, '앞', '뒤')} · {ax(fs, '왼쪽', '오른쪽')}"


def split_obs(task, obs):
    """관찰 → 이름 붙은 조각 (웹 observe 순서)"""
    o = [float(x) for x in obs]
    if task == "goto_target":
        return {"goal": o[0:3], "motion": o[3:5], "last": o[5:10], "rays": o[10:21] if len(o) == 21 else None}
    if task == "avoid_obstacles":
        return {"goal": o[0:3], "rays": o[3:14], "motion": o[14:16], "last": o[16:21]}
    if task == "follow_route":
        return {"wp": o[0:6], "lat": o[6], "head": o[7:9], "motion": o[9:11], "last": o[11:16], "rays": o[16:27] if len(o) == 27 else None}
    return {"cue": o[0:4], "obj": [o[4 + 4 * i:8 + 4 * i] for i in range(4)], "last": o[20:24], "motion": o[24:26]}


def state_text(task, obs):
    """관찰(두뇌가 본 숫자 그대로) → 사람이 읽는 로봇 화면 글"""
    p = split_obs(task, obs)
    lines = [f"LINK-01 로봇 · {sim.TASK_KO[task]} · {GOAL[task]} · 0.5초마다 한 번 고릅니다 (왼쪽 + · 오른쪽 −)"]
    nav = [n for n, _ in sim.NAV]
    if task in ("goto_target", "avoid_obstacles"):
        g = p["goal"]
        lines.append(f"목표({'화분' if task == 'avoid_obstacles' else '머그컵'})까지 {_dist(g[0])} · {_dir(_deg(g[1], g[2]))}")
    if task == "follow_route":
        w = p["wp"]
        lines.append(" | ".join(_wp(n, w[2 * k], w[2 * k + 1]) for k, n in enumerate(["다음 경유점", "그다음", "그 다음 다음"])))
        lat = p["lat"] * 0.5
        lines.append(("경로 위" if abs(lat) < 0.005 else f"경로에서 {'왼쪽' if lat > 0 else '오른쪽'}으로 {'1 m 이상' if abs(p['lat']) >= 1.999 else f'{abs(lat):.2f} m'} 벗어남")
                     + f" · 경로 방향 {_dir(_deg(*p['head']))}")
    if task == "choose_object":
        cue = [sim.TARGETS[c] for c, x in zip(sim.TARGET_KEYS, p["cue"]) if x > 0.5]
        lines.append(f"단서 · {cue[0] if cue else '없음'}")
        objs = []
        for c, f in zip(sim.TARGET_KEYS, p["obj"]):
            objs.append(f"{sim.TARGETS[c]} {_dist(f[1])} · {_dir(_deg(f[2], f[3]))}" if f[0] > 0.5 else f"{sim.TARGETS[c]} 없음")
        lines.append("물건 · " + " | ".join(objs))
    if p.get("rays") is not None:
        lines.append(_rays(p["rays"]))
    lines.append(_motion(*p["motion"]))
    lines.append(_last(p["last"], [sim.TARGETS[c] for c in sim.TARGET_KEYS] if task == "choose_object" else nav))
    W, ko = sim.WEIGHTS[task], dict(TERM_KO, progress=PROGRESS_KO.get(task, TERM_KO["progress"]))
    lines.append("보상 · " + " · ".join(f"{ko[k]} {v:+g}".replace("-", "−") for k, v in W.items() if k in ko))
    return "\n".join(lines)


def task_of(task, obs):
    if task == "choose_object":
        have = split_obs(task, obs)["obj"]
        opts = [(c, f"{sim.TARGETS[c]} 쪽으로 가기" + ("" if h[0] > 0.5 else " (이 코스에 없음 · 제자리에서 둘러보기)"))
                for c, h in zip(sim.TARGET_KEYS, have)]
        q = QUESTION["choose"]
    else:
        opts, q = NAV_OPTIONS, QUESTION["nav"]
    return {"state": state_text(task, obs), "question": q,
            "options": [{"label": LETTERS[i], "key": k, "description": d} for i, (k, d) in enumerate(opts)]}


def text_obs(task, obs):
    """글에 적힌 정밀도로 반올림한 관찰 · 두뇌가 이 값으로도 같은 답을 내는 비율 = 글로 옮길 때 남는 정보 (정답률 상한)"""
    o = [float(x) for x in obs]

    def goal(i, cap=4.0):
        d = 1.0 if o[i] >= 0.9995 else round(o[i] * cap, 2) / cap
        a = math.radians(round(_deg(o[i + 1], o[i + 2])))
        a = 0.0 if abs(math.degrees(a)) < 3 else a
        o[i], o[i + 1], o[i + 2] = d, math.sin(a), math.cos(a)

    def rays(i):
        for k in range(i, i + 11):
            o[k] = 0.0 if o[k] <= 0.0005 else 1 - round((1 - o[k]) * 1.2, 2) / 1.2

    def motion(i):
        o[i] = round(o[i] * 0.2, 2) / 0.2
        t = round(math.degrees(o[i + 1] * 0.6))
        o[i + 1] = 0.0 if abs(t) < 3 else math.radians(t) / 0.6

    def head(i):
        a = round(_deg(o[i], o[i + 1]))
        a = 0 if abs(a) < 3 else a
        o[i], o[i + 1] = math.sin(math.radians(a)), math.cos(math.radians(a))

    if task in ("goto_target", "avoid_obstacles"):
        goal(0)
    if task == "goto_target":
        motion(3)
        if len(o) == 21:
            rays(10)
    if task == "avoid_obstacles":
        rays(3), motion(14)
    if task == "follow_route":
        for k in range(6):
            o[k] = math.copysign(1.0, o[k]) if abs(o[k]) >= 0.9995 else round(o[k] * 2, 2) / 2
        o[6] = math.copysign(2.0, o[6]) if abs(o[6]) >= 1.999 else (0.0 if abs(o[6] * 0.5) < 0.005 else round(o[6] * 0.5, 2) / 0.5)
        head(7), motion(9)
        if len(o) == 27:
            rays(16)
    if task == "choose_object":
        for i in range(4):
            if o[4 + 4 * i] > 0.5:
                goal(5 + 4 * i)
        motion(24)
    return o


# ── 문제집 만들기 ─────────────────────────────────────────────────────────────────
def scores_now(w, brain, futures=2, horizon=20, seed=0):
    """지금 상황에서 보기마다의 점수 · 그 보기를 하고 나머지 horizon번은 두뇌가 걸었을 때의 보상 합 (흔들림 futures개 평균).
    흔들림(미래)은 보기끼리 같게 맞춥니다 (공정한 비교)."""
    out = [0.0] * w.n_opt
    for f in range(futures):
        for a in range(w.n_opt):
            c = w.clone(seed=seed * 7919 + w.k * 101 + f + 1)
            total, done, k = 0.0, False, 0
            r, done, _ = c.step(a)
            total += r
            while not done and k < horizon:
                r, done, _ = c.step(brain.act(c.observe()))
                total += r
                k += 1
            out[a] += total / futures
    return out


def _jitter(L, r, d=0.15, yaw=0.35):
    """출발점을 조금 흔든 코스 (같은 코스 · 조금 다른 출발) · 장애물과 겹치면 원래 출발점"""
    import copy
    L2 = copy.deepcopy(L)
    sx, sy, sa = L["start"]
    L2["start"] = [sx + r.uniform(-d, d), sy + r.uniform(-d, d), sa + r.uniform(-yaw, yaw)]
    if sim.circle_hits(L2["start"][0], L2["start"][1], sim.ROBOT_R + 0.05, sim.shapes_of(L2)):
        L2["start"] = list(L["start"])
    return L2


def rows_for_episode(brain, L, seed, explore=0.2, futures=2, horizon=20, jitter=True):
    """두뇌가 코스 한 판을 걷는 동안 0.5초마다 문제 하나 (L=None이면 그 씨앗의 무작위 코스)"""
    task, r = brain.task, sim.Rand(seed * 31 + 7)
    w = sim.make_world(task, _jitter(L, r) if (L is not None and jitter) else L, seed, brain.rays)
    out, done = [], False
    while not done:
        obs = w.observe()
        teacher = brain.act(obs)
        scores = scores_now(w, brain, futures, horizon, seed) if futures else [0.0] * w.n_opt
        out.append({"task": task_of(task, obs), "scores": [round(s, 4) for s in scores], "answer": LETTERS[teacher], "label": "lab",
                    "teacher": LETTERS[teacher], "dept": "link01", "world": task, "seed": seed, "index": w.k,
                    "obs": [round(float(x), 6) for x in obs]})
        a = r.randint(w.n_opt) if r() < explore else teacher
        _, done, _ = w.step(a)
    return out


def build(brain, out_dir, course=None, splits=None, explore=0.2, futures=2, horizon=20, log=print):
    """문제집을 만듭니다. brain은 WorldBrain, course는 실행용 배치(sim.load_course의 L) 또는 None(무작위 코스). 반환: {split: 문제 수}"""
    from ..llm.labdata import balanced
    splits = splits or SPLITS
    os.makedirs(os.path.join(out_dir, "sft"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "rl"), exist_ok=True)
    counts, ceiling = {}, {}
    for name, seeds in splits.items():
        rows = []
        for i, s in enumerate(seeds):
            rows += rows_for_episode(brain, course, s, explore, futures, horizon)
            if log and ((i + 1) % 10 == 0 or i + 1 == len(seeds)):
                log(f"{name} {i + 1}/{len(seeds)}판 · 문제 {len(rows)}개")
        with open(os.path.join(out_dir, "sft", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in (balanced(rows) if name == "train" else rows):
                f.write(json.dumps({"messages": messages(r["task"], r["answer"])}, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "rl", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
        ceiling[name] = sum(brain.act(text_obs(brain.task, r["obs"])) == LETTERS.index(r["teacher"]) for r in rows) / max(1, len(rows))
    spec = brain.spec
    meta = {"kind": "link01", "world": "link01-decision", "task": brain.task, "contract": spec.get("contract"), "counts": counts,
            "text_ceiling": ceiling, "explore": explore, "futures": futures, "horizon": horizon,
            "course": sim.course_hash(brain.task, course) if course is not None else "random", "trained_on": spec.get("trained_on"),
            "teacher": spec.get("name"), "algo": spec.get("algo"), "teacher_test": spec.get("test"), "train": {k: (spec.get("train") or {}).get(k) for k in ("updates", "decisions", "seed", "weights")},
            "sim": "2D 걷기 근사 (웹 MuJoCo 걷기를 명령별로 잰 속도 표)", "splits": {k: [v.start, v.stop] for k, v in splits.items()}}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return counts


def advantages(rows, dead=1.0, clip=3.0):
    """강화학습 장점 · 두뇌의 답과 비교해 dead점 넘게 나으면 +, 나쁘면 − (그 사이와 두뇌의 답은 0)."""
    out = []
    for r in rows:
        s, k = r["scores"], LETTERS.index(r["teacher"])
        out.append([0.0 if abs(x - s[k]) <= dead else max(-clip, min(clip, (x - s[k]) / (2 * dead))) for x in s])
    return out


def as_policy(pick, task):
    """pick(문제) → 글자 · Qwen을 로봇의 결정 자리에 앉히는 정책 (sim.run_episode의 act)"""
    return lambda obs, world=None: LETTERS.index(pick(task_of(task, obs)))


def drive_report(pick, brain, course=None, seeds=sim.TEST_SEEDS, log=print):
    """Qwen이 같은 코스를 20판 걷기 (2D 걷기 근사 · 씨앗 = 웹 공식 시험과 같은 91001–91020)"""
    act = as_policy(pick, brain.task)
    eps = []
    for i, s in enumerate(seeds):
        eps.append(sim.run_episode(act, brain.task, course, s, brain.rays))
        if log and (i + 1) % 5 == 0:
            log(f"  {i + 1}/{len(seeds)}판 · 성공 {sum(e['success'] for e in eps)}")
    n, ok = len(eps), [e for e in eps if e["success"]]
    return {"drive_success": sum(e["success"] for e in eps) / n, "drive_collide": sum(e["collisions"] > 0 for e in eps) / n,
            "drive_wrong": sum(e["wrong"] for e in eps) / n, "drive_ret": sum(e["ret"] for e in eps) / n,
            "drive_time_s": sum(e["k"] for e in ok) * 0.5 / len(ok) if ok else None, "drive_runs": n}
