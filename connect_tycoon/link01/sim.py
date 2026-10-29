"""LINK-01 결정 세상 (2D) · CONNECT AI WORLD 로봇 웹 학습기(ezerai.xyz/robot-lab/world)와 같은 과제 · 같은 관찰 · 같은 보상.

웹에서는 0.5초마다 두뇌가 고른 선택지가 속도 명령 [앞 속도, 회전]이 되어 「미리 배운 걷기」 신경망 → MuJoCo 물리로 걷습니다.
여기서는 로봇 모델 · 걷기 신경망 없이, 그 걷기를 실제로 재서 맞춘 **2D 걷기 근사**(명령별 실제 속도 표 + 반응 지연 + 흔들림)로 걷습니다.
그래서 같은 코스 · 같은 두뇌라도 성공률이 웹(MuJoCo)과 조금 다를 수 있습니다. 넘어짐은 없습니다.

    w = World(); L = course_layout(course)          # 또는 random_layout("avoid_obstacles", seed)
    run_episode(brain.act, "avoid_obstacles", L, seed=91001)   # → {"success", "collisions", "ret", "k", ...}
"""
import copy
import json
import math

TAU = 2 * math.pi
ARENA_HALF = 2.0
ROBOT_R = 0.11
RAY_ANGLES = [math.radians(-100 + 20 * i) for i in range(11)]
RAY_RANGE = 1.2
DT, N_SUB = 0.02, 25                                        # 50 Hz 걷기 · 결정 1번 = 0.5초
TEST_SEEDS = list(range(91001, 91021))                      # 웹 공식 시험 20판과 같은 씨앗

NAV = [("직진", (0.20, 0.0)), ("왼쪽으로", (0.14, 0.45)), ("오른쪽으로", (0.14, -0.45)),
       ("제자리 좌회전", (0.0, 0.6)), ("제자리 우회전", (0.0, -0.6))]
TARGETS = {"mug": "머그컵", "apple": "사과", "lantern": "랜턴", "plant": "화분"}
TARGET_KEYS = list(TARGETS)
OBSTACLES = {"crate_long": ("긴 나무상자", "box", (0.265, 0.583)), "barrel": ("드럼통", "cyl", (0.318,)),
             "box": ("종이상자", "box", (0.194, 0.258)), "crate": ("나무상자", "box", (0.413, 0.205)),
             "cabinet": ("배전함", "box", (0.26, 0.216))}
OBST_SLOTS = ["barrel", "barrel", "box", "box", "crate", "cabinet", "crate_long", "cabinet"]           # 무작위 코스
EDIT_SLOTS = OBST_SLOTS + ["barrel", "box", "crate", "cabinet", "crate_long", "box", "barrel", "crate"]  # 편집기 16칸
TASK_KO = {"goto_target": "목표로 가기", "choose_object": "물건 중 고르기", "avoid_obstacles": "장애물 피하기", "follow_route": "경로 따라가기"}
WEIGHTS = {"goto_target": {"progress": 5.0, "reach": 10.0, "time": -0.05, "fall": -10.0, "out": -5.0, "collision": -1.0},
           "choose_object": {"progress": 3.0, "reach": 10.0, "wrong": -10.0, "time": -0.05, "switch": -0.1, "fall": -10.0, "out": -5.0},
           "avoid_obstacles": {"progress": 5.0, "reach": 10.0, "collision": -2.0, "time": -0.05, "fall": -10.0, "out": -5.0},
           "follow_route": {"progress": 5.0, "offroute": -2.0, "gate": 1.0, "reach": 10.0, "time": -0.05, "fall": -10.0, "out": -5.0}}
MAX_DECISIONS = {"goto_target": 60, "choose_object": 60, "avoid_obstacles": 90, "follow_route": 110}

# 2D 걷기 근사 · 웹 학습기의 걷기(MuJoCo 3.16 · 걷기 신경망)를 명령마다 6판씩 고정해 걸려 잰 값 (명령 → 실제 앞 속도 v · 회전 w)
WALK = {"table": [((0.0, 0.0), 0.0333, 0.0764), ((0.2, 0.0), 0.1988, -0.0219), ((0.14, 0.45), 0.1443, 0.4906),
                  ((0.14, -0.45), 0.1437, -0.4875), ((0.0, 0.6), -0.0017, 0.5512), ((0.0, -0.6), -0.0106, -0.5001)],
        "tau_v": 0.15, "tau_w": 0.12, "noise_v": 0.0205, "noise_w": 0.45, "drift_w": 0.03, "vy_noise": 0.01}


# ── 난수 · 웹(rng.ts, mulberry32)과 같은 수열 → 같은 씨앗 = 같은 무작위 코스 ─────────────────────
class Rand:
    def __init__(self, seed):
        self.a = int(seed) & 0xFFFFFFFF

    def __call__(self):
        M = 0xFFFFFFFF
        self.a = (self.a + 0x6D2B79F5) & M
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & M
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M)) & M
        return ((t ^ (t >> 14)) & M) / 4294967296

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self()

    def randint(self, n):
        return int(math.floor(self() * n))

    def subseed(self):
        return int(math.floor(self() * 1073741824))

    def shuffle(self, a):
        for i in range(len(a) - 1, 0, -1):
            j = int(math.floor(self() * (i + 1)))
            a[i], a[j] = a[j], a[i]
        return a

    def gauss(self):
        u, v = max(self(), 1e-300), self()
        return math.sqrt(-2 * math.log(u)) * math.cos(TAU * v)


def wrap(a):
    r = (a + math.pi) % TAU
    return r - math.pi


# ── 평면 도형 · 레이 · 충돌 (geom2d) ─────────────────────────────────────────────────────
def shapes_of(L):
    out = []
    for o in L.get("obst", []):
        slot, x, y, yaw = o[:4]
        s = o[4] if len(o) > 4 else 1.0
        _, kind, half = OBSTACLES[EDIT_SLOTS[slot]]
        out.append(("circle", x, y, half[0] * s) if kind == "cyl" else ("box", x, y, half[0] * s, half[1] * s, yaw))
    return out


def _ray_circle(ox, oy, dx, dy, cx, cy, r):
    px, py = ox - cx, oy - cy
    b, q = px * dx + py * dy, px * px + py * py - r * r
    disc = b * b - q
    if disc < 0:
        return math.inf
    s = math.sqrt(disc)
    t = -b - s
    if t < 0:
        t = -b + s
    return t if t >= 0 else math.inf


def _ray_box(ox, oy, dx, dy, cx, cy, hx, hy, yaw):
    ca, sa = math.cos(yaw), math.sin(yaw)
    rx, ry = ox - cx, oy - cy
    lo = (ca * rx + sa * ry, -sa * rx + ca * ry)
    ld = (ca * dx + sa * dy, -sa * dx + ca * dy)
    tmin, tmax = -math.inf, math.inf
    for i, h in enumerate((hx, hy)):
        if abs(ld[i]) < 1e-9:
            if abs(lo[i]) > h:
                return math.inf
        else:
            t1, t2 = (-h - lo[i]) / ld[i], (h - lo[i]) / ld[i]
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
    if tmax < max(tmin, 0):
        return math.inf
    return tmin if tmin >= 0 else tmax


def _ray_bounds(ox, oy, dx, dy, half):
    t = math.inf
    for o, d in ((ox, dx), (oy, dy)):
        if abs(d) > 1e-9:
            for s in (-half, half):
                ti = (s - o) / d
                if ti > 0:
                    t = min(t, ti)
    return t


def cast(x, y, yaw, shapes):
    out = []
    for a in RAY_ANGLES:
        dx, dy = math.cos(yaw + a), math.sin(yaw + a)
        t = _ray_bounds(x, y, dx, dy, ARENA_HALF + 0.3)
        for s in shapes:
            t = min(t, _ray_circle(x, y, dx, dy, *s[1:]) if s[0] == "circle" else _ray_box(x, y, dx, dy, *s[1:]))
        out.append(min(t, RAY_RANGE))
    return out


def circle_hits(px, py, r, shapes):
    for s in shapes:
        if s[0] == "circle":
            if math.hypot(px - s[1], py - s[2]) < r + s[3]:
                return True
        else:
            _, cx, cy, hx, hy, yaw = s
            ca, sa = math.cos(yaw), math.sin(yaw)
            rx, ry = px - cx, py - cy
            lx, ly = ca * rx + sa * ry, -sa * rx + ca * ry
            qx, qy = min(max(lx, -hx), hx), min(max(ly, -hy), hy)
            if math.hypot(lx - qx, ly - qy) < r:
                return True
    return False


# ── 코스 ────────────────────────────────────────────────────────────────────────
def with_path(wp, yaw):
    """경유점 → 촘촘한 선 · 누적 길이 (웹 FollowRoute.withPath)"""
    dense = []
    for i in range(len(wp) - 1):
        for k in range(20):
            dense.append((wp[i][0] + k * (wp[i + 1][0] - wp[i][0]) / 20, wp[i][1] + k * (wp[i + 1][1] - wp[i][1]) / 20))
    dense.append(tuple(wp[-1]))
    s = [0.0]
    for i in range(1, len(dense)):
        s.append(s[-1] + math.hypot(dense[i][0] - dense[i - 1][0], dense[i][1] - dense[i - 1][1]))
    last = wp[-1]
    return {"start": [wp[0][0], wp[0][1], yaw], "waypoints": [list(p) for p in wp], "path": dense, "s": s, "obst": [],
            "targets": [["lantern", last[0], last[1]]]}


def random_layout(task, r, n_objects=3, n_obstacles=5, n_waypoints=6):
    """웹 과제의 무작위 코스 (tasks.ts layout) · 같은 Rand 수열 → 같은 코스"""
    if task == "goto_target":
        yaw, sx, sy = r.uniform(-math.pi, math.pi), r.uniform(-0.6, 0.6), r.uniform(-0.6, 0.6)
        gx = gy = 0.0
        for _ in range(100):
            gx, gy = r.uniform(-1.6, 1.6), r.uniform(-1.6, 1.6)
            d = math.hypot(gx - sx, gy - sy)
            if 1.0 < d < 2.4:
                break
        return {"start": [sx, sy, yaw], "goal": [gx, gy], "obst": [], "targets": [["mug", gx, gy]]}
    if task == "choose_object":
        cats = r.shuffle(TARGET_KEYS[:])[:n_objects]
        cue = r.randint(n_objects)
        sx, sy, yaw = -1.3, r.uniform(-0.4, 0.4), r.uniform(-0.5, 0.5)
        while True:
            angs = sorted(r.uniform(-1.0, 1.0) for _ in range(n_objects))
            if min(angs[i] - angs[i - 1] for i in range(1, n_objects)) > 0.45:
                break
        tg = []
        for c, a in zip(cats, angs):
            rr = r.uniform(1.3, 1.9)
            tg.append([c, sx + rr * math.cos(a), sy + rr * math.sin(a)])
        return {"start": [sx, sy, yaw], "targets": tg, "cue": tg[cue][0], "cue_idx": cue, "obst": []}
    if task == "avoid_obstacles":
        sy, gy, sx, gx = r.uniform(-0.8, 0.8), r.uniform(-0.8, 0.8), -1.7, 1.7
        slots = r.shuffle(list(range(len(OBST_SLOTS))))
        placed = []
        for k, s in enumerate(slots[:n_obstacles]):
            for _ in range(200):
                if k == 0:
                    u = r.uniform(0.35, 0.65)
                    x, y = sx + u * (gx - sx), sy + u * (gy - sy) + r.uniform(-0.1, 0.1)
                else:
                    x, y = r.uniform(-1.1, 1.1), r.uniform(-1.6, 1.6)
                yaw = (math.pi / 2 if r.randint(2) else 0) + r.uniform(-0.3, 0.3)
                rad = max(OBSTACLES[OBST_SLOTS[s]][2][:2])
                ok = math.hypot(x - sx, y - sy) > rad + 0.45 and math.hypot(x - gx, y - gy) > rad + 0.45
                ok = ok and all(math.hypot(x - px, y - py) > rad + pr + 0.32 for _, px, py, _, pr in placed)
                if ok:
                    placed.append((s, x, y, yaw, rad))
                    break
        yaw0 = math.atan2(gy - sy, gx - sx) + r.uniform(-0.3, 0.3)
        return {"start": [sx, sy, yaw0], "goal": [gx, gy], "obst": [[s, x, y, yaw] for s, x, y, yaw, _ in placed],
                "targets": [["plant", gx, gy]]}
    if task == "follow_route":
        n = n_waypoints
        xs = [-1.6 + 3.2 * i / (n - 1) for i in range(n)]
        amp, ph, fr = r.uniform(0.5, 1.1), r.uniform(0, TAU), r.uniform(0.8, 1.6)
        ys = [min(max(amp * math.sin(fr * x + ph) + r.uniform(-0.15, 0.15), -1.6), 1.6) for x in xs]
        if r() < 0.5:
            xs = xs[::-1]
        wp = [[x, y] for x, y in zip(xs, ys)]
        return with_path(wp, math.atan2(wp[1][1] - wp[0][1], wp[1][0] - wp[0][0]) + r.uniform(-0.5, 0.5))
    raise ValueError(task)


def _r4(v):
    return math.floor(v * 1e4 + 0.5) / 1e4                    # JS Math.round


def _js(v):
    """JSON.stringify와 같은 글자 (지문 계산용)"""
    if isinstance(v, bool) or v is None:
        return json.dumps(v)
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    return "[" + ",".join(_js(x) for x in v) + "]"


def course_hash(task, L):
    """웹 courseHash와 같은 8자리 지문 · brain.json의 trained_on과 비교합니다"""
    start = [_r4(L["start"][0]), _r4(L["start"][1]), _r4(L["start"][2])]
    obst = []
    for o in L.get("obst", []):
        sc = o[4] if len(o) > 4 else 1
        s2 = math.floor(sc * 100 + 0.5) / 100
        obst.append([o[0], _r4(o[1]), _r4(o[2]), _r4(o[3])] + ([] if s2 == 1 else [s2]))
    targets = [[t[0], _r4(t[1]), _r4(t[2])] for t in L["targets"]]
    wp = [[_r4(p[0]), _r4(p[1])] for p in L["waypoints"]] if task == "follow_route" else None
    if wp:
        targets = [["lantern", wp[-1][0], wp[-1][1]]]
    txt = _js([task, start, obst, targets, L.get("cue") if task == "choose_object" else None, wp])
    h = 0x811C9DC5
    for b in txt.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def load_course(path_or_dict):
    """world.json (connect-world-course/1) → (과제, 실행용 배치, 지문)"""
    c = path_or_dict
    if isinstance(c, str):
        with open(c, encoding="utf-8") as f:
            c = json.load(f)
    if not isinstance(c, dict) or c.get("format") != "connect-world-course/1" or c.get("world") != "link01-decision":
        raise ValueError("LINK-01 코스 파일(world.json · connect-world-course/1)이 아닙니다")
    task, L = c["task"], copy.deepcopy(c["layout"])
    if task in ("goto_target", "avoid_obstacles"):
        L["goal"] = [L["targets"][0][1], L["targets"][0][2]]
    elif task == "choose_object":
        L["cue_idx"] = [t[0] for t in L["targets"]].index(L["cue"])
    elif task == "follow_route":
        P = with_path(L["waypoints"], L["start"][2])
        P.update(start=L["start"], obst=L.get("obst", []))
        L = P
    h = course_hash(task, L)
    if c.get("hash") and c["hash"] != h:
        raise ValueError(f"코스 지문이 내용과 다릅니다 (파일 {c['hash']} · 계산 {h}) · 웹에서 내려받은 world.json을 그대로 쓰세요")
    return task, L, h


# ── 세상 · 과제 ──────────────────────────────────────────────────────────────────
class World:
    def __init__(self, task, L, rays=False, seed=0):
        self.task, self.L, self.rays_on = task, L, rays or task == "avoid_obstacles"
        self.shapes = shapes_of(L)
        self.x, self.y, self.yaw = L["start"]
        self.v = self.wz = 0.0
        self.collided = False
        self.last = self.prev_opt = None
        self.gate, self.route_i, self.k, self.ret = 1, None, 0, 0.0
        self.noise = Rand(seed)
        self.drift = WALK["drift_w"] * self.noise.gauss()
        self.weights = WEIGHTS[task]
        self.n_opt = len(TARGET_KEYS) if task == "choose_object" else len(NAV)
        self.prev = self._init_prev()

    def clone(self, seed=None):
        c = copy.copy(self)
        c.noise = copy.copy(self.noise) if seed is None else Rand(seed)
        c.prev = dict(self.prev)
        return c

    # 관찰 · brain.json의 observation 순서 그대로
    def body(self, x, y):
        dx, dy, c, s = x - self.x, y - self.y, math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def goal_feats(self, x, y):
        bx, by = self.body(x, y)
        d, b = math.hypot(bx, by), math.atan2(by, bx)
        return [min(d, 4.0) / 4.0, math.sin(b), math.cos(b)]

    def motion(self):
        return [self.v / 0.2, self.wz / 0.6]

    def ray_feats(self):
        return [1 - t / RAY_RANGE for t in cast(self.x, self.y, self.yaw, self.shapes)]

    def last_opt(self):
        v = [0.0] * self.n_opt
        if self.last is not None:
            v[self.last] = 1.0
        return v

    def observe(self):
        T, L = self.task, self.L
        if T == "goto_target":
            return self.goal_feats(*L["goal"]) + self.motion() + self.last_opt() + (self.ray_feats() if self.rays_on else [])
        if T == "avoid_obstacles":
            return self.goal_feats(*L["goal"]) + self.ray_feats() + self.motion() + self.last_opt()
        if T == "choose_object":
            f = [float(c == L["cue"]) for c in TARGET_KEYS]
            have = {t[0]: t for t in L["targets"]}
            for c in TARGET_KEYS:
                f += [1.0] + self.goal_feats(have[c][1], have[c][2]) if c in have else [0.0, 0.0, 0.0, 0.0]
            return f + self.last_opt() + self.motion()
        _, lat, tang = self.project()
        wp, f = L["waypoints"], []
        for k in range(3):
            j = min(self.gate + k, len(wp) - 1)
            bx, by = self.body(*wp[j])
            f += [min(max(bx / 2, -1), 1), min(max(by / 2, -1), 1)]
        he = wrap(tang - self.yaw)
        f += [min(max(lat / 0.5, -2), 2), math.sin(he), math.cos(he)] + self.motion() + self.last_opt()
        return f + (self.ray_feats() if self.rays_on else [])

    def project(self):
        P, px, py = self.L["path"], self.x, self.y
        n = len(P) - 1
        dd, tt = [0.0] * n, [0.0] * n
        for i in range(n):
            ax, ay = P[i]
            bx, by = P[i + 1][0] - ax, P[i + 1][1] - ay
            t = min(max(((px - ax) * bx + (py - ay) * by) / max(bx * bx + by * by, 1e-9), 0), 1)
            tt[i], dd[i] = t, math.hypot(ax + bx * t - px, ay + by * t - py)
        lo, hi = (0, n) if self.route_i is None else (max(0, self.route_i - 8), min(n, self.route_i + 12))
        i = lo
        for k in range(lo + 1, hi):
            if dd[k] < dd[i]:
                i = k
        self.route_i = i
        bx, by = P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1]
        s = self.L["s"][i] + tt[i] * math.hypot(bx, by)
        cross = bx * (py - P[i][1]) - by * (px - P[i][0])
        return s, math.copysign(dd[i], cross) if cross else 0.0, math.atan2(by, bx)

    # 선택지 → 속도 명령
    def steer(self, gx, gy):
        e = wrap(math.atan2(gy - self.y, gx - self.x) - self.yaw)
        if abs(e) > 0.7:
            return NAV[3 if e > 0 else 4][1]
        if abs(e) > 0.18:
            return NAV[1 if e > 0 else 2][1]
        return NAV[0][1]

    def command(self, a):
        if self.task != "choose_object":
            return NAV[a][1]
        have = {t[0]: t for t in self.L["targets"]}
        c = TARGET_KEYS[a]
        return self.steer(have[c][1], have[c][2]) if c in have else NAV[3][1]   # 무대에 없는 물건: 제자리에서 둘러보기

    # 보상 항목
    def _dists(self):
        return [math.hypot(self.x - x, self.y - y) for _, x, y in self.L["targets"]]

    def _phi(self, ds):
        ci = self.L["cue_idx"]
        others = [d for i, d in enumerate(ds) if i != ci]
        return -ds[ci] + 0.5 * (sum(others) / len(others))

    def _init_prev(self):
        if self.task == "choose_object":
            return {"phi": self._phi(self._dists())}
        if self.task == "follow_route":
            self.gate, self.route_i = 1, None
            return {"s": self.project()[0]}
        return {"d": math.hypot(self.x - self.L["goal"][0], self.y - self.L["goal"][1])}

    def terms(self):
        T, p = self.task, self.prev
        if T in ("goto_target", "avoid_obstacles"):
            d = math.hypot(self.x - self.L["goal"][0], self.y - self.L["goal"][1])
            reach = d < 0.22 + (0.03 if T == "avoid_obstacles" else 0)
            return {"progress": p["d"] - d, "reach": float(reach), "time": 1.0, "collision": float(self.collided)}, reach, False, {"d": d}
        if T == "choose_object":
            ds, ci = self._dists(), self.L["cue_idx"]
            hit = [i for i, d in enumerate(ds) if d < 0.27]
            reach = ci in hit
            wrong = not reach and len(hit) > 0
            sw = float(self.last is not None and self.prev_opt is not None and self.last != self.prev_opt)
            phi = self._phi(ds)
            return {"progress": phi - p["phi"], "reach": float(reach), "wrong": float(wrong), "time": 1.0, "switch": sw}, reach, wrong, {"phi": phi}
        s, lat, _ = self.project()
        wp, gate = self.L["waypoints"], 0
        while self.gate < len(wp) and math.hypot(self.x - wp[self.gate][0], self.y - wp[self.gate][1]) < 0.28:
            self.gate += 1
            gate += 1
        reach = self.gate >= len(wp)
        return {"progress": s - p["s"], "offroute": abs(lat), "gate": float(gate), "reach": float(reach), "time": 1.0}, reach, False, {"s": s}

    # 0.5초 걷기 (2D 걷기 근사)
    def _response(self, cmd):
        ws = v = w = 0.0
        for c, tv, tw in WALK["table"]:
            d = ((cmd[0] - c[0]) / 0.2) ** 2 + ((cmd[1] - c[1]) / 0.6) ** 2
            k = 1.0 / (d + 1e-4) ** 2
            ws, v, w = ws + k, v + k * tv, w + k * tw
        return v / ws, w / ws

    def walk(self, cmd):
        P, g = WALK, self.noise.gauss
        av, aw = DT / P["tau_v"], DT / P["tau_w"]
        tv, tw = self._response(cmd)
        self.collided = False
        for _ in range(N_SUB):
            self.v += av * (tv - self.v) + P["noise_v"] * g() * math.sqrt(av)
            self.wz += aw * (tw + self.drift - self.wz) + P["noise_w"] * g() * math.sqrt(aw)
            nx = self.x + DT * self.v * math.cos(self.yaw) + P["vy_noise"] * g() * DT
            ny = self.y + DT * self.v * math.sin(self.yaw) + P["vy_noise"] * g() * DT
            if circle_hits(nx, ny, ROBOT_R, self.shapes):
                self.collided = True
                self.v *= 0.3
            else:
                self.x, self.y = nx, ny
            self.yaw = wrap(self.yaw + DT * self.wz)

    def step(self, a):
        """선택지 하나 → (보상, 끝났나, info) · 보상은 웹과 같은 공식 가중치"""
        a = int(a)
        self.prev_opt, self.last = self.last, a
        self.walk(self.command(a))
        raw, success, fail, upd = self.terms()
        self.prev.update(upd)
        out = abs(self.x) > ARENA_HALF + 0.3 or abs(self.y) > ARENA_HALF + 0.3
        raw["fall"], raw["out"] = 0.0, float(out)
        r = sum(self.weights.get(k, 0.0) * v for k, v in raw.items())
        self.k += 1
        self.ret += r
        done = success or fail or out or self.k >= MAX_DECISIONS[self.task]
        return r, done, {"success": success, "fail": fail, "out": out, "collided": self.collided}


def make_world(task, L, seed, rays=False):
    """웹 env.reset(seed, layout)과 같은 순서로 씨앗을 씁니다 (무작위 코스면 그 씨앗의 코스)"""
    r = Rand(seed)
    rl = random_layout(task, r)
    return World(task, L if L is not None else rl, rays=rays, seed=r.subseed())


def run_episode(act, task, L=None, seed=91001, rays=False, log=None):
    """한 판 · act(obs, world) → 선택지 번호"""
    w = make_world(task, L, seed, rays)
    collisions, done, info = 0, False, {}
    while not done:
        a = act(w.observe(), w)
        if log is not None:
            log.append((w.observe(), a))
        _, done, info = w.step(a)
        collisions += info["collided"]
    return {"seed": seed, "success": bool(info["success"]), "wrong": bool(info["fail"]), "out": bool(info["out"]),
            "collisions": collisions, "ret": w.ret, "k": w.k}


def evaluate(act, task, L=None, seeds=TEST_SEEDS, rays=False):
    """20판 · 성공률 · 부딪힌 판 · 틀린 물건 · 평균 보상 · 성공한 판의 평균 시간(초)"""
    eps = [run_episode(act, task, L, s, rays) for s in seeds]
    n, ok = len(eps), [e for e in eps if e["success"]]
    return {"n": n, "success": sum(e["success"] for e in eps) / n, "collide": sum(e["collisions"] > 0 for e in eps) / n,
            "wrong": sum(e["wrong"] for e in eps) / n, "ret": sum(e["ret"] for e in eps) / n,
            "time_s": sum(e["k"] for e in ok) * 0.5 / len(ok) if ok else None}
