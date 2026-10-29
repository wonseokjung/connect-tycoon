"""Mac (Apple Silicon) · MLX로 지도학습 → 강화학습.

    pip install mlx-lm
    connect-tycoon llm data --out llm_data
    connect-tycoon llm sft  --data llm_data --out adapters/sft
    connect-tycoon llm rl   --data llm_data --from adapters/sft --out adapters/rl
"""
import json
import math
import os
import random
import shutil
import subprocess
import sys

from .data import load
from .format import BASE_MODEL, LETTERS, messages


class Scorer:
    """문제 하나 → 보기 글자별 확률. 학습 전 Qwen · 지도학습 · 강화학습 어댑터 모두 같은 방법으로 채점합니다."""

    def __init__(self, model=BASE_MODEL, adapter=None):
        import mlx.core as mx
        from mlx_lm import load as _load
        mx.set_cache_limit(2 * 1024 ** 3)                     # 길이가 다른 문제를 많이 읽어도 메모리가 쌓이지 않게
        self.model, self.tok = _load(model, adapter_path=adapter)
        self._n = 0
        self.letters = [self.tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
        self.name = os.path.basename(os.path.normpath(adapter)) if adapter else model
        self._cache = {}

    def ids(self, t):
        text = self.tok.apply_chat_template(messages(t), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok.encode(text, add_special_tokens=False)

    def probs(self, t):
        import mlx.core as mx
        key = json.dumps(t, ensure_ascii=False, sort_keys=True)
        if key in self._cache:
            return self._cache[key]
        z = self.model(mx.array([self.ids(t)]))[0, -1]
        z = z[mx.array(self.letters[:len(t["options"])])]
        p = mx.softmax(z.astype(mx.float32)).tolist()
        self._n += 1
        if self._n % 200 == 0: mx.clear_cache()
        if len(self._cache) > 20000: self._cache.clear()
        self._cache[key] = p
        return p

    def pick(self, t):
        p = self.probs(t)
        return LETTERS[max(range(len(p)), key=lambda i: p[i])]


def advantages(rows, clip=3.0):
    """선택지별 장점 = (점수 − 그 문제의 평균) ÷ 전체 문제의 표준편차. 차이가 작은 문제는 작게, 큰 문제는 크게 반영합니다.
    물류 실험실 문제(dept logistics)는 labdata.advantages · LINK-01 로봇 문제(dept link01)는 link01.questions.advantages ·
    두뇌의 답을 기준으로 확실한 차이만 씁니다."""
    if rows and rows[0].get("dept") == "logistics":
        from .labdata import advantages as lab
        return lab(rows, clip=clip)
    if rows and rows[0].get("dept") == "link01":                # LINK-01 로봇 문제 · 같은 방식 (두뇌의 답 기준)
        from ..link01.questions import advantages as world
        return world(rows, clip=clip)
    cen = [[x - sum(r["scores"]) / len(r["scores"]) for x in r["scores"]] for r in rows]
    flat = [x for c in cen for x in c]
    g = (sum(x * x for x in flat) / max(1, len(flat))) ** 0.5 or 1.0
    return [[max(-clip, min(clip, x / g)) for x in c] for c in cen]


def sft(data_dir, out, model=BASE_MODEL, epochs=1.0, lr=1e-4, batch=8, micro=4, rank=8, log=print):
    """② 지도학습 · 점수가 가장 높은 보기 글자를 따라 배웁니다 (mlx_lm.lora · 답 글자만 학습)."""
    sdir = os.path.join(data_dir, "sft")
    n = sum(1 for _ in open(os.path.join(sdir, "train.jsonl"), encoding="utf-8"))
    accum = max(1, batch // micro)
    updates = max(1, math.ceil(n * epochs / batch))
    iters = updates * accum
    cfg = {"model": model, "train": True, "data": sdir, "fine_tune_type": "lora", "mask_prompt": True, "num_layers": -1,
           "lora_parameters": {"rank": rank, "scale": 2.0, "dropout": 0.0}, "batch_size": micro, "grad_accumulation_steps": accum,
           "iters": iters, "learning_rate": lr,
           "lr_schedule": {"name": "cosine_decay", "arguments": [lr, updates, 0.0], "warmup": max(1, int(0.03 * updates)), "warmup_init": 0.0},
           "max_seq_length": 2048, "grad_checkpoint": True, "steps_per_report": 10, "steps_per_eval": max(20, iters // 5),
           "val_batches": -1, "save_every": iters, "adapter_path": out, "seed": 42}
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "train_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1)
    if log: log(f"지도학습 · 문제 {n}개 · {epochs}회 · 업데이트 {updates}번")
    code = subprocess.call([sys.executable, "-m", "mlx_lm.lora", "-c", path])
    if code: raise RuntimeError(f"mlx_lm.lora 실패 (code {code})")
    return out


def rl(data_dir, start, out, model=BASE_MODEL, steps=100, batch=8, lr=5e-6, beta=0.2, eval_every=50, seed=0, max_rows=1500, log=print):
    """③ 강화학습 · 선택지 전체를 한 묶음(group)으로 보고, 시뮬레이터 점수가 평균보다 높은 보기의 확률은 올리고 낮은 보기는 내립니다.

    손실 = −Σ p(보기) · 장점(보기)  +  β · KL(p ‖ 출발 모델)
    장점 = 그 문제 안에서 표준화한 선택지별 점수 (GRPO의 묶음 표준화). 표본을 뽑는 대신 모든 보기의 확률을 정확히 씁니다.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map

    sc = Scorer(model, start)
    m = sc.model
    rows = [r for r in load(data_dir, "train") if r.get("label") != "teacher"]   # 강화학습은 시뮬레이터 점수가 또렷한 낮의 일만
    valid = load(data_dir, "valid")
    rnd = random.Random(seed)
    if max_rows and len(rows) > max_rows:                     # 강화학습은 문제 일부로도 충분합니다 (출발 확률을 미리 계산하는 시간 절약)
        rows = rnd.sample(rows, max_rows)
    if log: log(f"강화학습 · 문제 {len(rows)}개 · {steps}걸음 × {batch}문제 · 출발 {start}")
    prep = []
    for r, adv in zip(rows, advantages(rows)):                # 출발 모델의 확률(KL 기준)과 장점을 미리 계산
        prep.append((sc.ids(r["task"]), len(adv), adv, sc.probs(r["task"])))

    m.freeze()
    m.apply_to_modules(lambda k, mod: mod.unfreeze(keys=["lora_a", "lora_b"], strict=False) if hasattr(mod, "lora_a") else None)
    m.train()
    letters = sc.letters

    def loss_fn(model, ids, n, adv, pref):
        z = model(ids[None])[0, -1]
        z = z[mx.array(letters[:n])].astype(mx.float32)
        logp = z - mx.logsumexp(z)
        p = mx.exp(logp)
        return -(p * adv).sum() + beta * (p * (logp - mx.log(pref))).sum()

    vg = nn.value_and_grad(m, loss_fn)
    opt = optim.Adam(learning_rate=lr)
    hist = []
    for step in range(1, steps + 1):
        acc_g, tot = None, 0.0
        for _ in range(batch):
            ids, n, adv, pref = prep[rnd.randrange(len(prep))]
            loss, g = vg(m, mx.array(ids), n, mx.array(adv), mx.array(pref))
            acc_g = g if acc_g is None else tree_map(lambda a, b: a + b, acc_g, g)
            mx.eval(loss, acc_g)                              # 문제마다 바로 계산 (미뤄 두면 8문제의 계산 그래프가 한꺼번에 메모리에 쌓입니다)
            tot += loss.item()
        opt.update(m, tree_map(lambda a: a / batch, acc_g))
        mx.eval(m.parameters(), opt.state)
        mx.clear_cache()
        if step % eval_every == 0 or step == steps:
            m.eval(); sc._cache.clear()
            from .data import agreement
            ev = agreement(valid, sc.pick)
            m.train()
            hist.append({"step": step, "loss": tot / batch, **ev})
            if log: log(f"걸음 {step:4d} · 손실 {tot / batch:+.3f} · 검증 정답률 {ev['accuracy']:.1%} · 후회 {ev['regret']:.2f}")
    os.makedirs(out, exist_ok=True)
    mx.save_safetensors(os.path.join(out, "adapters.safetensors"), dict(tree_flatten(m.trainable_parameters())))
    shutil.copy(os.path.join(start, "adapter_config.json"), os.path.join(out, "adapter_config.json"))
    with open(os.path.join(out, "rl_history.json"), "w") as f:
        json.dump({"start": start, "steps": steps, "batch": batch, "lr": lr, "beta": beta, "history": hist}, f, indent=1)
    m.eval()
    return out, hist
