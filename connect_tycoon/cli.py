# -*- coding: utf-8 -*-
"""connect-tycoon 명령

    connect-tycoon play                         게임 (pygame-ce 필요)
    connect-tycoon train --name 내두뇌 --profit 2   두뇌 키우기 → brain.json
    connect-tycoon eval --brain brain.json       처음 보는 40개월 시험
    connect-tycoon jev --brain hf:아이디/저장소 --out month.jsonl   한 달의 (JEV 요청, 응답, 보상) 기록
    connect-tycoon ask --brain default < request.json            JEV 요청 하나에 답하기
    connect-tycoon demo                          JEV 요청 · 응답 예시 출력
    connect-tycoon record --brain default --out frames/          하루를 그림으로 (Colab 영상용)
    connect-tycoon push --brain brain.json --repo 아이디/connect-company-이름   허깅페이스에 올리기
"""
import argparse
import json
import os
import sys

UPGRADE_HELP = "업그레이드 단계 JSON, 예: '{\"demand\":1,\"logistics\":1}'"
PROFILE_HELP = "회사의 분포: general(공개 데이터로 맞춘 일반 회사 · 기본) · lecture(강의 엔진) · 프로필 JSON 경로"


def _config(args):
    from .career import config_for
    from .company import DEFAULT_CONFIG
    cfg = dict(DEFAULT_CONFIG)
    if getattr(args, "upgrades", None):
        cfg = config_for(json.loads(args.upgrades))
    if getattr(args, "config", None):
        cfg.update(json.loads(args.config))
    return cfg


def _company(args, values=None):
    import connect_tycoon as ct
    return ct.make_company(config=_config(args), values=values, profile=getattr(args, "profile", None))


def cmd_play(args):
    try:
        import pygame  # noqa: F401
    except ImportError:
        sys.exit("게임에는 pygame-ce가 필요합니다:  pip install 'connect-tycoon[game]'")
    from .game.app import main
    w, h = (int(v) for v in args.size.lower().split("x"))
    main(brains=args.brain or (), size=(w, h), fullscreen=args.fullscreen, reset=args.reset, sfx=not args.no_sfx, music=not args.no_music)


def cmd_train(args):
    import connect_tycoon as ct
    values = {"이익": args.profit, "고객 만족": args.sat, "위험": args.risk}
    co = _company(args, values)
    print(f"회사 {co.profile_name} · 규모 {co.cfg}\n회사의 가치 {values}")
    start = ct.CompanyBrain.load(args.start) if args.start else None
    if start:                                                  # 파인튜닝: 이 두뇌에서 이어서
        co = ct.Company(start.intuition, ct.load_pool(), values=values, config=co.cfg, profile=co.profile)
        teacher, pol = start.policy, start.policy.copy()
        print(f"출발점 · {start.name} (이어서 강화학습)")
    else:
        teacher = ct.RulePolicy()
        pol = ct.LinearPolicy()
        acc = ct.imitate(co, pol, teacher, months=args.imitate_months)
        print("① 지도학습 · 규칙 사장과 같은 판단:", {k: f"{v:.0%}" for k, v in acc.items()})
    print("② 강화학습(PPO)")
    ct.train_ppo(co, pol, iterations=args.iters, months_per_iter=args.months, lr=0.15 if start else 0.2, seed=args.seed)
    test = ct.evaluate(co, pol)
    base = ct.evaluate(co, teacher)
    print(f"처음 보는 40개월 · 새 두뇌 {test['profit']:.0f}만원/달 · {'출발 두뇌' if start else '규칙 사장'} {base['profit']:.0f}만원/달")
    spec = ct.CompanyBrain.make_spec((start or ct.CompanyBrain.default()).spec, pol, name=args.name, values=values, researcher=args.researcher,
                                     config=co.cfg, results={"test_profit": test["profit"], "rule_profit": base["profit"], "test_months": 40},
                                     profile=co.profile_name)
    out = args.out or f"{args.name}.json"
    ct.CompanyBrain(spec).save(out)
    print("저장:", out)


def cmd_eval(args):
    import connect_tycoon as ct
    from . import jev as J
    co = _company(args)
    seeds = range(7000, 7000 + args.months)
    rows = [("규칙 사장", ct.RulePolicy())]
    for ref in args.brain:
        b = ct.CompanyBrain.load(ref)
        rows.append((b.name, J.JevPolicy(b, values=co.values, config=co.cfg, keep=False, check=False)))
    print(f"처음 보는 {args.months}개월 평균 · 회사 규모 {co.cfg}")
    for name, pol in rows:
        r = ct.evaluate(co, pol, seeds)
        print(f"{name:24s} 이익 {r['profit']:7.0f}만원 · 만족 {r['sat']:.0%} · 위험 사고 {r['incidents']:4.1f}건 · 사람 확인 초과 {r['human_over']:5.1f}건 · 품절 {r['stockout_days']:4.1f}일")


def cmd_backtest(args):
    import connect_tycoon as ct
    from . import backtest
    co = _company(args)
    rows = [("규칙 사장", ct.RulePolicy())] + [(b.name, b) for b in (ct.CompanyBrain.load(r) for r in (args.brain or ["default"]))]
    print(f"실제 데이터 시험 · 회사 {co.profile_name} · 학습에 쓰지 않은 실제 기록")
    for name, pol in rows:
        c, d = backtest.card(pol, co), backtest.demand(pol, co)
        print(f"{name:24s} 실제 결제 {c['cases']}건 · 사기 {c['caught']}/{c['fraud']} 잡음 · 정상 차단 {c['false_block']} · 손실 {c['saved']:.0%} 줄임 | "
              f"실제 수요 재생 {d['profit']:6.0f}만원/달 (" + " · ".join(f"{k} {v['profit']:.0f}" for k, v in d.items() if k != "profit") + ")")


def cmd_jev(args):
    import connect_tycoon as ct
    from . import jev as J
    co = _company(args)
    b = ct.CompanyBrain.load(args.brain)
    rows = []
    for seed in range(args.seed, args.seed + args.months):
        pol = J.JevPolicy(b, values=co.values, config=co.cfg)
        log, days = co.run_month(seed, pol)
        rows += J.dataset(pol.log, log, seed=seed)
        print(f"{seed}: 판단 {len(pol.log)}개 · 이익 {sum(d['profit'] for d in days):.0f}만원")
    J.write_jsonl(rows, args.out)
    print(f"{len(rows)}줄 → {args.out}")


def cmd_ask(args):
    import connect_tycoon as ct
    b = ct.CompanyBrain.load(args.brain)
    req = json.load(open(args.request, encoding="utf-8")) if args.request else json.load(sys.stdin)
    print(json.dumps(b(req), ensure_ascii=False, indent=2))


def cmd_demo(args):
    import connect_tycoon as ct
    co = ct.make_company()
    b = ct.CompanyBrain.default()
    env = ct.CompanyEnv(company=co)
    env.reset(options={"month_seed": 7000})
    while env.decision.dept != args.dept:
        d = env.decision
        env.step(ct.DEPTS[d.dept]["actions"].index(b.act(d.dept, d.obs, d.ev)))
    req = env.jev_request()
    print("── JEV 요청 (회사 → 두뇌)"); print(json.dumps(req, ensure_ascii=False, indent=2))
    print("── JEV 응답 (두뇌 → 회사)"); print(json.dumps(b(req), ensure_ascii=False, indent=2))


def cmd_record(args):
    from .game.record import record
    record(args.brain, args.out, day=args.day, seed=args.seed, size=args.size, fps=args.fps, mp4=args.mp4, upgrades=args.upgrades)


def cmd_push(args):
    import importlib.util
    if not importlib.util.find_spec("huggingface_hub"):
        sys.exit("pip install huggingface_hub 가 필요합니다")
    from .hub import push_brain
    from .llm.hub import make_author
    from .lab import is_lab
    if is_lab(args.brain):                      # 웹 학습 실험실에서 내려받은 두뇌
        from .hub import push_lab_brain
        print(push_lab_brain(args.brain, repo=args.repo, version=args.version, name=args.name, author=make_author(_about(args))))
    else:
        print(push_brain(args.brain, repo=args.repo, version=args.version, name=args.name, author=make_author(_about(args))))


def _llm_backend(name):
    """mlx = Mac(Apple Silicon) · torch = Windows · Linux · Colab (NVIDIA GPU 권장) · auto = 설치된 것으로."""
    import importlib.util
    import platform
    if name == "auto":
        mac = platform.system() == "Darwin" and platform.machine() == "arm64"
        name = "mlx" if mac and importlib.util.find_spec("mlx_lm") else "torch"
    if name == "mlx":
        from .llm import train_mlx as M
    else:
        if not importlib.util.find_spec("torch"):
            sys.exit('PyTorch가 필요합니다:  pip install "connect-tycoon[llm]"  (Mac이면 pip install "connect-tycoon[llm-mac]")')
        from .llm import train_torch as M
    print(f"언어 모델 학습 도구 · {name}")
    return M


def cmd_llm(args):
    """언어 모델 단계: data → sft → rl → eval → push (Mac은 MLX, Colab은 노트북의 PyTorch 경로)."""
    import json as _json

    import connect_tycoon as ct
    from .llm import data as D
    if args.step == "collect":                                  # 모두의 모델 모으기 · 같은 시험지 · 연결
        from .llm import collect as C
        out = args.out if args.out != "llm_out" else "collect"
        models = C.find(prefix=args.prefix, limit=args.limit) + [C.local(p) for p in (args.adapter or [])]
        print("모델 연결(Connect AI)은 테스트 중인 기능입니다 · 결과는 참고용입니다")
        print(f"찾은 모델 {len(models)}개 · " + ", ".join(m["repo"] for m in models))
        test = C.common_test(os.path.join(out, "test"))
        res = C.score(models, test, out_dir=out, max_rows=args.questions)
        if not res:
            sys.exit("채점할 수 있는 모델이 없습니다 (Mac은 MLX · PyTorch 어댑터 모두, 그 밖에는 PyTorch 어댑터만)")
        board = C.leaderboard(res, test)
        print(C.table(board)); print("저장:", C.save(board, out))
        return
    if args.step == "data" and args.teacher.endswith(".json") and _is_lab(args.teacher):    # 웹 학습 실험실의 물류 두뇌
        from .lab import LabBrain
        from .llm import labdata as L
        splits = {"train": range(30000, 30000 + args.train_days), "valid": range(31000, 31004), "test": range(32000, 32004)}
        print(L.build(LabBrain.from_file(args.teacher), args.out, splits=splits, stride=args.stride))
        return
    if args.step == "data":
        teacher = ct.CompanyBrain.load(args.teacher)
        # 선생님이 배운 회사의 가치 · 분포로 문제집을 만듭니다 (노트북과 같게 · 기본 회사로 만들면 가치가 사라집니다)
        co = ct.make_company(profile=args.profile or teacher.spec["company"].get("profile") or "general", values=teacher.values)
        splits = {"train": range(20000, 20000 + args.train_months), "valid": range(21000, 21002), "test": range(22000, 22002)}
        print(D.build(co, teacher, args.out, splits=splits, stride=args.stride))
        return
    M = _llm_backend(args.backend)
    if args.step == "sft":
        M.sft(args.data, args.out, epochs=args.epochs, lr=args.lr or 1e-4)
    elif args.step == "rl":
        M.rl(args.data, args.start, args.out, steps=args.steps, batch=args.batch, lr=args.lr or 5e-6, beta=args.beta, max_rows=args.max_rows)
    elif args.step == "eval":
        from .llm.policy import LLMPolicy, report
        meta = _meta(args.data)
        co = ct.make_company(profile=args.profile or meta.get("profile") or "general", values=meta.get("values"))   # 문제집을 만든 회사와 같은 가치
        test = D.load(args.data, "test")
        lab = meta.get("kind") == "lab"
        out = {}
        for ad in [None] + (args.adapter or []):                 # 학습 전 Qwen도 같은 시험으로
            sc = M.Scorer(adapter=ad)
            name = "Qwen 학습 전" if ad is None else os.path.basename(os.path.normpath(ad))
            out[name] = D.agreement(test, sc.pick)
            if lab:                                               # 물류센터장 자리에 앉혀 시험 24일 (웹 실험실과 같은 날)
                from .llm.labdata import day_report
                out[name].update(day_report(sc.pick, log=None))
            elif args.months:
                out[name].update(report(LLMPolicy(sc, co.values), co, months=args.months, demand_months=0))
            print(name, _json.dumps(out[name], ensure_ascii=False))
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                _json.dump(out, f, ensure_ascii=False, indent=1)
    elif args.step == "push":
        from .llm.hub import push
        res = _json.load(open(args.results, encoding="utf-8")) if args.results else None
        meta = _meta(args.data) or None
        from huggingface_hub import HfApi
        from .llm.hub import repo_name
        kind = "logistics-qwen" if (meta or {}).get("kind") == "lab" else None
        repo = args.repo or f"{HfApi().whoami()['name']}/{repo_name(args.name or HfApi().whoami()['name'], args.version, kind)}"
        from .llm.hub import make_author
        print(push(args.adapter[0], repo=repo, results=res, meta=meta, author=make_author(_about(args))))


def _meta(data_dir):
    path = os.path.join(data_dir or "", "meta.json")
    if not data_dir or not os.path.exists(path):
        return {}
    import json as _json
    with open(path, encoding="utf-8") as f:
        return _json.load(f)


def _is_lab(path):
    from .lab import is_lab
    return is_lab(path)


def _about(args):
    """--about "글" 또는 --about-file 소개.md · 정해진 형식 없이 자유롭게."""
    if getattr(args, "about_file", None):
        return open(args.about_file, encoding="utf-8").read()
    return getattr(args, "about", "") or ""


ABOUT_HELP = "자기소개 글 · 자유롭게 (사업 · 서비스 · 채널 링크 등) · 모델 카드 맨 위 '만든 사람'에 그대로 들어갑니다"


def _utf8_console():
    """윈도우 콘솔(cp1252 등)에서도 한글이 깨지거나 멈추지 않도록 출력을 UTF-8로 바꿉니다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    _utf8_console()
    p = argparse.ArgumentParser(prog="connect-tycoon", description="AI 사장 키우기 · JEV 의사결정 두뇌를 강화학습으로 키우는 기업 경영 시뮬레이터")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("play", help="게임"); s.add_argument("--brain", action="append", help="두뇌 추가 (경로 또는 hf:아이디/저장소)")
    s.add_argument("--size", default="1280x800"); s.add_argument("--fullscreen", action="store_true"); s.add_argument("--reset", action="store_true", help="새 회사로 시작")
    s.add_argument("--no-sfx", action="store_true", help="효과음 끄기"); s.add_argument("--no-music", action="store_true", help="배경 음악 끄기")
    s.set_defaults(fn=cmd_play)
    s = sub.add_parser("train", help="두뇌 키우기"); s.add_argument("--name", default="my-company"); s.add_argument("--researcher", default="")
    s.add_argument("--profit", type=float, default=1.0); s.add_argument("--sat", type=float, default=1.0); s.add_argument("--risk", type=float, default=1.0)
    s.add_argument("--iters", type=int, default=60); s.add_argument("--months", type=int, default=24); s.add_argument("--imitate-months", type=int, default=40)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--upgrades", help=UPGRADE_HELP); s.add_argument("--config"); s.add_argument("--out")
    s.add_argument("--profile", default="general", help=PROFILE_HELP)
    s.add_argument("--from", dest="start", help="이 두뇌에서 이어서 훈련 (파인튜닝) · 경로 · default · hf:아이디/저장소")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("eval", help="시험"); s.add_argument("--brain", action="append", default=[]); s.add_argument("--months", type=int, default=40)
    s.add_argument("--upgrades", help=UPGRADE_HELP); s.add_argument("--config"); s.add_argument("--profile", default="general", help=PROFILE_HELP)
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("backtest", help="실제 데이터 시험"); s.add_argument("--brain", action="append")
    s.add_argument("--upgrades", help=UPGRADE_HELP); s.add_argument("--config"); s.add_argument("--profile", default="general", help=PROFILE_HELP)
    s.set_defaults(fn=cmd_backtest)
    s = sub.add_parser("llm", help="언어 모델(Qwen)에 옮기고 강화학습 · data → sft → rl → eval → push")
    s.add_argument("step", choices=["data", "sft", "rl", "eval", "push", "collect"])
    s.add_argument("--teacher", default="default", help="선생님 두뇌 (brain.json · default · hf:아이디/저장소 · 웹 실험실의 물류 brain.json도 됩니다)")
    s.add_argument("--profile", default="general", help=PROFILE_HELP); s.add_argument("--train-months", type=int, default=10)
    s.add_argument("--train-days", type=int, default=30, help="data: 물류 두뇌(웹 실험실 brain.json)일 때 문제집을 만들 날 수")
    s.add_argument("--stride", type=int, default=1); s.add_argument("--data", default="llm_data"); s.add_argument("--out", default="llm_out")
    s.add_argument("--from", dest="start", help="강화학습 출발 어댑터 (보통 sft 결과)"); s.add_argument("--adapter", action="append")
    s.add_argument("--epochs", type=float, default=1.0); s.add_argument("--lr", type=float, help="기본 sft 1e-4 · rl 5e-6")
    s.add_argument("--steps", type=int, default=100); s.add_argument("--batch", type=int, default=8); s.add_argument("--beta", type=float, default=0.2); s.add_argument("--max-rows", type=int, default=1500)
    s.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"], help="mlx(Mac) · torch(Windows · Linux · NVIDIA GPU)")
    s.add_argument("--prefix", default="connectailab-rl-", help="collect: 모을 모델 이름의 시작"); s.add_argument("--limit", type=int, default=200)
    s.add_argument("--questions", type=int, default=300, help="collect: 모델마다 채점할 문제 수")
    s.add_argument("--months", type=int, default=0, help="eval: 시뮬레이터에서 운영해 볼 달 수 (느림)"); s.add_argument("--save")
    s.add_argument("--results", help="push: eval --save로 저장한 성적"); s.add_argument("--repo")
    s.add_argument("--name", help="push: 업로드 이름 connectailab-rl-이름-버전의 이름 (영문)"); s.add_argument("--version", default="v1")
    s.add_argument("--about", default="", help="push: " + ABOUT_HELP); s.add_argument("--about-file", help="push: 자기소개 글 파일 (마크다운)")
    s.set_defaults(fn=cmd_llm)
    s = sub.add_parser("jev", help="JEV 기록 내보내기"); s.add_argument("--brain", default="default"); s.add_argument("--seed", type=int, default=7000)
    s.add_argument("--months", type=int, default=1); s.add_argument("--out", default="jev_month.jsonl"); s.add_argument("--upgrades", help=UPGRADE_HELP); s.add_argument("--config")
    s.set_defaults(fn=cmd_jev)
    s = sub.add_parser("ask", help="JEV 요청 하나에 답하기"); s.add_argument("--brain", default="default"); s.add_argument("--request"); s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("demo", help="JEV 예시"); s.add_argument("--dept", default="refund", choices=["staffing", "inventory", "marketing", "cs", "refund", "security"])
    s.set_defaults(fn=cmd_demo)
    s = sub.add_parser("record", help="하루를 그림/영상으로"); s.add_argument("--brain", default="default"); s.add_argument("--out", default="frames")
    s.add_argument("--day", type=int, default=0); s.add_argument("--seed", type=int, default=7000); s.add_argument("--size", default="1280x720")
    s.add_argument("--fps", type=int, default=30); s.add_argument("--mp4", action="store_true"); s.add_argument("--upgrades", help=UPGRADE_HELP)
    s.set_defaults(fn=cmd_record)
    s = sub.add_parser("push", help="brain.json을 허깅페이스에 올리기 · connectailab-rl-이름-버전-teacher (hf auth login 또는 HF_TOKEN)")
    s.add_argument("--brain", required=True); s.add_argument("--repo")
    s.add_argument("--name", help="이름 (영문) · 없으면 허깅페이스 아이디"); s.add_argument("--version", default="v1", help="버전 (예: v2)")
    s.add_argument("--about", default="", help=ABOUT_HELP); s.add_argument("--about-file", help="자기소개 글 파일 (마크다운)")
    s.set_defaults(fn=cmd_push)
    args = p.parse_args(argv)
    if not args.cmd:
        args = p.parse_args(["play"] + (argv or sys.argv[1:]))
    args.fn(args)


if __name__ == "__main__":
    main()
