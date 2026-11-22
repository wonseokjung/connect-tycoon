import json

from connect_tycoon.cli import main


def test_demo_prints_jev(capsys):
    main(["demo", "--dept", "security"])
    out = capsys.readouterr().out
    assert '"choice"' in out and '"criteria"' in out


def test_jev_export(tmp_path):
    out = tmp_path / "m.jsonl"
    main(["jev", "--seed", "7000", "--out", str(out)])
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert rows and {"request", "response", "reward"} <= set(rows[0])


def test_ask(tmp_path, capsys):
    req = {"state": {"department": "refund", "amount": 11, "reason": "의심", "tier": "신규", "fraud_score": 0.9},
           "questions": {"decision": {"type": "choice", "criteria": {"full": "전액 환불", "deny": "거절", "human": "사람 확인"}}}}
    p = tmp_path / "r.json"; p.write_text(json.dumps(req, ensure_ascii=False), encoding="utf-8")
    main(["ask", "--request", str(p)])
    ans = json.loads(capsys.readouterr().out)["answers"]["decision"]
    assert ans["choice"] in ("full", "deny", "human")


def test_llm_data_keeps_teacher_values(tmp_path):
    """llm data는 선생님 brain.json이 배운 회사의 가치로 문제집을 만들어야 합니다 (기본 1·1·1로 덮어쓰면 가치가 사라짐)."""
    import connect_tycoon as ct
    b = ct.CompanyBrain.default()
    b.spec["company"]["values"] = {"이익": 1.0, "고객 만족": 1.5, "위험": 2.0}
    brain = tmp_path / "brain.json"; brain.write_text(json.dumps(b.spec, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "llm_data"
    main(["llm", "data", "--teacher", str(brain), "--out", str(out), "--train-months", "1", "--stride", "50"])
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["values"] == {"이익": 1.0, "고객 만족": 1.5, "위험": 2.0}
    row = json.loads(open(out / "rl" / "train.jsonl", encoding="utf-8").readline())
    assert "고객 만족 1.5 · 위험 회피 2" in row["task"]["state"]
