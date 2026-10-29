"""Colab에서 brain.json 찾기 · 파일 선택 버튼(files.upload)은 브라우저 설정에 따라 안 보이거나 멈추는 경우가 있어,
왼쪽 📁 파일 창에 끌어다 놓은 파일을 먼저 찾습니다."""
import glob
import os


class BrainNotFound(Exception):
    pass


HOW = ("brain.json을 찾지 못했습니다.\n"
       "  1) Colab 왼쪽 세로 막대의 📁(파일) 아이콘을 누릅니다\n"
       "  2) 내려받은 brain.json(게임의 회사이름-v2.json · 학습 실험실 · LINK-01 로봇 학습기의 link01-…-brain.json)을 그 창에 끌어다 놓습니다\n"
       "  3) 이 셀을 다시 실행합니다 (▶)\n"
       "  * 런타임을 다시 시작하면 올린 파일이 지워지니, 그때는 다시 끌어다 놓으세요")


def find_brain(folder="/content", upload_button=True):
    """두 가지 다 됩니다 · 1) 📁 파일 창에 끌어다 놓은 brain.json (가장 최근 것)  2) 없으면 파일 선택 버튼
    버튼이 안 보이는 브라우저도 있어서, 버튼 위에 끌어다 놓는 방법을 함께 안내합니다."""
    import connect_tycoon as ct
    found = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json")), key=os.path.getmtime, reverse=True):
        try:
            from .lab import is_lab
            from .link01.brain import is_world
            if not is_lab(p) and not is_world(p):              # 학습 실험실 · LINK-01 로봇 두뇌 · 게임 두뇌
                ct.CompanyBrain.from_file(p)
            found.append(p)
        except Exception:
            continue
    if found:
        print("찾은 brain.json ·", os.path.basename(found[0]) + (f" (다른 후보 {len(found) - 1}개는 무시)" if len(found) > 1 else ""))
        return found[0]
    if upload_button:
        print("아래 [파일 선택] 버튼을 눌러 brain.json을 고르세요.\n"
              "버튼이 안 보이거나 눌러도 반응이 없으면: 이 셀을 멈추고(■) → 왼쪽 📁(파일) 창에 brain.json을 끌어다 놓고 → 이 셀을 다시 실행(▶)")
        from google.colab import files
        up = files.upload()
        if up:                                                 # brain.json과 world.json을 같이 골라도 두뇌 쪽을 돌려줍니다
            return find_brain(os.getcwd(), upload_button=False)
    raise BrainNotFound(HOW)


def find_course(folder="/content"):
    """📁에 함께 끌어다 놓은 LINK-01 코스(world.json · connect-world-course/1) · 가장 최근 것, 없으면 None"""
    import json
    for p in sorted(glob.glob(os.path.join(folder, "*.json")), key=os.path.getmtime, reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                if json.load(f).get("format") == "connect-world-course/1":
                    return p
        except Exception:
            continue
    return None
