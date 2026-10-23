"""数据接地护栏：确保知识库 corpus 与权威真相源一致，阻断错误资料回流。

背景：swan_rag 曾因 corpus 凭空编造「43 家含韶关」（韶关根本不存在）且改源后未重建索引，
导致线上长期返回错误门店数。本测试把门店数据钉死在 data/store_facts.json（唯一真相源），
任何「虚构城市 / 错门店数 / 总数漂移 / 分布缺失」都会在 pytest（即 CI）阶段直接失败。
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
FACTS_PATH = ROOT / "data" / "store_facts.json"

facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))


def _corpus_text() -> str:
    return "\n".join(
        f.read_text(encoding="utf-8") for f in sorted(CORPUS_DIR.glob("*.md"))
    )


def test_truth_source_sum_consistent():
    """真相源自身须自洽：各城市之和 = 总数。"""
    assert sum(facts["distribution"].values()) == facts["total_stores"]


def test_no_fabricated_city():
    """corpus 不得出现虚构城市（如曾凭空写入的『韶关』）。"""
    text = _corpus_text()
    for city in facts["nonexistent_cities"]:
        assert city not in text, f"corpus 仍含虚构城市『{city}』，须从真相源剔除"


def test_store_total_matches_truth():
    """corpus 必须声明正确的门店总数。"""
    text = _corpus_text()
    assert f"{facts['total_stores']} 家" in text, (
        f"corpus 未出现正确门店总数『{facts['total_stores']} 家』"
    )


def test_distribution_present():
    """corpus 必须覆盖真相源中每个城市及其数量。"""
    text = _corpus_text()
    for city, n in facts["distribution"].items():
        base = city.split("（")[0]  # 容忍「清远（含连州）」等带括号写法
        assert base in text, f"corpus 缺失城市『{city}』"
        assert str(n) in text, f"corpus 缺失城市『{city}』的数量 {n}"


def test_no_wrong_store_count():
    """corpus 不得再出现曾错误写出的门店数表述（如『43 家』）。"""
    text = _corpus_text()
    for wrong in ("43 家", "43家", "43 家直营"):
        assert wrong not in text, f"corpus 仍含错误门店数表述『{wrong}』"
