"""聚合层（wiki）测试：验证 raw→wiki 双轨的聚合页检索。

覆盖：
1. 聚合索引能构建出页面（实体/概念）
2. 检索命中聚合页：问"冰面包"命中 entity/冰面包
3. 聚合页含来源列表与关键事实
"""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="swan_rag_wiki_test_")
os.environ["RAG_RAG_DB_FILE"] = os.path.join(_TMP, "test.db")
os.environ["RAG_API_KEY"] = "test-secret-key"
os.environ["RAG_LLM_PROVIDER"] = "mock"
os.environ["RAG_EMBEDDER_PROVIDER"] = "mock"
os.environ["RAG_AUTO_SEED"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.wiki import build_wiki  # noqa: E402


def test_wiki_index_builds_pages():
    wiki = build_wiki()
    assert len(wiki.pages) > 0, "聚合索引为空"
    assert "冰面包" in wiki.pages, f"缺少 entity/冰面包 聚合页，实际: {list(wiki.pages.keys())}"


def test_wiki_search_hit_entity():
    wiki = build_wiki()
    hits = wiki.search("冰面包怎么保存？")
    assert hits, "检索冰面包未命中聚合页"
    assert hits[0].title == "冰面包"
    assert hits[0].kind == "entity"
    assert hits[0].sources, "聚合页缺少来源列表"
    assert hits[0].facts, "聚合页缺少关键事实"


def test_wiki_search_hit_concept():
    wiki = build_wiki()
    hits = wiki.search("加盟政策需要什么条件？")
    assert hits, "检索加盟政策未命中聚合页"
    assert hits[0].title == "加盟政策"
    assert hits[0].kind == "concept"


def test_wiki_save_index():
    wiki = build_wiki()
    wiki.save_index()
    index_path = ROOT / "data" / "wiki" / "index.json"
    assert index_path.exists(), "聚合索引文件未生成"