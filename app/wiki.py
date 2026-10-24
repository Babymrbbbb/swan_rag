"""聚合层（wiki 层）：把 raw corpus 编译为可导航的聚合页。

这是 llm-wiki「raw → wiki → RAG 双轨」的 swan_rag 可运行版。
设计取舍：
- 当前实现为**确定性规则聚合**（基于业务词表 + 文档标题/首行），不依赖外部 LLM API，
  离线可跑、可测试、可复现；接入真实 LLM 后可升级为「LLM 识别实体/概念 + 生成综合摘要」。
- 聚合页是检索的第一入口：命中聚合页 → 返回聚合内容与来源列表；
  未命中则 fallback 到原始 chunk（RAG 常规路径）。

产出：
- `data/wiki/index.json`：聚合索引（entity/concept → 源文件列表）
- 内存聚合页对象：标题、摘要、关键事实、来源
"""
import json
import re
from pathlib import Path
from typing import Dict, List

# 业务实体/概念词表：来自 corpus 实际覆盖的主题（与 business_facts/store_facts 一致）
_ENTITY_TERMS = {
    "黑金可颂": "entity/黑金可颂",
    "联名桃酥": "entity/联名桃酥",
    "荔枝玫瑰冻芝士": "entity/荔枝玫瑰冻芝士",
    "青提冷萃欧包": "entity/青提冷萃欧包",
    "冰面包": "entity/冰面包",
    "金粒门": "entity/金粒门",
    "玛芝莲": "entity/玛芝莲",
    "天鹅公主": "entity/天鹅公主",
    "天鹅卡": "entity/天鹅卡",
}

_CONCEPT_TERMS = {
    "门店网络": "concept/门店网络",
    "加盟政策": "concept/加盟政策",
    "包材设计": "concept/包材设计",
    "会员体系": "concept/会员体系",
    "供应链": "concept/供应链",
    "数字化": "concept/数字化",
    "运营FAQ": "concept/运营FAQ",
    "夏季新品": "concept/夏季新品",
}

WIKI_DIR = Path(__file__).resolve().parent.parent / "data" / "wiki"
INDEX_PATH = WIKI_DIR / "index.json"


class WikiPage:
    """聚合页：标题、摘要、关键事实、来源文档列表。"""

    def __init__(self, title: str, kind: str, summary: str, facts: List[str], sources: List[str]):
        self.title = title
        self.kind = kind  # entity | concept
        self.summary = summary
        self.facts = facts
        self.sources = sources

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "kind": self.kind,
            "summary": self.summary,
            "facts": self.facts,
            "sources": self.sources,
        }


def _clean_title(text: str) -> str:
    """去掉 markdown 标题符号。"""
    return text.lstrip("# ").strip()


def _extract_key_facts(text: str) -> List[str]:
    """从正文提取关键事实：包含数字/日期/温度等的句子（简单规则）。"""
    sentences = re.split(r"[。\n]+", text)
    facts = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if re.search(r"\d|℃|%|㎡|万元|元|周|天", s):
            facts.append(s)
    return facts[:5]


class WikiIndex:
    """聚合层：从 corpus 编译聚合页，并提供检索。"""

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir
        self.pages: Dict[str, WikiPage] = {}
        self.term_to_page: Dict[str, str] = {}  # 业务词 -> 聚合页标题
        self._build()

    def _build(self) -> None:
        """扫描 corpus 文件，按业务词表生成聚合页。"""
        docs: List[tuple[str, str]] = []  # (title, content)
        for f in sorted(self.corpus_dir.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            title = _clean_title(text.splitlines()[0]) if text.startswith("#") else f.stem
            docs.append((title, text))

        # 按实体词聚合：所有包含该词的文档
        for term, page_path in {**_ENTITY_TERMS, **_CONCEPT_TERMS}.items():
            sources = [t for t, c in docs if term in c]
            if not sources:
                continue
            # 摘要：取第一个包含该词的文档首行描述
            summary = ""
            facts: List[str] = []
            for t, c in docs:
                if term in c:
                    summary = c.split("\n", 1)[-1].strip()[:120]
                    facts = _extract_key_facts(c)
                    break
            page_title = page_path.split("/", 1)[1]
            self.pages[page_title] = WikiPage(
                title=page_title,
                kind=page_path.split("/", 1)[0],
                summary=summary,
                facts=facts,
                sources=sources,
            )
            self.term_to_page[term] = page_title

    def search(self, query: str) -> List[WikiPage]:
        """检索聚合页：命中业务词则返回对应聚合页。"""
        hits = []
        for term, page_title in self.term_to_page.items():
            if term in query:
                hits.append(self.pages[page_title])
        return hits

    def to_index_json(self) -> dict:
        return {
            "generated": "deterministic-rule-based",
            "pages": {title: p.to_dict() for title, p in self.pages.items()},
        }

    def save_index(self) -> None:
        WIKI_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(json.dumps(self.to_index_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def build_wiki(corpus_dir: Path = None) -> WikiIndex:
    """构建聚合层。corpus_dir 默认指向 data/corpus。"""
    if corpus_dir is None:
        corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"
    return WikiIndex(corpus_dir)