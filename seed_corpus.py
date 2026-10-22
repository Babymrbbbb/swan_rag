"""灌库脚本：把 data/corpus/*.md 批量 ingest 进向量库。

用法：
    python seed_corpus.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlmodel import Session

from app.database import engine, init_db
from app.rag import ingest


def main() -> None:
    init_db()
    corpus_dir = Path(__file__).resolve().parent / "data" / "corpus"
    count = 0
    with Session(engine) as s:
        for f in sorted(corpus_dir.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip() if text.startswith("#") else f.stem
            ingest(s, title=title, content=text, source=f.name)
            count += 1
    print(f"已灌库 {count} 篇文档（位于 {corpus_dir}）")


if __name__ == "__main__":
    main()
