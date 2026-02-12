from __future__ import annotations

from pathlib import Path
import uuid

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.retrieval.chroma_store import get_vectorstore


RAW_DIR = Path("data/raw")


def load_docs() -> list[Document]:
    docs: list[Document] = []
    for p in RAW_DIR.rglob("*"):
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in {".md", ".txt"}:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
    return docs


def main() -> None:
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    out: list[Document] = []
    for idx, c in enumerate(chunks):
        content = c.page_content
        meta = c.metadata or {}
        doc_id = str(meta.get("doc_id", Path(str(meta.get("source", "unknown"))).stem or "doc"))
        chunk_id = f"chunk_{idx}"
        source = str(meta.get("source", "unknown_source"))
        meta.update(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "source": source,
                "start_offset": 0,
                "end_offset": len(content),
                "uid": str(uuid.uuid4()),
            }
        )
        out.append(Document(page_content=content, metadata=meta))

    vs = get_vectorstore()
    if out:
        vs.add_documents(out)
        vs.persist()
    print(f"indexed_chunks={len(out)}")


if __name__ == "__main__":
    main()
