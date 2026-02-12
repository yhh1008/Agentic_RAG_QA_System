from __future__ import annotations

from langchain_chroma import Chroma

from src.config.settings import settings
from src.retrieval.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=settings.chroma_persist_dir,
        embedding_function=get_embeddings(),
    )
