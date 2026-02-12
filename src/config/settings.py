from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    embedding_model_path: str = os.getenv(
        "EMBEDDING_MODEL_PATH",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./rag_cache/chroma_db")
    trace_dir: str = os.getenv("TRACE_DIR", "./train/data/traces")

    top_k_recall: int = int(os.getenv("TOP_K_RECALL", "10"))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", "5"))
    keyword_bonus: float = float(os.getenv("KEYWORD_BONUS", "0.1"))
    max_attempts: int = int(os.getenv("MAX_ATTEMPTS", "3"))
    selective_read_char_threshold: int = int(os.getenv("SELECTIVE_READ_CHAR_THRESHOLD", "1000"))


settings = Settings()
Path(settings.trace_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
