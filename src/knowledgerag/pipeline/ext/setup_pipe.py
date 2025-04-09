import os
from pathlib import Path
from tempfile import mkdtemp

from docling_haystack.converter import ExportType
from dotenv import load_dotenv


def _get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata

        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)


load_dotenv()

HF_TOKEN = _get_env_from_colab_or_os("HF_TOKEN")
PATHS = ["https://arxiv.org/pdf/2408.09869"]  # Docling Technical Report
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
EXPORT_TYPE = ExportType.DOC_CHUNKS
QUESTION = "Which are the main AI models in Docling?"
TOP_K = 3
MILVUS_URI = str(Path(mkdtemp()) / "docling.db")


