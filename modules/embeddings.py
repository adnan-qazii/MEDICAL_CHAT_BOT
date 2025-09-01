from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from typing import Any

logger = logging.getLogger(__name__)

def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> Any:
    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise
