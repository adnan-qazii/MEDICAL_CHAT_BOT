from pinecone import Pinecone, ServerlessSpec
import logging
from typing import Any

logger = logging.getLogger(__name__)

def setup_pinecone(api_key: str, index_name: str, dimension: int = 384) -> Any:
    try:
        pc = Pinecone(api_key=api_key)
        # Check if index exists before creating
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(region="us-east-1", cloud="aws")
            )
        return pc
    except Exception as e:
        logger.error(f"Error setting up Pinecone: {e}")
        raise
