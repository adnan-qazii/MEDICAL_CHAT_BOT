from langchain_pinecone import PineconeVectorStore
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

def create_vector_store(index_name: str, embedding: Any, documents: List) -> Any:
    try:
        return PineconeVectorStore.from_documents(
            index_name=index_name,
            embedding=embedding,
            documents=documents
        )
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise
