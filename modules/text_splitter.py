from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from typing import List

logger = logging.getLogger(__name__)

def split_documents(documents: List, chunk_size: int = 500, chunk_overlap: int = 200) -> List:
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = splitter.split_documents(documents)
        logger.info(f'Split into {len(texts)} chunks.')
        return texts
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise
