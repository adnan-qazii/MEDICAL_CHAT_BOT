from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import logging
from typing import List

logger = logging.getLogger(__name__)

def load_pdf(data_path: str) -> List:
    try:
        loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        logger.info(f'Loaded {len(documents)} document(s).')
        return documents
    except Exception as e:
        logger.error(f"Error loading PDFs: {e}")
        raise
