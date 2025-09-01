from langchain.chains import ConversationalRetrievalChain
import logging
from typing import Any

logger = logging.getLogger(__name__)

def get_chain(model: Any, retriever: Any, system_prompt: str = None) -> Any:
    logger.info("Creating conversational retrieval chain.")
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever
    )
