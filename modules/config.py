import os
import logging
from dotenv import load_dotenv
from typing import Dict

logger = logging.getLogger(__name__)

def load_env() -> Dict[str, str]:
    try:
        load_dotenv()
        pinecone_key = os.getenv("PINECONE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        if not pinecone_key or not groq_key:
            logger.error("Missing required environment variables.")
            raise EnvironmentError("PINECONE_API_KEY and GROQ_API_KEY must be set in .env file.")
        return {
            "PINECONE_API_KEY": pinecone_key,
            "GROQ_API_KEY": groq_key
        }
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        raise
