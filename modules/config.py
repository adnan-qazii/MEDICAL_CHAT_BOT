import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    return {
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY")
    }
