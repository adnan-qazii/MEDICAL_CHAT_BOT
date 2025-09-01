from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings(model_name="all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)
