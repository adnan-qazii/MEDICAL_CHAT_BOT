from pinecone import Pinecone, ServerlessSpec

def setup_pinecone(api_key, index_name, dimension=384):
    pc = Pinecone(api_key=api_key)
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(region="us-east-1", cloud="aws")
    )
    return pc
