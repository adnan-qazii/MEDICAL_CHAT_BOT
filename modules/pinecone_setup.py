from pinecone import Pinecone, ServerlessSpec

def setup_pinecone(api_key, index_name, dimension=384):
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
