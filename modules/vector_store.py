from langchain_pinecone import PineconeVectorStore

def create_vector_store(index_name, embedding, documents):
    return PineconeVectorStore.from_documents(
        index_name=index_name,
        embedding=embedding,
        documents=documents
    )
