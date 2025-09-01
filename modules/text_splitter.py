from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=500, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_documents(documents)
    print(f'Split into {len(texts)} chunks.')
    return texts
