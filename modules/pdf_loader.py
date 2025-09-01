from langchain.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdf(data_path):
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f'Loaded {len(documents)} document(s).')
    return documents
