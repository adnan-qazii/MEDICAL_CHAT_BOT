from langchain.chains import ConversationalRetrievalChain

def get_chain(model, retriever):
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever
    )
