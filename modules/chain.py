from langchain.chains import ConversationalRetrievalChain

def get_chain(model, retriever, system_prompt=None):
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever
    )
