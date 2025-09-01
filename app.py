from flask import Flask, request, render_template
from modules.config import load_env
from modules.pdf_loader import load_pdf
from modules.text_splitter import split_documents
from modules.embeddings import get_embeddings
from modules.pinecone_setup import setup_pinecone
from modules.vector_store import create_vector_store
from modules.chain import get_chain
from langchain_groq import ChatGroq
import os

app = Flask(__name__)

# Initialize components once s
env = load_env()
index_name = "medical-chatbot"
pc = setup_pinecone(env["PINECONE_API_KEY"], index_name)
existing_indexes = [idx.name for idx in pc.list_indexes()]
embedding = get_embeddings()
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=env["GROQ_API_KEY"])
if index_name in existing_indexes:
    # Use existing vector store, skip all data processing
    from langchain_pinecone import PineconeVectorStore
    docsearch = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding
    )
else:
    # Run data processing and create new vector store
    data_path = "data/"
    documents = load_pdf(data_path)
    texts = split_documents(documents)
    docsearch = create_vector_store(index_name, embedding, texts)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa_chain = get_chain(model, retriever)
chat_history = []

SYSTEM_PROMPT = (
    "You are a good medical assistant. Your name is Zoya. You answer medical questions based on the given information, "
    "but you are also friendly and can chat socially. If a question is out of data, reply with something beautiful like: "
    "I don't know, but I'm always here to help! If someone greets you or asks something social, respond warmly as Zoya."
)

SOCIAL_KEYWORDS = ["hi", "hello", "hey", "how are you", "who are you", "good morning", "good evening", "good night","hlo"]

SOCIAL_PROMPT = (
    "You are Zoya, a professional and friendly medical assistant. When users greet you or ask social questions, respond warmly and politely, "
    "maintaining a helpful and caring tone. Always introduce yourself as Zoya and encourage users to ask medical questions if they need help. "
    "Keep your response short, friendly, and professional. Only reply with a single sentence."
)

@app.route('/', methods=['GET', 'POST'])
def chat():
    global chat_history
    answer = None
    if request.method == 'POST':
        user_input = request.form['question'].strip()
        user_input_lower = user_input.lower()
        # Check for social interaction
        if any(keyword in user_input_lower for keyword in SOCIAL_KEYWORDS):
            social_prompt = (
                f"{SOCIAL_PROMPT}\nUser: {user_input}\n"
                "Respond as Zoya, the friendly medical assistant, in a professional, caring, and polite manner."
            )
            response = model.invoke(social_prompt)
            main_answer = response if isinstance(response, str) else getattr(response, 'content', str(response))
            answer = main_answer if main_answer else "I'm Zoya, your friendly medical assistant! How can I help you today?"
        else:
            prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}"
            response = qa_chain({
                "question": prompt,
                "chat_history": chat_history
            })
            # Fallback if no relevant answer
            if (not response['answer']) or ("does not contain the answer" in response['answer'].lower()):
                answer = "I don't know, but I'm always here to help!"
            else:
                answer = response['answer']
        # Store only the main answer text in chat_history
        chat_history.append((user_input, answer))
    return render_template('chat.html', chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
