import logging
from flask import Flask, request, render_template, session, redirect, url_for
from modules.config import load_env
from modules.pdf_loader import load_pdf
from modules.text_splitter import split_documents
from modules.embeddings import get_embeddings
from modules.pinecone_setup import setup_pinecone
from modules.vector_store import create_vector_store
from modules.chain import get_chain
from langchain_groq import ChatGroq
import os
from typing import List, Tuple, Optional
from datetime import datetime, timedelta, timezone
import pytz

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components once
def initialize_components() -> Tuple:
    env = load_env()
    index_name = "medical-chatbot"
    try:
        pc = setup_pinecone(env["PINECONE_API_KEY"], index_name)
    except Exception as e:
        logger.error(f"Pinecone setup failed: {e}")
        raise
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    try:
        embedding = get_embeddings()
    except Exception as e:
        logger.error(f"Embedding setup failed: {e}")
        raise
    try:
        model = ChatGroq(model="Gemma2-9b-It", groq_api_key=env["GROQ_API_KEY"])
    except Exception as e:
        logger.error(f"LLM setup failed: {e}")
        raise
    if index_name in existing_indexes:
        from langchain_pinecone import PineconeVectorStore
        docsearch = PineconeVectorStore(
            index_name=index_name,
            embedding=embedding
        )
    else:
        data_path = "data/"
        try:
            documents = load_pdf(data_path)
            texts = split_documents(documents)
            docsearch = create_vector_store(index_name, embedding, texts)
        except Exception as e:
            logger.error(f"PDF/vector store setup failed: {e}")
            raise
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = get_chain(model, retriever)
    return qa_chain, model

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

qa_chain, model = initialize_components()

@app.route('/', methods=['GET', 'POST'])
def chat() -> str:
    if 'chat_history' not in session:
        session['chat_history'] = []
    answer: Optional[str] = None
    if request.method == 'POST':
        user_input: str = request.form['question'].strip()
        user_input_lower: str = user_input.lower()
        # Get current time in IST
        ist = pytz.timezone('Asia/Kolkata')
        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        now_ist = now_utc.astimezone(ist)
        timestamp = now_ist.strftime('%I:%M %p')
        try:
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
                    "chat_history": [(msg[0], msg[1]) for msg in session['chat_history']]
                })
                if (not response['answer']) or ("does not contain the answer" in response['answer'].lower()):
                    answer = "I don't know, but I'm always here to help!"
                else:
                    answer = response['answer']
            # Store only the main answer text and IST time in chat_history
            session['chat_history'].append((user_input, answer, timestamp))
            session.modified = True
        except Exception as e:
            logger.error(f"Error during chat processing: {e}")
            answer = "Sorry, something went wrong. Please try again later."
        return redirect(url_for('chat'))
    return render_template('chat.html', chat_history=session.get('chat_history', []))

@app.route('/reset')
def reset_chat() -> str:
    session['chat_history'] = []
    return redirect(url_for('chat'))

if __name__ == "__main__":
    app.run(debug=True)
