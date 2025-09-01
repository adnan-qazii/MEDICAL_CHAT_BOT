from flask import Flask, request, render_template_string
from modules.config import load_env
from modules.pdf_loader import load_pdf
from modules.text_splitter import split_documents
from modules.embeddings import get_embeddings
from modules.pinecone_setup import setup_pinecone
from modules.vector_store import create_vector_store
from modules.chain import get_chain
from langchain_groq import ChatGroq

app = Flask(__name__)

# Initialize components once
env = load_env()
data_path = "data/"
index_name = "medical-chatbot"
documents = load_pdf(data_path)
texts = split_documents(documents)
embedding = get_embeddings()
pc = setup_pinecone(env["PINECONE_API_KEY"], index_name)
docsearch = create_vector_store(index_name, embedding, texts)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=env["GROQ_API_KEY"])
qa_chain = get_chain(model, retriever)
chat_history = []

HTML = '''
<!doctype html>
<title>Medical Chatbot</title>
<h2>Ask a medical question:</h2>
<form method=post>
  <input name=question style="width:300px">
  <input type=submit value=Ask>
</form>
{% if answer %}
  <h3>Answer:</h3>
  <div style="background:#f0f0f0;padding:10px">{{ answer }}</div>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        user_input = request.form['question']
        response = qa_chain({
            "question": user_input,
            "chat_history": chat_history
        })
        answer = response['answer']
        chat_history.append((user_input, answer))
    return render_template_string(HTML, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
