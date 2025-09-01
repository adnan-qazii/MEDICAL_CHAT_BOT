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
print("Initializing components...")
env = load_env()
print("Environment variables loaded.")
data_path = "data/"
print("Data path set.")
index_name = "medical-chatbot-2"
print(f"Index name set to {index_name}.")
documents = load_pdf(data_path)
print(f"Loaded {len(documents)} documents.")
texts = split_documents(documents)
print(f"Split documents into {len(texts)} text chunks.")
embedding = get_embeddings()
print(f"Generated embeddings for {len(texts)} text chunks.")
pc = setup_pinecone(env["PINECONE_API_KEY"], index_name)
print("Pinecone setup complete.")
docsearch = create_vector_store(index_name, embedding, texts)
print("Vector store created.")
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
print("Retriever created.")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=env["GROQ_API_KEY"])
print("Language model initialized.")
qa_chain = get_chain(model, retriever)
print("QA chain created.")
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
