# 🩺 Medical Chatbot

![Medical Chatbot Banner](https://img.freepik.com/free-vector/medical-chatbot-concept-illustration_114360-10970.jpg)

Welcome to the **Medical Chatbot** project! This intelligent assistant leverages state-of-the-art language models and vector search to answer medical questions using your own PDF data. Built with modular Python code and a beautiful web interface powered by Flask.

---

## 🌟 Features
- **Conversational AI**: Ask medical questions and get instant, relevant answers.
- **PDF Knowledge Base**: Uses your own medical PDFs for context-rich responses.
- **Vector Search**: Fast, semantic retrieval using Pinecone and HuggingFace embeddings.
- **Web Interface**: Simple, user-friendly chat powered by Flask.
- **Modular Code**: Clean, maintainable Python modules for every step.

---

## 🗂️ Project Structure
```
MEDICAL_CHAT_BOT/
│
├── app.py                  # Flask web app (main entry)
├── requirements.txt        # All required packages
├── setup.py                # (optional) For packaging
├── data/
│   └── Medical_book.pdf    # Your medical data
├── modules/
│   ├── config.py           # Environment/config loading
│   ├── pdf_loader.py       # PDF loading functions
│   ├── text_splitter.py    # Text splitting functions
│   ├── embeddings.py       # Embedding functions
│   ├── pinecone_setup.py   # Pinecone index setup
│   ├── vector_store.py     # Vector store functions
│   └── chain.py            # Conversational chain logic
└── notebooks/
    └── try.ipynb           # Your notebook
```

---

## 🎨 Screenshots
![Chatbot UI](https://img.freepik.com/free-vector/medical-chatbot-concept-illustration_114360-10970.jpg)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
https://github.com/adnan-qazii/MEDICAL_CHAT_BOT.git
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory:
```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4. Add Your PDF Data
Place your medical PDFs in the `data/` folder.

### 5. Run the Web App
```bash
python app.py
```
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🧩 How It Works
1. **PDF Loader**: Loads all PDFs from the `data/` folder.
2. **Text Splitter**: Splits documents into manageable chunks.
3. **Embeddings**: Converts text chunks into vector embeddings.
4. **Pinecone Vector Store**: Stores and retrieves chunks using semantic search.
5. **Conversational Chain**: Uses a language model to answer user questions based on retrieved chunks.
6. **Flask Web UI**: Provides a simple chat interface for users.

---

## 🛠️ Modular Code Example
```python
# modules/pdf_loader.py
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdf(data_path):
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
```

---

## 📦 Requirements
- Python 3.8+
- Flask
- langchain
- langchain_groq
- langchain_pinecone
- pinecone-client
- python-dotenv

---

## 💡 Customization
- Add more PDFs to `data/` for a richer knowledge base.
- Tweak chunk size and overlap in `text_splitter.py` for best results.
- Change the language model in `app.py` for different LLMs.

---

## 🖌️ Credits & License
- Built by [adnan-qazii](https://github.com/adnan-qazii)
- Free to use for educational and research purposes.

---

## 🌈 Enjoy your Medical Chatbot!
