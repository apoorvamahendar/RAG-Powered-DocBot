# RAG-Powered-DocBot

# 🧠 RAG AI Assistant 

A Retrieval-Augmented Generation (RAG) AI Assistant that allows you to upload documents (PDF, DOCX) and ask natural language questions about their contents — powered by **Groq API**, **LangChain**, and **ChromaDB**.  
Everything runs from a **single script: `main.py`**.

---

## 🚀 Features

- 📄 Upload and parse PDF or DOCX files
- 🧠 Embed documents and store them in a Chroma vector database
- 🔍 Retrieve relevant chunks using semantic search
- 💬 Ask questions and get context-aware answers using Groq LLMs
- ✅ All logic in a single file (`main.py`) — no extra modules needed

---

## 📁 Project Structure

```
rag_ai_assistant_fixed/
├── main.py             # All logic lives here (upload, embedding, RAG, Groq)
├── uploaded_docs/      # Folder for user-uploaded PDFs/DOCX files
├── rag_index/          # Chroma vector DB for document embeddings
├── .env                # Contains your Groq API key
└── requirements.txt    # Python dependencies
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-ai-assistant.git
cd rag-ai-assistant

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Usage

```bash
python main.py
```

Then upload your PDF/DOCX file when prompted and start chatting with your documents!

---

## 🧪 Example

> Upload: `python.pdf`  
> Ask: _"What is a Python decorator?"_  
> ✅ Instant answer from the document using Groq-powered RAG.

---

## 🧩 Built With

- [LangChain](https://github.com/langchain-ai/langchain)
- [Groq API](https://groq.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Python](https://www.python.org/)

---

## 📜 License

MIT License. See `LICENSE` file for more info.

---

## 🙌 Acknowledgements

Inspired by open-source RAG stacks and Groq LPU acceleration for blazing-fast inference.
