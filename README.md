# 📘 Context-Aware Study Assistant

An AI-powered exam preparation tool that reads your personal notes, understands them, and helps you revise smarter. Built with Retrieval-Augmented Generation (RAG), FAISS, and GPT-4.

---

## 🎯 Project Objective

The Context-Aware Study Assistant enables students to ask questions and receive answers **directly from their own uploaded PDFs** — class notes, study guides, or reference material — using AI.

It mimics how a smart study partner would revise with you: by finding the right material from your notes and answering your questions with precision.

---

## 🛠️ Tech Stack

| Component           | Technology              |
|--------------------|-------------------------|
| Language Model      | OpenAI GPT-4 / GPT-4o   |
| Embedding Model     | `text-embedding-3-small`|
| Vector Database     | FAISS (Flat Index)      |
| PDF Parser          | PyMuPDF                 |
| Backend             | Python 3.10+            |
| Environment Config  | python-dotenv           |

---

## 🧠 How It Works

1. 📂 Drop 1 or more PDFs into the `data/` folder  
2. 🧩 Text is extracted, chunked, and embedded via OpenAI  
3. 🔍 FAISS searches for the most relevant chunks based on your question  
4. 🤖 GPT-4 generates a natural-language answer grounded in your notes  
5. 🧾 Metadata (like document name and chunk ID) is tracked for traceability

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/context-aware-study-assistant.git
cd context-aware-study-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Create a `.env` file in the root:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Run the assistant

```bash
python app.py
```

Ask any question based on your uploaded notes!

---

## 💬 Example Interaction

```
Ask a question: What is the difference between supervised and unsupervised learning?

Answer:
Supervised learning involves labeled data...
(Source: ML_notes.pdf – Chunk 3)
```

---

## 📂 Project Structure

```
.
├── data/notes.pdf        # Your study PDFs go here
├── app.py                # Main CLI app
├── embedder.py           # Embedding + chunking
├── retriever.py          # Top-k retrieval logic
├── vector_store.py       # FAISS vector index management
├── qa_engine.py          # GPT-4 response generation
├── utils.py              # PDF parsing, helpers
└── .env                  # Your API keys (not committed)
```

---

## 📈 Future Enhancements

- [ ] Streamlit UI for real-time Q&A  
- [ ] Local LLM support (e.g. Mistral or LLaMA)  
- [ ] Caching and persistent vector store  
- [ ] Highlighted citations from source PDFs  
- [ ] Multi-format support (`.docx`, `.txt`, `.md`)
