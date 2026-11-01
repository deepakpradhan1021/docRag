# 📄 docRag — Production-Ready Zero-Budget Document RAG System

[![CI Test Suite](https://github.com/deepakpradhan1021/docRag/actions/workflows/ci.yml/badge.svg)](https://github.com/deepakpradhan1021/docRag/actions)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%20Vector%20Store-green.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![docRag Live Application UI](static/docrag_live_preview.png)

> **A 100% zero-cost Document Intelligence & Retrieval-Augmented Generation (RAG) platform.**  
> Upload PDFs, cross-reference multiple documents in isolated sessions, and receive accurate answers with exact 1-indexed page citations and quoted excerpts — featuring prompt injection defense, browser-isolated chats, sliding-window rate limiting, and multi-model quota fallback.

---

## 🌟 Live Demo & Deployment

- **Live Production App (Render Free Tier):** [https://docrag-3gb4.onrender.com](https://docrag-3gb4.onrender.com)
- **Interactive API Documentation (Swagger):** [https://docrag-3gb4.onrender.com/docs](https://docrag-3gb4.onrender.com/docs)
- **GitHub Repository:** [https://github.com/deepakpradhan1021/docRag](https://github.com/deepakpradhan1021/docRag)

> **Free-Tier Backend Notice:**  
> Render's free tier sleeps during inactivity and may take up to 45 seconds to wake up on the first request. All sessions and uploaded documents are held ephemerally in RAM and reset upon container sleep or restart.

---

## ✨ Key Capabilities

- **📚 Multi-Document Workspaces:** Attach and cross-reference multiple PDFs inside a single chat session.
- **🎯 1-Indexed Page References:** Every answer cites the exact document page number alongside quoted source excerpts.
- **🛡️ Prompt Delimiters:** Document content is strictly fenced in untrusted XML delimiters to separate reference text from system instructions.
- **⚡ In-Memory Vector Search:** Pure Python hybrid retrieval runs ephemerally in RAM per session with no persistent database dependencies.
- **🪶 Ultra-Low Memory Footprint:** Tuned with single-threaded embeddings and active garbage collection to operate reliably within 512 MB RAM containers.
- **🚦 Production Hardening:** Equipped with an in-memory sliding window rate limiter (40 req/min/IP) and client-side retry recovery.

---

## 🏗️ Architecture

```
                          ┌────────────────────────┐
                          │   User Uploads PDF     │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │   PyMuPDF Extraction   │
                          │   • Clean text & pages │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  Recursive Chunking    │
                          │  • 1000 char / 200 ovlp│
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │   FAISS Vector Index   │
                          │   • In-memory CPU store│
                          └───────────┬────────────┘
                                      │
      ┌───────────────────────────────┴──────────────────────────────┐
      │                                                              │
      ▼                                                              ▼
┌─────────────────────────┐                            ┌─────────────────────────┐
│     User Asks Question  │                            │  Page Inspector Modal   │
└─────────────┬───────────┘                            └─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│ Similarity Retrieval    │
│ • Top-K relevant chunks │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Grounded Synthesis     │
│  • Gemini 2.5 Flash     │
│  • Citation extraction  │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│ Formatted Answer        │
│ • Verified citations    │
│ • Quoted text excerpts  │
└─────────────────────────┘
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/deepakpradhan1021/docRag.git
cd docRag

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Copy the example environment configuration:

```bash
cp .env.example .env
```

Add your Google Gemini API key to `.env` (obtainable for free at [Google AI Studio](https://aistudio.google.com/apikey)):

```ini
GOOGLE_API_KEY=AIzaSy...
```

*(You can also leave this blank and supply your key directly in the web UI Settings modal).*

### 3. Launch Application

**FastAPI Full-Stack Server (Recommended):**
```bash
python server.py
# Access UI at http://localhost:8000
```

**Streamlit Dashboard:**
```bash
streamlit run app.py
```

---

## 📊 Evaluation Benchmark

The repository includes a synthetic evaluation suite (`eval_benchmark.py`) to quantify retrieval precision and recall across multi-page ground-truth facts:

```bash
python eval_benchmark.py
```

**Benchmark Results:**
- **Top-1 Precision:** 80.0% (8/10 queries)
- **Recall@3 (Passage Hit Rate):** 90.0% (9/10 queries)
- **Execution Time:** < 0.3 seconds on standard CPU

---

## 🧪 Testing

Execute the complete automated test suite:

```bash
python -m unittest discover tests
```

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
