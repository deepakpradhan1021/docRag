# docRag — Lightweight Document Q&A with Grounded Citations

[![CI Test Suite](https://github.com/deepakpradhan1021/docRag/actions/workflows/ci.yml/badge.svg)](https://github.com/deepakpradhan1021/docRag/actions)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Google Gemini API](https://img.shields.io/badge/Google%20Gemini-API-orange.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight Document Question & Answering (RAG) web application built with **FastAPI**, **Streamlit**, and the **Google Gemini API**. It allows users to upload PDF documents, ask questions, and receive answers accompanied by 1-indexed page citations and quoted source snippets.

To run reliably on free-tier cloud containers (such as Render's 512 MB RAM limit) without out-of-memory errors, docRag uses an in-memory **Okapi BM25** keyword index combined with sublinear character n-grams instead of heavy local embedding models.

---

## 🌟 Live Demo & Deployment

- **Live Web App (Render Free Tier):** [https://docrag-3gb4.onrender.com](https://docrag-3gb4.onrender.com)
- **API Documentation (Swagger):** [https://docrag-3gb4.onrender.com/docs](https://docrag-3gb4.onrender.com/docs)
- **Repository:** [https://github.com/deepakpradhan1021/docRag](https://github.com/deepakpradhan1021/docRag)

> *Note on Free-Tier Hosting:* The Render instance spins down after periods of inactivity. Please allow 30–45 seconds for cold start on the first request. Uploaded documents and sessions reside ephemerally in RAM.

---

## ✨ Features

- **Multi-Document Sessions:** Upload and query multiple PDFs within the same conversation session.
- **1-Indexed Page Citations:** Every answer references the source document name, exact page number, and relevant excerpt.
- **Memory-Safe Architecture (<85 MB RAM):** Built with pure Python BM25 and scikit-learn TF-IDF n-grams to avoid PyTorch memory overhead.
- **Prompt Isolation:** Document passages are wrapped in strict XML delimiters to separate reference material from instruction prompts.
- **In-Memory Rate Limiting:** Sliding-window rate limiter (120 req/min per IP) to guard against API spam.
- **Dual Interfaces:** Full REST API served by FastAPI and an optional Streamlit dashboard.

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
                          │  BM25 In-Memory Index  │
                          │  • Lightweight (<85MB) │
                          └───────────┬────────────┘
                                      │
       ┌──────────────────────────────┴──────────────────────────────┐
       │                                                             │
       ▼                                                             ▼
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
│  • Google Gemini API    │
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

### 1. Clone the repository & create environment

```bash
git clone https://github.com/deepakpradhan1021/docRag.git
cd docRag

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Add your Google Gemini API key to `.env` (available at [Google AI Studio](https://aistudio.google.com/apikey)):

```ini
GOOGLE_API_KEY=your_gemini_api_key_here
```

*(You can also leave `.env` empty and enter your key in the web UI settings modal).*

### 3. Run the application

**FastAPI Server (Recommended):**
```bash
python server.py
# Open your browser at http://localhost:8000
```

**Streamlit Dashboard:**
```bash
streamlit run app.py
```

---

## 📊 Evaluation Benchmark

The repository includes a synthetic evaluation script (`eval_benchmark.py`) to test retrieval precision and recall over multi-page documents:

```bash
python eval_benchmark.py
```

**Synthetic Benchmark Results (10-query test on 10-page document):**
- **Top-1 Precision:** 80.0% (8/10 queries)
- **Recall@3 (Passage Hit Rate):** 90.0% (9/10 queries)
- **Execution Time:** < 0.3 seconds on CPU

---

## ⚠️ Known Limitations & Design Decisions

- **Keyword vs. Dense Semantic Matching:** To guarantee zero OOM crashes under 512 MB RAM, docRag uses BM25 and character n-grams instead of deep transformer models. Queries with zero lexical overlap with the document (e.g. asking *"How much money did the company make?"* when the text only says *"gross revenues"*) may rank lower than exact keyword matches.
- **Ephemeral Storage:** Document indices and chat histories live in RAM and are cleared when the container restarts.
- **Cold Starts:** Free-tier cloud instances sleep when inactive and require ~30–45 seconds to wake up on the first request.

---

## 🧪 Testing

```bash
# Run tests with pytest
pytest tests/ -v

# Or run with standard library unittest
python -m unittest discover tests
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
