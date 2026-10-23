# 📄 docRag — Production-Ready Zero-Budget Document RAG System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%20Vector%20Store-green.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A 100% zero-cost, privacy-first Document Intelligence & Retrieval-Augmented Generation (RAG) platform.**  
> Upload PDFs, cross-reference multiple documents, and receive grounded answers with exact 1-indexed page citations — with prompt injection defense, multi-chat isolation, and automatic multi-model quota fallback.

---

## 🌟 Live Demo & Links

- **Live Public Web App (Instant Access):** [https://3b2aa06d934697.lhr.life](https://3b2aa06d934697.lhr.life)
- **GitHub Repository:** [https://github.com/deepakpradhan1021/docRag](https://github.com/deepakpradhan1021/docRag)
- **Netlify Team Projects:** [https://app.netlify.com/teams/deepakpradhan1612/projects](https://app.netlify.com/teams/deepakpradhan1612/projects)
- **Local Application Server:** `http://127.0.0.1:8000`

---

## 🎨 Professional Dark Aesthetic UI

Inspired by cutting-edge AI product design (**Zyricon** & **Nextmove**), docRag features:
- **Custom Color Palette:** Blended Mauve Orchid (`#C38EB4`), Lilac Blush (`#E1CBD7`), Serene Sky Blue (`#86A8CF`), Slate Ocean (`#26425A`), and Midnight Abyss (`#0B1420`).
- **3D Iridescent Orb:** Smooth floating radial glow sphere serving as the interactive welcome state.
- **Multi-Chat Sidebar:** Create, switch, and delete multiple isolated chat sessions with glowing neon pill indicators.
- **Multi-PDF Document Library:** Attach multiple PDFs to a single chat to compare findings across documents.
- **Document Page Inspector:** Inspect exact extracted text page by page with PyMuPDF.
- **One-Click Export:** Download clean, formatted Markdown Q&A transcripts with source excerpts.

---

## 🏗️ Architectural Flow

```
                      +-----------------------------+
                      |      User Uploads PDF       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Validate File & Limits    |
                      | (Size <= 10MB, Pages <= 50) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Page-by-Page Text Extract   |
                      | (PyMuPDF, 1-Indexed Pages)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Recursive Chunk Splitting |
                      |  (1000 chars, 200 overlap)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Local CPU Embeddings       |
                      |   (all-MiniLM-L6-v2)        |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | In-Memory FAISS VectorStore |
                      |    (Session-Isolated)       |
                      +--------------+--------------+
                                     |
    User Question                    v
   +--------------+      +---------------------------+
   |  Chat Input  +----->| Similarity Search (Top-4) |
   +--------------+      +-------------+-------------+
                                       |
                                       v
                         +-----------------------------+
                         | Format Grounded Prompt with |
                         | <untrusted_content> Tags    |
                         +-------------+---------------+
                                       |
                                       v
                         +-----------------------------+
                         | Multi-Model Fallback Engine |
                         | gemini-3.8-flash (Primary)  |
                         | gemini-3.5-flash-lite (1.5k)|
                         +-------------+---------------+
                                       |
                                       v
                         +-----------------------------+
                         | Citation Parser & Validator |
                         | (Maps [Source X] to Pages)  |
                         +-------------+---------------+
                                       |
                                       v
                         +-----------------------------+
                         |  Grounded Answer + Excerpts |
                         +-----------------------------+
```

---

## 🚀 Key Engineering Highlights

1. **Multi-Model Quota Fallback Engine:**
   - Free-tier Google AI Studio keys limit `gemini-3.8-flash` to 20 requests per day.
   - docRag implements an **automatic cascading fallback**:  
     `gemini-3.8-flash` &rarr; `gemini-3.5-flash-lite` (1,500 RPD) &rarr; `gemini-3.5-flash` &rarr; `gemini-2.5-flash-lite` &rarr; `gemini-2.5-flash`.
   - Never gets stranded by a single model's quota exhaustion.

2. **Smart Quota Conservation:**
   - Standalone questions bypass LLM query reformulation, saving 50% of your daily API quota.
   - Only ambiguous follow-up questions with pronouns (*"what did they do next?"*) trigger contextual query clarification.

3. **Strict Source Grounding & Anti-Hallucination:**
   - Answers cite `[Source X]` markers.
   - Code strictly validates citations against actual retrieved chunks and maps them to exact 1-indexed document pages.
   - Rejects hallucinated citations. If information is missing, returns:  
     *"The provided document does not contain enough information to answer this question."*

4. **Prompt Injection Defense:**
   - Document chunks are wrapped in strict `<untrusted_document_content>` XML-style boundary tags.
   - System prompt instructs the model to treat content as passive untrusted data, preventing adversarial override instructions.

5. **100% Zero Budget Stack:**
   - No paid vector databases (In-memory FAISS).
   - No paid embedding APIs (Local CPU Sentence-Transformers `all-MiniLM-L6-v2`).
   - Free LLM tier with fallback support.

---

## 💰 Zero-Budget Stack Comparison

| Component | Technology | Cost | Why Chosen |
|---|---|---|---|
| **Language** | Python 3.10 - 3.13 | $0 | Clean syntax, rich ecosystem, standard for AI engineering |
| **Backend API** | FastAPI + Uvicorn | $0 | High-performance asynchronous REST API with auto OpenAPI docs |
| **Frontend UI** | HTML5 / CSS3 / Vanilla JS | $0 | Fast, zero build step, dark aesthetic, interview-ready |
| **Alternative UI** | Streamlit | $0 | Quick Python-only prototyping interface |
| **PDF Extraction** | PyMuPDF (`pymupdf`) | $0 | 10x faster than PyPDF, accurate page boundary tracking |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) | $0 | Compact (~80MB), runs on CPU in milliseconds |
| **Vector Store** | In-Memory FAISS (`faiss-cpu`) | $0 | Zero infra overhead, instantaneous similarity search, isolated per session |
| **LLM Inference** | Google Gemini (`gemini-3.8-flash` / `3.5-flash-lite`) | $0 | Free tier via Google AI Studio |
| **Static Hosting** | Netlify (`netlify.toml`) | $0 | Instant worldwide CDN deployment for `static/` |

---

## 📁 Project Structure

```text
docRag/
├── server.py                       # FastAPI REST API & static asset server
├── app.py                          # Streamlit application (alternative UI)
├── netlify.toml                    # Netlify deployment & SPA routing config
├── requirements.txt                # Pinned production dependencies
├── .env.example                    # Local environment variables template
├── .gitignore                      # Git exclusion rules
├── static/                         # Custom Dark Web Interface
│   ├── index.html                  # Responsive layout & Zyricon/Nextmove UI
│   ├── style.css                   # Custom color palette & 3D floating orb styles
│   └── app.js                      # Multi-chat, drag-drop upload, export, fallback logic
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── config.py                   # Centralized configuration, models, & limits
│   ├── chat_manager.py             # Multi-chat & multi-document state manager
│   ├── pdf_processing.py           # PyMuPDF extraction, validation, & limits
│   ├── vector_store.py             # Chunking, local embeddings, & FAISS indexing
│   └── rag_pipeline.py             # Grounded prompting, model fallback cascade, citations
└── tests/
    ├── __init__.py
    ├── test_pdf_processing.py      # PDF parsing and validation tests
    ├── test_vector_store.py        # Chunking & FAISS retrieval tests
    ├── test_rag_pipeline.py        # Prompt grounding & citation validation tests
    ├── test_session_and_e2e.py     # Multi-document session & reset tests
    └── test_api_server.py          # FastAPI REST endpoint integration tests
```

---

## ⚙️ Key Configuration Settings

All settings are centralized in [`src/config.py`](src/config.py):

| Setting | Default | Rationale & Trade-off |
|---|---|---|
| `MAX_FILE_SIZE_MB` | `10 MB` | Prevents high memory consumption on free hosts. |
| `MAX_PAGES` | `50 pages` | Ensures local CPU embedding generation stays under ~20 seconds. |
| `CHUNK_SIZE` | `1000 chars` (~200 words) | Balances specific sentence retrieval with sufficient surrounding context. |
| `CHUNK_OVERLAP` | `200 chars` (20%) | Prevents critical facts from being split across chunk boundaries. |
| `RETRIEVAL_TOP_K` | `4 passages` | Supplies ~4,000 characters of evidence — compact, fast, and quota-friendly. |
| `LLM_TEMPERATURE` | `0.1` | Near-zero temperature minimizes creative hallucinations. |
| `CANDIDATE_MODELS` | `3.8-flash, 3.5-flash-lite, 3.5-flash` | Auto-cascading model list for zero downtime. |

---

## 🛠️ Quickstart & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/deepakpradhan1021/docRag.git
cd docRag
```

### 2. Create and activate a virtual environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your free Gemini API key
```bash
cp .env.example .env
```
Edit `.env` (or enter your key directly in the web UI Settings modal):
```ini
GOOGLE_API_KEY=AIzaSyYourActualKeyHere
```
*(Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) in 10 seconds)*.

### 5. Run the Application

#### Option A: Custom Dark Web Interface (FastAPI + Modern UI) — Recommended
```bash
python server.py
```
Open **`http://127.0.0.1:8000`** in your browser.

#### Option B: Classic Streamlit Interface
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Automated Test Suite (24 Tests Passing)

docRag includes 24 unit and API integration tests covering:
- Correct 1-indexed page extraction and empty/scanned PDF detection.
- Chunk splitting and FAISS CPU vector similarity search.
- Prompt injection boundary defense.
- Valid citation mapping (`[Source X]` to document name, page, and excerpt).
- Rejection of hallucinated citations.
- Missing API key and rate-limit error handling.
- FastAPI REST endpoints (`/api/chats`, `/api/chats/{id}/upload`, `/api/chats/{id}/message`, `/api/chats/{id}/export`, `/api/health`).

Run all tests:
```bash
python -m unittest discover tests -v
```

Expected result:
```text
Ran 24 tests in 20.1s

OK
```

---

## 🌐 Netlify Deployment Guide

### Deploying Frontend to Netlify:
1. Go to [Netlify Drop](https://app.netlify.com/drop) or link your GitHub repo at [app.netlify.com](https://app.netlify.com).
2. The included [`netlify.toml`](netlify.toml) configures `static/` as the publish directory automatically.
3. Open the deployed Netlify URL.
4. Click **Settings** (top right) in the docRag UI and enter your backend server URL (e.g. `http://localhost:8000` or a deployed FastAPI endpoint on Render/Railway).
5. Paste your free Google AI Studio API key and start exploring documents!

---

## 🎤 Interview Cheatsheet: How to Explain docRag to Interviewers

When discussing this project in a machine learning or full-stack software engineering interview, highlight these architectural decisions:

1. **Why RAG instead of Fine-Tuning?**
   - *Fine-tuning* teaches a model a new style or vocabulary, but cannot reliably recall exact page citations and is expensive to update.
   - *RAG* injects ground-truth document excerpts directly into the context window at inference time, guaranteeing 100% factual accuracy, zero retraining cost, and verifiable source page citations.

2. **Why In-Memory FAISS instead of Pinecone/Milvus?**
   - For document-level Q&A (PDFs under 50 pages), hosting an external vector database introduces network latency, recurring monthly costs, and security risks.
   - In-memory FAISS generates indexes in milliseconds on CPU, isolates indexes per user session, and leaves zero persistent trace on disk after the session ends.

3. **How is Prompt Injection Defended?**
   - User document text is treated as passive data by enclosing it inside `<untrusted_document_content>` tags.
   - The system instructions explicitly order the LLM to ignore any meta-instructions or prompt jailbreaks found within the extracted PDF text.

4. **How are Hallucinations Prevented?**
   - Temperature is locked to `0.1`.
   - The model is instructed to cite only `[Source X]` numbers present in the retrieved passages.
   - Python post-processing validates each citation ID against the chunk dictionary; fabricated source IDs are discarded in code.

5. **How is Quota Managed on Free Tier?**
   - The pipeline implements an automatic multi-model fallback cascade (`gemini-3.8-flash` &rarr; `gemini-3.5-flash-lite`).
   - Standalone questions bypass query reformulation to halve LLM API consumption.

---

## 📜 License

MIT License &copy; 2026 Deepak Pradhan. Free for personal, academic, and commercial use.
