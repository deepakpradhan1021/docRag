# 📄 docRag

> **Zero-cost, grounded Document Question Answering with source page citations.**  
> Built with Streamlit, LangChain, FAISS, Sentence Transformers, and Google Gemini.

---

## 🌟 Overview

**docRag** allows you to upload any text-based PDF, ask questions about its content, and receive concise answers grounded strictly in the document text. Every factual response includes **validated source citations** displaying the document name, page number, and supporting excerpt.

---

## 🏗️ Architecture & Pipeline Flow

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
                         |  Google Gemini (Free-Tier)  |
                         |   (gemini-2.0-flash, T=0.1) |
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

## 🚀 Key Features (Version 1 Scope)

1. **Single PDF Upload:** Validates file signatures, file size (up to 10 MB), and page count (up to 50 pages).
2. **Page-by-Page Extraction:** Preserves exact page numbers (1-indexed) across the entire pipeline.
3. **Smart Chunking:** Text is split using `RecursiveCharacterTextSplitter` with configurable chunk size and overlap.
4. **100% Local Embeddings:** Vectors are created locally using `all-MiniLM-L6-v2` via CPU without calling paid APIs.
5. **In-Memory FAISS Indexing:** Fast vector similarity search with no external vector database servers or disk persistence.
6. **Prompt Injection Defense:** Reference excerpts are wrapped in `<untrusted_document_content>` tags and treated as reference text rather than executable instructions.
7. **Strict Grounding:** The LLM is instructed to answer *only* from retrieved passages and state *"The provided document does not contain enough information to answer this question"* when absent.
8. **Validated Citations:** Answers cite `[Source X]` markers, which Python code validates against actual retrieved chunks and renders with the page number and original excerpt.
9. **Follow-Up Query Reformulation:** Clarifies contextual follow-up questions into standalone search queries without treating previous assistant answers as ground-truth evidence.
10. **Session Isolation & Reset:** Data resides strictly within the active browser session (`st.session_state`). Clicking **Reset** instantly purges the document, index, and chat history.
11. **Graceful Error Handling:** Detects scanned/empty PDFs, invalid headers, missing API keys, and rate limits cleanly.

---

## 💰 Zero-Budget Stack

| Layer | Technology | Cost | Why Chosen |
|---|---|---|---|
| **Language** | Python 3.10 - 3.13 | Free / Open-source | Industry standard for AI & RAG |
| **Interface** | Streamlit | Free / Open-source | Clean, native Python web UI |
| **PDF Extraction** | PyMuPDF (`pymupdf`) | Free / Open-source | Fast, lightweight C engine, page-level accuracy |
| **RAG Pipeline** | LangChain Core & Text Splitters | Free / Open-source | Standardized document abstractions |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) | Free / Open-source | Tiny (~80MB), runs on CPU in milliseconds |
| **Vector Index** | FAISS CPU (`faiss-cpu`) | Free / Open-source | In-memory, zero infrastructure needed |
| **LLM Provider** | Google Gemini Free Tier (`gemini-2.0-flash`) | $0 / Free Tier | Free API via Google AI Studio |
| **Deployment** | Streamlit Community Cloud | $0 / Free Tier | Instant GitHub-linked deployment |

---

## 📁 Project Structure

```
docRag/
├── app.py                          # Streamlit web application & session state
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── config.py                   # Central settings & trade-offs
│   ├── pdf_processing.py           # Extraction, validation, & error classes
│   ├── vector_store.py             # Chunking, embeddings, & FAISS retrieval
│   └── rag_pipeline.py             # Prompting, Gemini LLM, & citation validation
├── tests/
│   ├── __init__.py
│   ├── test_pdf_processing.py      # PDF validation & extraction tests
│   ├── test_vector_store.py        # Chunking & FAISS retrieval tests
│   ├── test_rag_pipeline.py        # Prompt grounding & citation parsing tests
│   └── test_session_and_e2e.py     # End-to-end integration & reset tests
├── .streamlit/
│   └── secrets.toml.example        # Cloud secrets template
├── .env.example                    # Local environment variables template
├── .gitignore                      # Git exclusion rules
├── requirements.txt                # Pinned dependency requirements
└── README.md                       # Documentation
```

---

## ⚙️ Configuration & Trade-Offs

All settings are centralized in [`src/config.py`](src/config.py):

| Setting | Default | Rationale & Trade-off |
|---|---|---|
| `MAX_FILE_SIZE_MB` | `10` | Prevents high memory consumption on free-tier hosting. Text PDFs are usually < 5 MB. |
| `MAX_PAGES` | `50` | Keeps local CPU embedding generation under 30 seconds on standard machines. |
| `CHUNK_SIZE` | `1000` (~200 words) | Balances specific sentence retrieval against providing enough surrounding context for the LLM. |
| `CHUNK_OVERLAP` | `200` (20%) | Prevents key sentences from being split in half across chunk boundaries. |
| `RETRIEVAL_TOP_K` | `4` passages | Supplies ~4000 characters of evidence — compact enough to minimize prompt latency and quota. |
| `LLM_TEMPERATURE` | `0.1` | Low temperature ensures factual, deterministic responses and reduces hallucinations. |
| `MAX_QUESTIONS_PER_SESSION` | `20` | Protects free-tier API quotas against excessive requests. |

---

## 🔒 Privacy & Security Model

- **API Keys:** Kept server-side. Local development reads from `.env` (ignored by Git). Cloud deployment reads from `st.secrets`. Users may also input a session-specific key via the UI.
- **Session Isolation:** Document bytes, extracted text, and the FAISS vector store live entirely in memory inside the user's `st.session_state`. They are **never** shared across users or persisted to disk.
- **No Deserialization Attacks:** docRag never loads pre-built or untrusted pickled FAISS indices from disk. Indexes are generated directly from verified uploaded bytes.
- **External Transmission Notice:** When a user asks a question, the top relevant excerpts are transmitted to the Google Gemini API for answer generation.
- **Free Tier Policy Disclaimer:** On Google Gemini's free tier, Google may retain and analyze prompt inputs to improve future models. Do not upload classified, confidential, or proprietary material.

---

## 🛠️ Local Setup Guide

### 1. Prerequisites
- Python 3.10 - 3.13
- Git
- A free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/apikey)

### 2. Clone and navigate to the project
```bash
git clone <your-repo-url> docRag
cd docRag
```

### 3. Create and activate a virtual environment
**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure your API key
Copy `.env.example` to `.env` and insert your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GOOGLE_API_KEY=AIzaSyYourActualKeyHere
```
*(Alternatively, you can skip this step and enter your key directly into the sidebar input field in the web app).*

### 6. Run the application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running the Automated Tests

docRag includes 20 comprehensive unit and integration tests covering extraction, page metadata, citation parsing, hallucination rejection, and error fallbacks.

Run the entire test suite:
```bash
python -m unittest discover tests -v
```

Expected output:
```
test_empty_bytes (tests.test_pdf_processing.TestPDFProcessing.test_empty_bytes) ... ok
test_empty_or_scanned_pdf (tests.test_pdf_processing.TestPDFProcessing.test_empty_or_scanned_pdf) ... ok
test_extract_valid_multipage_pdf (tests.test_pdf_processing.TestPDFProcessing.test_extract_valid_multipage_pdf) ... ok
...
Ran 20 tests in 16.8s

OK
```

---

## ☁️ Free Deployment to Streamlit Community Cloud

Streamlit Community Cloud offers **100% free hosting** for public GitHub repositories with 1 GB of RAM and 1 CPU core, which fully meets docRag's lightweight footprint.

### Step-by-Step Deployment:
1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "feat: complete docRag v1 implementation"
   git branch -M main
   git remote add origin https://github.com/<your-username>/docRag.git
   git push -u origin main
   ```
2. **Sign in to Streamlit Cloud:**
   Visit [share.streamlit.io](https://share.streamlit.io) and authenticate with your GitHub account.
3. **Deploy New App:**
   - Click **"New app"**.
   - Select your repository (`<your-username>/docRag`), branch (`main`), and set Main file path to `app.py`.
4. **Configure Secrets (API Key):**
   - Click **"Advanced settings..."** before deploying (or under App Settings -> Secrets later).
   - In the **Secrets** text box, add:
     ```toml
     GOOGLE_API_KEY = "AIzaSyYourActualKeyHere"
     ```
   - Click **Save**.
5. **Launch!**
   - Click **Deploy!**
   - Your app will be live with a public URL in 2-3 minutes.

---

## 📜 License

MIT License. Free for personal, academic, and commercial use.
