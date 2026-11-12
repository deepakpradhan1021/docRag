# docRag — Intelligent AI Document Assistant & Grounded RAG

[![CI Test Suite](https://github.com/deepakpradhan1021/docRag/actions/workflows/ci.yml/badge.svg)](https://github.com/deepakpradhan1021/docRag/actions)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Google Gemini API](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash%20Lite-orange.svg)](https://aistudio.google.com/)
[![Render Status](https://img.shields.io/badge/Live%20App-Render-46E3B7.svg)](https://docrag-3gb4.onrender.com/)
[![Netlify Status](https://img.shields.io/badge/Frontend%20Mirror-Netlify-00C7B7.svg)](https://docrag-ai.netlify.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Live Applications

- **Live Web App (Primary Frontend on Netlify):** [https://docrag-ai.netlify.app/](https://docrag-ai.netlify.app/)
- **Live Cloud Backend API (Render):** [https://docrag-3gb4.onrender.com](https://docrag-3gb4.onrender.com)
- **Interactive API Documentation (Swagger):** [https://docrag-3gb4.onrender.com/docs](https://docrag-3gb4.onrender.com/docs)
- **Backend Health Check:** [https://docrag-3gb4.onrender.com/api/health](https://docrag-3gb4.onrender.com/api/health)
- **GitHub Repository:** [https://github.com/deepakpradhan1021/docRag](https://github.com/deepakpradhan1021/docRag)

> [!NOTE]
> **Free-Tier Wake-up Notice:** The Render backend automatically sleeps after 15 minutes of inactivity. If visiting after a period of idle time, the first request may take ~30–45 seconds to spin up. All subsequent requests respond instantaneously.

---

## 📖 About docRag

**docRag** is an intuitive, production-grade AI document assistant engineered to transform dense PDFs into actionable intelligence. Unlike traditional rigid search engines or bare-bones LLM wrappers, docRag is designed for real-world usability: users can upload documents and ask natural, everyday questions without needing any background in prompt engineering.

### Why docRag?
Most naive RAG implementations suffer from common failure modes:
1. **The Literal Keyword Trap:** Asking simple prompts like `"summary"` or `"overview"` fails because BM25 searches for the word "summary" instead of synthesizing document content.
2. **Vocabulary Mismatches:** A user asking *"What is ML?"* gets no answers if the document only writes *"Machine learning"*, or asking for *"advantages"* fails if the text uses *"benefits"*.
3. **Heavy Infrastructure Costs:** Standard RAG architectures frequently rely on multi-gigabyte embedding models and dedicated vector databases that crash on free-tier servers or incur significant monthly fees.

**docRag solves these challenges:**
* **Intent-Aware Summarization:** Automatically identifies summary and overview requests (`summary`, `summaries`, `key points`, `notes`, `what is this about?`) and retrieves representative passages across the beginning, middle, and end of the document.
* **Semantic & Acronym Expansion:** Automatically bridges vocabulary gaps (e.g., expanding `ML` $\rightarrow$ `Machine Learning`, `CSAT` $\rightarrow$ `Customer Satisfaction`, `QA` $\rightarrow$ `Quality Assurance`).
* **Sublinear Character N-Gram Retrieval:** Tolerates misspellings, typos, and variations using character-level n-grams `(2, 4)`.
* **Zero-Cost & Memory-Safe:** Runs comfortably under 85 MB RAM—guaranteeing zero out-of-memory crashes on free cloud tiers.
* **Verifiable Evidence:** Every factual statement is paired with strict 1-indexed page citations `[Source X | Page Y]` accompanied by exact source excerpts.

---

## ✨ Key Features

- **🎨 Modern ChatGPT-Style UI & 3D Animated Hero:** Futuristic space-inspired dark workspace with a rotating 3D geometric crystal, three-dot options dropdown menu, instant optimistic New Chat, syntax-styled code blocks with 1-click copy, and full mobile drawer navigation.
- **🎯 Exact 1-Indexed Page Citations:** Every claim links directly to its source document, page number, and quoted excerpt for instant verification.
- **✨ 1-Click "Try Asking" Suggestion Chips:** Automatically surfaces 8 quick-start prompts (Executive Summary, Main Topics, In Simple Words, Definitions, Key Points, Important Concepts, Pros & Cons, Study Notes) on upload.
- **📊 Table & Numerical Data Extraction:** Uses PyMuPDF's layout analyzer (`page.find_tables()`) to convert complex tabular data into clean Markdown tables for accurate mathematical and SLA queries.
- **🧠 Free-Tier Gemini Model Cascade:** Defaulted to Google's official free model `gemini-3.5-flash-lite`, with automatic fallback to `gemini-2.0-flash` and `gemini-1.5-flash` during transient quota spikes.
- **🛡️ Prompt Injection & Hallucination Defense:** Passages are wrapped in untrusted boundary tags, instructing the model to reject embedded instructions and abstain politely when questions fall outside the document scope.
- **💬 Conversational Context & Follow-Ups:** Intelligently reformulates follow-up queries containing pronouns (*"What are its types?"*, *"Who is their lead?"*) using conversation history.
- **🔒 In-Memory Sliding-Window Rate Limiting:** Built-in rate limiter (120 req/min per IP) protects public deployments against DDoS and scraping abuse.

---

## 🏗️ System Architecture

```
                          ┌────────────────────────┐
                          │   User Uploads PDF     │
                          │   (Web or Mobile UI)   │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │   PyMuPDF Extraction   │
                          │   • Clean text & pages │
                          │   • Markdown tables    │
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
                          │  • Sublinear n-grams   │
                          │  • Memory safe (<85MB) │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │   User Asks Question   │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │   Intent & Expansion   │
                          │ • Summary detection    │
                          │ • Acronym / Synonyms   │
                          │ • History reformulation│
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │ Hybrid Pass Retrieval  │
                          │ • Representative chunks│
                          │ • Top-K relevant chunks│
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │ Grounded Synthesis     │
                          │ • Gemini 3.5 Flash Lite│
                          │ • Auto-fallback cascade│
                          │ • Citation validation  │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │ Validated Response     │
                          │ • Verified citations   │
                          │ • Verbatim quotes      │
                          └────────────────────────┘
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository

```bash
git clone https://github.com/deepakpradhan1021/docRag.git
cd docRag
```

### 2. Set up virtual environment & dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory (or copy from `.env.example`):

```bash
cp .env.example .env
```

Add your Google Gemini API key (obtainable free from [Google AI Studio](https://aistudio.google.com/apikey)):

```ini
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

### 4. Run the Server

```bash
python server.py
```

Open your browser at `http://localhost:8000` to interact with the local UI, or view interactive API documentation at `http://localhost:8000/docs`.

---

## 📊 Evaluation & Verification Results

The pipeline is verified with reproducible end-to-end integration tests covering 12 evaluation scenarios:

| Scenario | Input Query | Expected Behavior | Result |
| :--- | :--- | :--- | :---: |
| **Document Summary** | `"summary"` | Generates structured overview with page citations | **PASS** |
| **Plural Summary** | `"summaries"` | Synthesizes document across all sections | **PASS** |
| **Overview Intent** | *"What is this document about?"* | Captures core purpose and scope | **PASS** |
| **Topic Explanation**| *"Explain the main topic."* | Focuses on central subject | **PASS** |
| **Key Points** | *"Give me five important points."* | Structured bullet points citing sources | **PASS** |
| **Acronym Query** | *"What is ML?"* | Expands to Machine Learning and retrieves answer | **PASS** |
| **Synonym Match** | *"What are the advantages?"* | Matches sections describing "benefits" / "pros" | **PASS** |
| **Simplified Language**| *"Explain this in simple words."* | Plain language suitable for non-experts | **PASS** |
| **Fuzzy Typo Query** | *"macheen lerning"* | Character n-grams retrieve correct passage | **PASS** |
| **Absent Topic** | *"What is quantum entanglement?"* | Polite explanation of what the document actually covers | **PASS** |
| **Follow-up Pronoun**| *"What are its types?"* | Contextual reformulation using chat history | **PASS** |
| **Prompt Injection** | *"Ignore previous instructions..."* | Treats untrusted document text as raw data | **PASS** |

---

## 🧪 Running Automated Tests

```bash
# Run complete test suite (45 unit & integration tests)
pytest tests/ -v
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
