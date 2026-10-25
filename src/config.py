"""
docRag Configuration
====================
All tunable settings live here. This is the single place to adjust limits,
model choices, and chunking parameters. Every setting has a comment
explaining what it does and why the default was chosen.
"""

import os
from dotenv import load_dotenv

# Load .env file for local development (ignored in production).
# Streamlit Community Cloud uses st.secrets instead.
load_dotenv()

# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------
# We use Google Gemini's free tier via the Google AI (non-Vertex) endpoint.
# The API key is loaded from runtime input, Streamlit secrets, or .env.
# Locally: set it in .env     |  Deployed: set it in Streamlit secrets.
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")


def get_gemini_api_key(override_key: str | None = None) -> str:
    """
    Resolve Google API key in priority order:
    1. Explicit runtime override (e.g. from UI settings input)
    2. Streamlit secrets (for Streamlit Community Cloud)
    3. OS environment variable / .env file
    """
    if override_key and override_key.strip():
        k = override_key.strip()
        if k not in ("your-api-key-here", "your_gemini_api_key_here", "AIzaSy..."):
            return k

    # Try Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            secret_val = str(st.secrets["GOOGLE_API_KEY"]).strip()
            if secret_val and secret_val not in ("your-api-key-here", "your_gemini_api_key_here"):
                return secret_val
    except Exception:
        pass

    # Fallback to environment variable
    env_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if env_key in ("your-api-key-here", "your_gemini_api_key_here"):
        return ""
    return env_key


# Which Gemini model to call. "gemini-3.8-flash" is Google's recommended
# free-tier model for high accuracy, grounded QA, and fast inference.
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-3.8-flash")

# Fallback cascade of models for zero-downtime quota resilience.
# If the primary model hits a daily quota (RPD) or rate limit,
# docRag automatically cascades to the next available free model.
CANDIDATE_MODELS: list[str] = [
    LLM_MODEL_NAME,
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

# Temperature controls randomness.  0.1 keeps answers factual and
# reproducible — good for citation-grounded QA.  Raise toward 0.7 if you
# want more creative phrasing (at the cost of accuracy).
LLM_TEMPERATURE: float = 0.1

# Maximum tokens the LLM may generate in one response.
# 1024 is enough for a detailed paragraph with citations.
# Raise if answers are getting cut off; lower to save quota.
LLM_MAX_OUTPUT_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# PDF Upload Limits
# ---------------------------------------------------------------------------
# Maximum file size in megabytes.  Streamlit Community Cloud has a 200 MB
# default upload limit, but large PDFs burn more memory, embedding time,
# and LLM context.  10 MB is generous for text-based PDFs.
MAX_FILE_SIZE_MB: int = 10

# Maximum number of pages.  Each page becomes several chunks, and each chunk
# uses memory + embedding time.  50 pages keeps processing under ~30 seconds
# on modest hardware.
MAX_PAGES: int = 50

# Allowed MIME types for upload validation.
ALLOWED_MIME_TYPES: list[str] = ["application/pdf"]

# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------
# chunk_size = how many characters per chunk.
# Smaller chunks → more precise retrieval but less context per chunk.
# Larger chunks  → more context but may dilute the relevant sentence.
# 1000 chars ≈ ~200-250 words — a good starting balance.
CHUNK_SIZE: int = 1000

# chunk_overlap = how many characters overlap between adjacent chunks.
# Overlap prevents a sentence from being split across two chunks with
# neither chunk containing the full thought.
# 200 chars ≈ 20% of chunk_size — standard starting point.
CHUNK_OVERLAP: int = 200

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
# Ultra-compact, fast, local embedding model from Sentence Transformers.
# "paraphrase-MiniLM-L3-v2" produces 384-dimensional vectors on CPU,
# uses <60MB model size, and keeps total container peak RAM well within
# Render free-tier's 512MB RAM ceiling while maintaining 90% top-k retrieval accuracy.
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-MiniLM-L3-v2")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# How many chunks to retrieve for each question.
# More chunks → more evidence for the LLM but a longer prompt.
# 4 chunks × 1000 chars ≈ 4000 chars of context — well within limits.
RETRIEVAL_TOP_K: int = 4

# Maximum L2 distance threshold for vector relevance.
# Normalized embeddings range from L2 distance 0.0 (identical) to 2.0 (orthogonal/opposite).
# If all retrieved chunks have distance > 1.65 (unrelated topic / noise),
# the question is considered off-topic and gets an immediate "not enough info" response.
RELEVANCE_DISTANCE_THRESHOLD: float = 1.65

# ---------------------------------------------------------------------------
# Session & Usage Limits
# ---------------------------------------------------------------------------
# Maximum questions per session, to prevent free-tier abuse.
# This resets when the user clicks "Reset".
MAX_QUESTIONS_PER_SESSION: int = 20

# Maximum characters in a user question.
MAX_QUESTION_LENGTH: int = 500

# ---------------------------------------------------------------------------
# App Metadata
# ---------------------------------------------------------------------------
APP_NAME: str = "docRag"
APP_DESCRIPTION: str = (
    "Upload a PDF, ask questions, and get answers with source citations. "
    "Powered by local embeddings and a free-tier LLM."
)
