"""
docRag Vector Store & Retrieval Module
======================================
Handles text chunking, local Sentence-Transformers embeddings,
in-memory FAISS vector indexing, and grounded similarity retrieval.
"""

import gc
import os
from typing import Optional
import torch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Force single-thread execution to prevent CPU contention and memory spikes in Docker
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    RELEVANCE_DISTANCE_THRESHOLD,
    RETRIEVAL_TOP_K,
)

# Global singleton cache for embedding model weights (shared across reruns & sessions)
_EMBEDDINGS_INSTANCE: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Retrieve or initialize the HuggingFace embedding model instance.
    Uses 'paraphrase-MiniLM-L3-v2' running on single-thread CPU.
    Weights are loaded once with batch_size=2 and garbage collection
    to strictly enforce <512MB RAM limits on free cloud containers.
    """
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        gc.collect()
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
        )
        gc.collect()
    return _EMBEDDINGS_INSTANCE


def split_documents_into_chunks(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split page-level documents into smaller, overlapping chunks while preserving
    page number and source metadata.

    Parameters:
        documents: List of Document objects with metadata={"source": ..., "page": ...}
        chunk_size: Target character count per chunk
        chunk_overlap: Overlapping characters between adjacent chunks

    Returns:
        List of chunked Document objects with enriched metadata (chunk_id, page, source).
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    raw_chunks = splitter.split_documents(documents)

    # Assign unambiguous chunk IDs for citation mapping
    enriched_chunks: list[Document] = []
    page_chunk_counts: dict[int, int] = {}

    for chunk in raw_chunks:
        page_num = chunk.metadata.get("page", 1)
        source = chunk.metadata.get("source", "document.pdf")

        # Track chunk index per page for readable IDs like 'Page 2, Chunk 1'
        page_chunk_counts[page_num] = page_chunk_counts.get(page_num, 0) + 1
        chunk_index = page_chunk_counts[page_num]
        chunk_id = f"p{page_num}-c{chunk_index}"

        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["source"] = source
        chunk.metadata["page"] = page_num
        enriched_chunks.append(chunk)

    return enriched_chunks


def create_vector_store(
    chunks: list[Document],
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> FAISS:
    """
    Build an in-memory FAISS vector store from chunked documents.

    Security & Privacy:
    - Built strictly in-memory per user session.
    - No pickled index files loaded from untrusted disk storage.
    """
    if not chunks:
        raise ValueError("Cannot create a vector store from an empty list of chunks.")

    if embeddings is None:
        embeddings = get_embedding_model()

    gc.collect()
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    gc.collect()
    return vector_store


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    score_threshold: Optional[float] = RELEVANCE_DISTANCE_THRESHOLD,
) -> list[Document]:
    """
    Perform similarity search on the FAISS vector store for a given question.
    Filters out chunks whose L2 distance exceeds score_threshold (i.e. low relevance).

    Returns:
        Top-k most relevant Document chunks, each carrying metadata with
        'source', 'page', and 'chunk_id'.
    """
    if not query or not query.strip():
        return []

    if score_threshold is None:
        return vector_store.similarity_search(query.strip(), k=top_k)

    # Perform search with L2 distance score
    results_with_scores = vector_store.similarity_search_with_score(query.strip(), k=top_k)

    filtered_docs: list[Document] = []
    for doc, distance in results_with_scores:
        if distance <= score_threshold:
            filtered_docs.append(doc)

    return filtered_docs
