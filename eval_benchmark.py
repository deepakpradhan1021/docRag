"""
docRag Retrieval Evaluation Benchmark
=====================================
Evaluates the retrieval accuracy and page citation fidelity of the RAG pipeline.
Measures Recall@K and Page Hit Rate against a synthetic multi-page ground-truth document.
"""

import sys
import unittest
import pymupdf
from langchain_core.documents import Document

from src.pdf_processing import extract_pages_from_pdf, pages_to_langchain_documents
from src.vector_store import (
    create_vector_store,
    retrieve_relevant_chunks,
    split_documents_into_chunks,
)

# 10 Ground-Truth Questions mapped to exact pages in the synthetic document
BENCHMARK_DATA = [
    {
        "page": 1,
        "question": "When was docRag founded and what was its original seed funding?",
        "expected_keywords": ["founded in 2024", "seed funding of $2.5 million"],
    },
    {
        "page": 2,
        "question": "What is the primary embedding architecture used for vector search?",
        "expected_keywords": ["all-MiniLM-L6-v2", "384-dimensional"],
    },
    {
        "page": 3,
        "question": "How does the caching system handle LRU eviction policies?",
        "expected_keywords": ["LRU eviction threshold", "maximum capacity of 500 MB"],
    },
    {
        "page": 4,
        "question": "What security protocols are enforced for data in transit and rest?",
        "expected_keywords": ["AES-256 encryption", "TLS 1.3 protocol"],
    },
    {
        "page": 5,
        "question": "What is the maximum token limit supported by the summarization pipeline?",
        "expected_keywords": ["summarization pipeline supports up to 8192 tokens", "context compression"],
    },
    {
        "page": 6,
        "question": "Which cloud storage adapters are supported out of the box?",
        "expected_keywords": ["AWS S3, Google Cloud Storage, and Azure Blob Storage"],
    },
    {
        "page": 7,
        "question": "What are the latency benchmarks for p95 and p99 query responses?",
        "expected_keywords": ["p95 latency is 45ms", "p99 latency remains under 85ms"],
    },
    {
        "page": 8,
        "question": "What is the database indexing strategy for session metadata?",
        "expected_keywords": ["B-tree indexing", "partitioned monthly by session UUID"],
    },
    {
        "page": 9,
        "question": "What disaster recovery RTO and RPO metrics are guaranteed in the SLA?",
        "expected_keywords": ["RTO of 15 minutes", "RPO of 0 seconds"],
    },
    {
        "page": 10,
        "question": "What open source license governs the docRag core engine?",
        "expected_keywords": ["Apache 2.0 open source license", "permissive commercial use"],
    },
]


def generate_benchmark_pdf_bytes() -> bytes:
    """Generate a 10-page PDF with distinct domain facts per page."""
    doc = pymupdf.open()
    for item in BENCHMARK_DATA:
        page = doc.new_page()
        text_content = (
            f"Page {item['page']} Reference Document\n"
            f"Subject: Technical Specifications and Operations\n\n"
            f"Section {item['page']}.1 Overview:\n"
            f"{' '.join(item['expected_keywords'])}.\n"
            f"This page provides exhaustive operational details regarding chapter {item['page']}.\n"
            f"Additional padding context to reflect real-world document formatting and paragraphs.\n"
        )
        page.insert_text((50, 72), text_content, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def run_benchmark():
    print("=" * 65)
    print("      docRag Retrieval Precision & Page Hit Benchmark")
    print("=" * 65)

    pdf_bytes = generate_benchmark_pdf_bytes()
    pages, meta = extract_pages_from_pdf(pdf_bytes, file_name="benchmark_doc.pdf")
    print(f"Extracted {len(pages)} pages from synthetic benchmark PDF ({meta.file_size_bytes} bytes).")

    langchain_docs = pages_to_langchain_documents(pages)
    chunks = split_documents_into_chunks(langchain_docs, chunk_size=1000, chunk_overlap=200)
    print(f"Created {len(chunks)} text chunks with app settings (1000 chunk_size, 200 overlap). Initializing FAISS vector store...")

    vector_store = create_vector_store(chunks)

    correct_page_retrievals = 0
    top1_page_retrievals = 0
    total_questions = len(BENCHMARK_DATA)

    print("\nRunning 10 Ground-Truth Test Queries:")
    print("-" * 65)

    for i, item in enumerate(BENCHMARK_DATA, 1):
        q = item["question"]
        expected_page = item["page"]
        retrieved = retrieve_relevant_chunks(vector_store, query=q, top_k=3)

        retrieved_pages = [c.metadata.get("page") for c in retrieved]
        hit = expected_page in retrieved_pages
        top1_hit = len(retrieved_pages) > 0 and retrieved_pages[0] == expected_page

        status = "PASS (Top-1)" if top1_hit else ("PASS (Recall@3)" if hit else "FAIL")
        if hit:
            correct_page_retrievals += 1
        if top1_hit:
            top1_page_retrievals += 1

        top_page = retrieved_pages[0] if retrieved_pages else "None"
        print(f"Q{i:02d}: Expected Page {expected_page:02d} | Top-1: Page {top_page} | Hits: {retrieved_pages} -> [{status}]")

    recall3 = (correct_page_retrievals / total_questions) * 100
    top1_acc = (top1_page_retrievals / total_questions) * 100
    print("-" * 65)
    print(f"Top-1 Precision: {top1_page_retrievals}/{total_questions} ({top1_acc:.1f}%)")
    print(f"Recall@3 (Passage Hit Rate): {correct_page_retrievals}/{total_questions} ({recall3:.1f}%)")
    print("=" * 65)
    return correct_page_retrievals, top1_page_retrievals, total_questions


class TestRetrievalBenchmark(unittest.TestCase):
    def test_benchmark_hit_rate(self):
        recall_hits, top1_hits, total = run_benchmark()
        self.assertGreaterEqual(recall_hits, 9, "Retrieval Recall@3 hit rate must be >= 90%")


if __name__ == "__main__":
    unittest.main()
