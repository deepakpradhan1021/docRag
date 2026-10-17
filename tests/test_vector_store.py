"""
Unit tests for text chunking, FAISS vector store creation, and retrieval.
"""

import unittest
from langchain_core.documents import Document

from src.vector_store import (
    create_vector_store,
    retrieve_relevant_chunks,
    split_documents_into_chunks,
)


class TestVectorStore(unittest.TestCase):

    def test_split_documents_preserves_metadata(self):
        docs = [
            Document(
                page_content=(
                    "docRag is an open-source tool for document question answering. "
                    "It breaks documents into chunks and computes vector embeddings. "
                    "This is page one content."
                ),
                metadata={"source": "manual.pdf", "page": 1},
            ),
            Document(
                page_content=(
                    "FAISS was created by Facebook AI Research. It performs fast similarity search "
                    "on dense vector representations. This is page two content."
                ),
                metadata={"source": "manual.pdf", "page": 2},
            ),
        ]

        chunks = split_documents_into_chunks(docs, chunk_size=80, chunk_overlap=20)
        self.assertGreater(len(chunks), 2)

        # Check metadata preservation
        for chunk in chunks:
            self.assertIn("page", chunk.metadata)
            self.assertIn("source", chunk.metadata)
            self.assertIn("chunk_id", chunk.metadata)
            self.assertIn(chunk.metadata["page"], [1, 2])
            self.assertEqual(chunk.metadata["source"], "manual.pdf")

    def test_empty_chunks_error(self):
        with self.assertRaises(ValueError):
            create_vector_store([])

    def test_vector_store_retrieval(self):
        docs = [
            Document(
                page_content="Apples, oranges, and bananas are fruits rich in vitamins.",
                metadata={"source": "nutrition.pdf", "page": 1, "chunk_id": "p1-c1"},
            ),
            Document(
                page_content="Python 3.13 was released with improvements to performance and typing.",
                metadata={"source": "python_release.pdf", "page": 5, "chunk_id": "p5-c1"},
            ),
        ]

        vs = create_vector_store(docs)

        # Test query matching Python
        results_python = retrieve_relevant_chunks(vs, "What are the new features of Python 3.13?", top_k=1)
        self.assertEqual(len(results_python), 1)
        self.assertEqual(results_python[0].metadata["page"], 5)
        self.assertIn("Python 3.13", results_python[0].page_content)

        # Test query matching fruit
        results_fruit = retrieve_relevant_chunks(vs, "Which fruits contain vitamins?", top_k=1)
        self.assertEqual(len(results_fruit), 1)
        self.assertEqual(results_fruit[0].metadata["page"], 1)
        self.assertIn("Apples", results_fruit[0].page_content)


if __name__ == "__main__":
    unittest.main()
