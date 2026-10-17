"""
Unit tests for the RAG pipeline: prompt construction, citation mapping,
hallucination filtering, and missing API key handling.
"""

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag_pipeline import (
    NOT_ENOUGH_INFO_PHRASE,
    answer_question,
    build_system_prompt,
    extract_and_validate_citations,
    format_context_prompt,
)


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            Document(
                page_content="Sentence Transformers use Siamese networks for semantic embedding.",
                metadata={"source": "ml_guide.pdf", "page": 3, "chunk_id": "p3-c1"},
            ),
            Document(
                page_content="FAISS scales to billions of vectors using inverted index structures.",
                metadata={"source": "ml_guide.pdf", "page": 7, "chunk_id": "p7-c1"},
            ),
        ]

    def test_format_context_prompt(self):
        context_str, source_map = format_context_prompt(self.sample_chunks)

        self.assertIn("[Source 1 | Page 3 | Document: ml_guide.pdf]", context_str)
        self.assertIn("[Source 2 | Page 7 | Document: ml_guide.pdf]", context_str)
        self.assertIn("<untrusted_document_content>", context_str)
        self.assertEqual(len(source_map), 2)
        self.assertEqual(source_map[1].metadata["page"], 3)
        self.assertEqual(source_map[2].metadata["page"], 7)

    def test_build_system_prompt_security_rules(self):
        prompt = build_system_prompt()
        self.assertIn("SECURITY RULES", prompt)
        self.assertIn("untrusted", prompt.lower())
        self.assertIn(NOT_ENOUGH_INFO_PHRASE, prompt)

    def test_citation_extraction_and_page_mapping(self):
        _, source_map = format_context_prompt(self.sample_chunks)

        # Model answer citing both sources
        answer = (
            "Sentence Transformers utilize Siamese architectures [Source 1], "
            "whereas FAISS utilizes inverted file indexes [Source 2]."
        )

        citations = extract_and_validate_citations(answer, source_map)

        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0].source_index, 1)
        self.assertEqual(citations[0].page_number, 3)
        self.assertEqual(citations[0].document_name, "ml_guide.pdf")

        self.assertEqual(citations[1].source_index, 2)
        self.assertEqual(citations[1].page_number, 7)
        self.assertEqual(citations[1].document_name, "ml_guide.pdf")

    def test_hallucinated_citation_discarded(self):
        _, source_map = format_context_prompt(self.sample_chunks)

        # Model hallucinates [Source 99] and [Source 0]
        answer = "This fact is backed by [Source 99] and [Source 1]."
        citations = extract_and_validate_citations(answer, source_map)

        # Only Source 1 is valid and should be kept
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].source_index, 1)
        self.assertEqual(citations[0].page_number, 3)

    def test_missing_api_key_returns_graceful_error(self):
        response = answer_question(
            vector_store=MagicMock(),
            question="What is FAISS?",
            api_key="",  # Empty API key
        )

        self.assertIsNotNone(response.error)
        self.assertIn("API key is missing", response.error)
        self.assertEqual(response.answer, "")

    def test_question_not_answered_by_pdf(self):
        _, source_map = format_context_prompt(self.sample_chunks)

        fallback_answer = (
            "The provided document does not contain enough information to answer this question."
        )
        citations = extract_and_validate_citations(fallback_answer, source_map)
        self.assertEqual(len(citations), 0)


if __name__ == "__main__":
    unittest.main()
