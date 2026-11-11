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

    @patch("src.rag_pipeline.get_gemini_api_key", return_value="")
    def test_missing_api_key_returns_graceful_error(self, mock_get_key):
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

    def test_extract_text_from_content_variants(self):
        from src.rag_pipeline import extract_text_from_content

        # Plain string
        self.assertEqual(extract_text_from_content("  Hello World  "), "Hello World")
        # List of strings
        self.assertEqual(extract_text_from_content(["Section 1", "Section 2"]), "Section 1\n\nSection 2")
        # List of dicts (LangChain block format)
        self.assertEqual(
            extract_text_from_content([{"type": "text", "text": "Part A"}, {"type": "text", "text": "Part B"}]),
            "Part A\n\nPart B"
        )
        # Empty list & None
        self.assertEqual(extract_text_from_content([]), "")
        self.assertEqual(extract_text_from_content(None), "")

    def test_llm_list_content_regression_executive_summary(self):
        """
        Regression test for production bug:
        LLM generation failed: 'list' object has no attribute 'strip'
        Verifies that AIMessage with list content blocks does not raise AttributeError.
        """
        from langchain_core.messages import AIMessage
        from src.vector_store import create_vector_store

        vs = create_vector_store(self.sample_chunks)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "Executive Summary: Sentence Transformers use Siamese networks [Source 1].",
                    "extras": {"signature": "abc"}
                }
            ]
        )

        with patch("src.rag_pipeline.ChatGoogleGenerativeAI", return_value=mock_llm):
            with patch("src.rag_pipeline.get_gemini_api_key", return_value="AIzaSy_fake_test_key"):
                response = answer_question(
                    vector_store=vs,
                    question="Provide a clear, high-level executive summary of the attached document.",
                )

        self.assertIsNone(response.error)
        self.assertIn("Executive Summary:", response.answer)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].source_index, 1)
        self.assertEqual(response.citations[0].page_number, 3)

    def test_llm_list_of_strings_content(self):
        from langchain_core.messages import AIMessage
        from src.vector_store import create_vector_store

        vs = create_vector_store(self.sample_chunks)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content=["FAISS scales to billions of vectors ", "using inverted indexes [Source 1]."]
        )

        with patch("src.rag_pipeline.ChatGoogleGenerativeAI", return_value=mock_llm):
            with patch("src.rag_pipeline.get_gemini_api_key", return_value="AIzaSy_fake_test_key"):
                response = answer_question(
                    vector_store=vs,
                    question="How does FAISS scale?",
                )

        self.assertIsNone(response.error)
        self.assertIn("FAISS scales to billions", response.answer)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].source_index, 1)
        self.assertEqual(response.citations[0].page_number, 7)

    def test_model_fallback_on_transient_error(self):
        """Verifies that transient model errors trigger automatic cascade to candidate models."""
        from langchain_core.messages import AIMessage
        from src.vector_store import create_vector_store

        vs = create_vector_store(self.sample_chunks)

        mock_failing_llm = MagicMock()
        mock_failing_llm.invoke.side_effect = Exception("503 Service Unavailable: High load")

        mock_working_llm = MagicMock()
        mock_working_llm.invoke.return_value = AIMessage(
            content="Grounded answer from fallback model [Source 1]."
        )

        with patch("src.rag_pipeline.ChatGoogleGenerativeAI", side_effect=[mock_failing_llm, mock_working_llm]):
            with patch("src.rag_pipeline.get_gemini_api_key", return_value="AIzaSy_fake_test_key"):
                response = answer_question(
                    vector_store=vs,
                    question="Tell me about Siamese networks.",
                )

        self.assertIsNone(response.error)
        self.assertIn("Grounded answer from fallback model", response.answer)


if __name__ == "__main__":
    unittest.main()
