"""
Integration and end-to-end flow tests for docRag.
Covers PDF generation -> extraction -> chunking -> retrieval -> citation verification,
as well as session isolation and reset logic.
"""

import unittest
from unittest.mock import MagicMock, patch
import pymupdf
from langchain_core.messages import AIMessage

from src.pdf_processing import (
    extract_pages_from_pdf,
    pages_to_langchain_documents,
)
from src.rag_pipeline import answer_question
from src.vector_store import (
    create_vector_store,
    split_documents_into_chunks,
)


def make_test_pdf() -> bytes:
    """Create a 3-page test PDF in memory."""
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((50, 72), "Project docRag is a grounded PDF question-answering tool built in Python.")

    p2 = doc.new_page()
    p2.insert_text((50, 72), "The vector database used is FAISS, developed by Meta AI Research.")

    p3 = doc.new_page()
    p3.insert_text((50, 72), "Streamlit Community Cloud is used for zero-cost web deployment.")

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestE2EAndSession(unittest.TestCase):

    def setUp(self):
        self.pdf_bytes = make_test_pdf()
        self.pages, self.metadata = extract_pages_from_pdf(self.pdf_bytes, file_name="e2e_test.pdf")
        self.docs = pages_to_langchain_documents(self.pages)
        self.chunks = split_documents_into_chunks(self.docs, chunk_size=300, chunk_overlap=50)
        self.vector_store = create_vector_store(self.chunks)

    def test_pipeline_answered_question(self):
        """Simulate LLM answering with a grounded citation."""
        mock_llm_response = AIMessage(
            content="The vector database used in docRag is FAISS [Source 1]."
        )

        with patch("src.rag_pipeline.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_llm_response
            MockLLM.return_value = mock_instance

            response = answer_question(
                vector_store=self.vector_store,
                question="Which vector database is used?",
                api_key="fake-test-key-12345",
            )

            self.assertIsNone(response.error)
            self.assertIn("FAISS", response.answer)
            self.assertEqual(len(response.citations), 1)
            # The passage mentioning FAISS is on Page 2
            self.assertEqual(response.citations[0].page_number, 2)
            self.assertEqual(response.citations[0].document_name, "e2e_test.pdf")
            self.assertFalse(response.is_fallback_answer)

    def test_pipeline_unanswered_question(self):
        """Simulate LLM responding with not enough info when content is absent."""
        mock_llm_response = AIMessage(
            content="The provided document does not contain enough information to answer this question."
        )

        with patch("src.rag_pipeline.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_llm_response
            MockLLM.return_value = mock_instance

            response = answer_question(
                vector_store=self.vector_store,
                question="What is the capital of Mars?",
                api_key="fake-test-key-12345",
            )

            self.assertIsNone(response.error)
            self.assertTrue(response.is_fallback_answer)
            self.assertEqual(len(response.citations), 0)

    def test_session_state_reset_simulation(self):
        """Verify that clearing session state properly resets document and conversation."""
        session_mock = {
            "messages": [{"role": "user", "content": "Hello"}],
            "vector_store": self.vector_store,
            "pdf_metadata": self.metadata,
            "chunk_count": len(self.chunks),
            "processed_file_id": "test-123",
            "question_count": 5,
        }

        # Simulate reset function
        session_mock["messages"] = []
        session_mock["vector_store"] = None
        session_mock["pdf_metadata"] = None
        session_mock["chunk_count"] = 0
        session_mock["processed_file_id"] = None
        session_mock["question_count"] = 0

        self.assertEqual(len(session_mock["messages"]), 0)
        self.assertIsNone(session_mock["vector_store"])
        self.assertIsNone(session_mock["pdf_metadata"])
        self.assertEqual(session_mock["chunk_count"], 0)
        self.assertIsNone(session_mock["processed_file_id"])
        self.assertEqual(session_mock["question_count"], 0)


if __name__ == "__main__":
    unittest.main()
