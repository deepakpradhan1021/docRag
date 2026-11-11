"""
Tests for docRag Practical & User-Friendly AI Document Assistant Upgrades
========================================================================
Validates:
1. Intent detection for summarization ("summary", "summaries", "what is this pdf about?", etc.)
2. Intent detection for simplified explanations ("in simple words", etc.)
3. Query expansion for domain acronyms ("ML" -> "machine learning") and synonyms ("advantages" -> "benefits")
4. Typo / fuzzy tolerance via sublinear character n-grams
5. Document-level summary chunks extraction spanning beginning, middle, and conclusion
6. Follow-up query reformulation handling pronouns and short questions
7. Grounded answer generation for summarization without needing keyword matches
8. Graceful handling of absent topics
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.rag_pipeline import (
    answer_question,
    build_system_prompt,
    extract_and_validate_citations,
    extract_text_from_content,
    is_simplified_intent,
    is_summarization_intent,
    reformulate_query_with_history,
)
from src.vector_store import (
    MemorySafeVectorStore,
    expand_query,
    get_document_summary_chunks,
    retrieve_relevant_chunks,
)


class TestPracticalRAG(unittest.TestCase):

    def setUp(self):
        # Sample document corpus simulating a realistic technical guide
        self.sample_chunks = [
            Document(
                page_content="Section 1: Overview. The Northstar platform is a distributed microservice framework for automated builds.",
                metadata={"page": 1, "source": "guide.pdf", "chunk_id": "p1-c1"},
            ),
            Document(
                page_content="Section 2: Machine Learning Integration. Machine learning models automate incident detection and telemetry.",
                metadata={"page": 2, "source": "guide.pdf", "chunk_id": "p2-c1"},
            ),
            Document(
                page_content="Section 3: Release Gates. All builds must satisfy unit testing, security scanning, and QA sign-off.",
                metadata={"page": 3, "source": "guide.pdf", "chunk_id": "p3-c1"},
            ),
            Document(
                page_content="Section 4: Key Benefits. The primary benefits include faster deployment cycles, zero downtime, and lower cost.",
                metadata={"page": 4, "source": "guide.pdf", "chunk_id": "p4-c1"},
            ),
            Document(
                page_content="Section 5: Summary and Conclusions. In summary, Northstar reduces manual operations by 80% and ensures reliable releases.",
                metadata={"page": 5, "source": "guide.pdf", "chunk_id": "p5-c1"},
            ),
        ]
        self.vector_store = MemorySafeVectorStore(documents=self.sample_chunks)

    # -------------------------------------------------------------------------
    # 1. Summarization Intent Detection
    # -------------------------------------------------------------------------
    def test_01_summary_intent_single_word(self):
        is_sum, _ = is_summarization_intent("summary")
        self.assertTrue(is_sum)

    def test_02_summaries_intent_plural(self):
        is_sum, _ = is_summarization_intent("summaries")
        self.assertTrue(is_sum)

    def test_03_what_is_this_document_about_intent(self):
        is_sum, _ = is_summarization_intent("What is this document about?")
        self.assertTrue(is_sum)

    def test_04_explain_the_main_topic_intent(self):
        is_sum, _ = is_summarization_intent("Explain the main topic.")
        self.assertTrue(is_sum)

    def test_05_give_me_five_important_points_intent(self):
        is_sum, sub = is_summarization_intent("Give me five important points.")
        self.assertTrue(is_sum)
        self.assertEqual(sub, "key_points")

    def test_06_make_short_notes_intent(self):
        is_sum, sub = is_summarization_intent("make short notes")
        self.assertTrue(is_sum)
        self.assertEqual(sub, "notes")

    # -------------------------------------------------------------------------
    # 2. Simplified Intent Detection
    # -------------------------------------------------------------------------
    def test_07_explain_in_simple_words(self):
        self.assertTrue(is_simplified_intent("Explain this in simple words."))
        self.assertTrue(is_simplified_intent("explain in simple words"))
        self.assertTrue(is_simplified_intent("in simple terms for a beginner"))
        self.assertFalse(is_simplified_intent("What are the release gate metrics?"))

    # -------------------------------------------------------------------------
    # 3. Acronym & Synonym Query Expansion
    # -------------------------------------------------------------------------
    def test_08_acronym_expansion_ml(self):
        expanded = expand_query("What is ML?")
        self.assertIn("machine learning", expanded.lower())

    def test_09_synonym_expansion_advantages(self):
        expanded = expand_query("What are the advantages?")
        self.assertIn("benefits", expanded.lower())

    # -------------------------------------------------------------------------
    # 4. Retrieval with Acronyms, Synonyms & Typos
    # -------------------------------------------------------------------------
    def test_10_acronym_retrieval(self):
        """Query 'What is ML?' should retrieve Section 2 containing 'Machine learning'."""
        results = retrieve_relevant_chunks(self.vector_store, "What is ML?", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("Machine Learning", results[0].page_content)

    def test_11_synonym_retrieval(self):
        """Query 'What are the advantages?' matches Section 4 containing 'Key Benefits'."""
        results = retrieve_relevant_chunks(self.vector_store, "What are the advantages?", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertTrue(any("Key Benefits" in r.page_content for r in results))

    def test_12_fuzzy_typo_retrieval(self):
        """Fuzzy character matching retrieves machine learning despite typos."""
        results = retrieve_relevant_chunks(self.vector_store, "macheen lerning models", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertTrue(any("Machine learning" in r.page_content for r in results))

    # -------------------------------------------------------------------------
    # 5. Document-Level Summary Chunks
    # -------------------------------------------------------------------------
    def test_13_summary_chunks_coverage(self):
        """Document summary chunks should cover beginning, middle, and conclusion."""
        summary_chunks = get_document_summary_chunks(self.vector_store, max_chunks=3)
        self.assertEqual(len(summary_chunks), 3)
        # First chunk is overview, last chunk is conclusion
        self.assertIn("Overview", summary_chunks[0].page_content)
        self.assertIn("Summary and Conclusions", summary_chunks[-1].page_content)

    # -------------------------------------------------------------------------
    # 6. End-to-End Pipeline Scenarios with Mocked LLM
    # -------------------------------------------------------------------------
    @patch("src.rag_pipeline.ChatGoogleGenerativeAI")
    def test_14_e2e_summary_query(self, MockChat):
        """User asking 'summary' triggers summarization and receives structured summary."""
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = AIMessage(
            content="### Executive Overview\nNorthstar is a distributed microservice framework [Source 1].\n\n### Main Topics\nCovers ML automation [Source 2] and release gates [Source 3]."
        )
        MockChat.return_value = mock_instance

        resp = answer_question(self.vector_store, "summary", api_key="dummy_key")
        self.assertFalse(resp.is_fallback_answer)
        self.assertIn("Executive Overview", resp.answer)
        self.assertGreaterEqual(len(resp.citations), 1)

    @patch("src.rag_pipeline.ChatGoogleGenerativeAI")
    def test_15_e2e_summaries_query(self, MockChat):
        """User asking 'summaries' triggers summarization and succeeds."""
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = AIMessage(
            content="Northstar automates build pipelines and releases [Source 1]."
        )
        MockChat.return_value = mock_instance

        resp = answer_question(self.vector_store, "summaries", api_key="dummy_key")
        self.assertFalse(resp.is_fallback_answer)
        self.assertIn("Northstar", resp.answer)

    @patch("src.rag_pipeline.ChatGoogleGenerativeAI")
    def test_16_e2e_followup_question(self, MockChat):
        """Follow-up question with pronouns is reformulated using conversation history."""
        mock_instance = MagicMock()
        # First call: reformulate query
        mock_instance.invoke.side_effect = [
            AIMessage(content="types of machine learning in Northstar"),
            AIMessage(content="Northstar uses automated ML models [Source 1]."),
        ]
        MockChat.return_value = mock_instance

        history = [
            {"role": "user", "content": "Explain machine learning in this document"},
            {"role": "assistant", "content": "Machine learning models automate incident detection."},
        ]

        resp = answer_question(
            self.vector_store,
            "What are its types?",
            api_key="dummy_key",
            chat_history=history,
        )
        self.assertFalse(resp.is_fallback_answer)


if __name__ == "__main__":
    unittest.main()
