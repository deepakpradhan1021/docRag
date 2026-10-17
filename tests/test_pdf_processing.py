"""
Unit tests for PDF validation and text extraction.
"""

import unittest
import pymupdf

from src.pdf_processing import (
    PDFEmptyError,
    PDFPageLimitError,
    PDFSizeLimitError,
    PDFValidationError,
    clean_page_text,
    extract_pages_from_pdf,
    pages_to_langchain_documents,
)


def create_in_memory_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory PDF with given text per page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPDFProcessing(unittest.TestCase):

    def test_clean_page_text(self):
        raw = "Para-\ngraph with multi   spaces and\n\n\n\nextra lines."
        cleaned = clean_page_text(raw)
        self.assertIn("Paragraph", cleaned)
        self.assertNotIn("Para-", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_extract_valid_multipage_pdf(self):
        pdf_bytes = create_in_memory_pdf([
            "First page content with important details about project docRag.",
            "Second page content with architecture explanations and tests.",
        ])

        pages, metadata = extract_pages_from_pdf(pdf_bytes, file_name="sample.pdf")

        self.assertEqual(len(pages), 2)
        self.assertEqual(metadata.total_pages, 2)
        self.assertEqual(metadata.file_name, "sample.pdf")

        # Verify page numbers are 1-indexed
        self.assertEqual(pages[0].page_number, 1)
        self.assertEqual(pages[1].page_number, 2)
        self.assertIn("First page content", pages[0].text)
        self.assertIn("Second page content", pages[1].text)

    def test_pages_to_langchain_documents(self):
        pdf_bytes = create_in_memory_pdf([
            "Page 1 details for test validation.",
            "Page 2 details for test validation.",
        ])
        pages, _ = extract_pages_from_pdf(pdf_bytes, file_name="sample.pdf")
        docs = pages_to_langchain_documents(pages)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[0].metadata["source"], "sample.pdf")
        self.assertEqual(docs[1].metadata["page"], 2)

    def test_empty_or_scanned_pdf(self):
        # A PDF with 2 blank pages (no text)
        pdf_bytes = create_in_memory_pdf(["", ""])
        with self.assertRaises(PDFEmptyError):
            extract_pages_from_pdf(pdf_bytes, file_name="scanned.pdf")

    def test_invalid_header(self):
        with self.assertRaises(PDFValidationError):
            extract_pages_from_pdf(b"NOT A REAL PDF FILE", file_name="corrupt.pdf")

    def test_empty_bytes(self):
        with self.assertRaises(PDFValidationError):
            extract_pages_from_pdf(b"", file_name="empty.pdf")

    def test_size_limit_exceeded(self):
        # Enforce a tiny size limit of 0.0001 MB
        pdf_bytes = create_in_memory_pdf(["Testing tiny size limit."])
        with self.assertRaises(PDFSizeLimitError):
            extract_pages_from_pdf(pdf_bytes, file_name="big.pdf", max_size_mb=0.00001)

    def test_page_limit_exceeded(self):
        pdf_bytes = create_in_memory_pdf([
            "Page 1 text here.",
            "Page 2 text here.",
            "Page 3 text here.",
        ])
        with self.assertRaises(PDFPageLimitError):
            extract_pages_from_pdf(pdf_bytes, file_name="multipage.pdf", max_pages=2)


if __name__ == "__main__":
    unittest.main()
