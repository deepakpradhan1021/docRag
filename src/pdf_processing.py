"""
docRag PDF Processing Module
============================
Handles PDF validation, page-by-page text extraction, metadata preservation,
and graceful error handling for scanned or empty PDFs.
"""

from dataclasses import dataclass
import io
import re
from typing import BinaryIO, Union

import pymupdf
from langchain_core.documents import Document

from src.config import MAX_FILE_SIZE_MB, MAX_PAGES


class PDFProcessingError(Exception):
    """Base exception for all PDF processing errors."""
    pass


class PDFSizeLimitError(PDFProcessingError):
    """Raised when the uploaded PDF exceeds the configured size limit."""
    pass


class PDFPageLimitError(PDFProcessingError):
    """Raised when the PDF exceeds the maximum allowed page count."""
    pass


class PDFValidationError(PDFProcessingError):
    """Raised when the file is not a valid PDF or is corrupted."""
    pass


class PDFEmptyError(PDFProcessingError):
    """Raised when the PDF contains no extractable text (e.g. scanned or empty)."""
    pass


@dataclass(frozen=True)
class PDFPage:
    """Represents a single extracted page with associated metadata."""
    page_number: int  # 1-indexed for human readability
    text: str
    source_name: str


@dataclass(frozen=True)
class PDFMetadata:
    """Summary metadata for the processed PDF."""
    file_name: str
    total_pages: int
    file_size_bytes: int
    total_characters: int


def clean_page_text(raw_text: str) -> str:
    """
    Clean and normalize extracted page text:
    - Replace null bytes and non-printable control characters
    - Fix soft hyphens and hyphenated linebreaks (e.g. 'instruc-\ntion' -> 'instruction')
    - Normalize excessive whitespace while preserving paragraph structure
    """
    if not raw_text:
        return ""

    # Remove null bytes and carriage returns
    text = raw_text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")

    # Repair hyphenated word breaks at end of line (e.g., 'para-\ngraph' -> 'paragraph')
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Collapse multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize arbitrary consecutive whitespace within lines (tabs, multiple spaces)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    return text


def validate_pdf_bytes(
    file_bytes: bytes,
    file_name: str = "document.pdf",
    max_size_mb: int = MAX_FILE_SIZE_MB,
    max_pages: int = MAX_PAGES,
) -> pymupdf.Document:
    """
    Validate raw PDF bytes for size, magic bytes, integrity, and page count.
    Returns an open PyMuPDF Document instance if valid.
    """
    file_size = len(file_bytes)
    max_size_bytes = max_size_mb * 1024 * 1024

    if file_size == 0:
        raise PDFValidationError("The uploaded file is empty (0 bytes).")

    if file_size > max_size_bytes:
        size_mb_actual = file_size / (1024 * 1024)
        raise PDFSizeLimitError(
            f"File '{file_name}' ({size_mb_actual:.1f} MB) exceeds the "
            f"maximum allowed limit of {max_size_mb} MB."
        )

    # Magic byte verification (%PDF-)
    if not file_bytes.startswith(b"%PDF-"):
        raise PDFValidationError(
            f"File '{file_name}' is not a valid PDF file. Header does not match PDF signature."
        )

    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFValidationError(f"Could not open '{file_name}': {exc}") from exc

    if len(doc) == 0:
        doc.close()
        raise PDFValidationError(f"File '{file_name}' contains zero pages.")

    if len(doc) > max_pages:
        page_count = len(doc)
        doc.close()
        raise PDFPageLimitError(
            f"Document '{file_name}' has {page_count} pages, which exceeds the "
            f"maximum limit of {max_pages} pages."
        )

    return doc


def extract_pages_from_pdf(
    source: Union[bytes, BinaryIO],
    file_name: str = "document.pdf",
    max_size_mb: int = MAX_FILE_SIZE_MB,
    max_pages: int = MAX_PAGES,
) -> tuple[list[PDFPage], PDFMetadata]:
    """
    Extract text page by page from a PDF file while preserving page numbers.

    Parameters:
        source: Raw PDF bytes or a file-like stream (e.g. Streamlit's UploadedFile).
        file_name: Original filename for citation metadata.
        max_size_mb: Configurable maximum size limit in MB.
        max_pages: Configurable maximum page limit.

    Returns:
        tuple of (list of PDFPage, PDFMetadata)

    Raises:
        PDFSizeLimitError: if file exceeds size limit.
        PDFPageLimitError: if file exceeds page limit.
        PDFValidationError: if file is not a valid or readable PDF.
        PDFEmptyError: if the PDF is scanned or has no extractable text.
    """
    if isinstance(source, bytes):
        raw_bytes = source
    elif hasattr(source, "read"):
        raw_bytes = source.read()
    else:
        raise PDFValidationError("Invalid source provided. Expected bytes or file-like object.")

    doc = validate_pdf_bytes(
        raw_bytes,
        file_name=file_name,
        max_size_mb=max_size_mb,
        max_pages=max_pages,
    )

    extracted_pages: list[PDFPage] = []
    total_chars = 0

    try:
        for idx in range(len(doc)):
            page = doc[idx]
            page_num = idx + 1  # 1-indexed for citations
            raw_text = page.get_text("text") or ""
            cleaned = clean_page_text(raw_text)

            total_chars += len(cleaned)
            # We record each page even if empty, but we validate overall content below
            extracted_pages.append(
                PDFPage(
                    page_number=page_num,
                    text=cleaned,
                    source_name=file_name,
                )
            )
    finally:
        doc.close()

    # Detect scanned or empty PDFs (very low extractable character count)
    # A genuine 1-page document with real text typically has at least ~40-50 characters.
    if total_chars < 30:
        raise PDFEmptyError(
            f"'{file_name}' contains no readable text (extracted only {total_chars} characters). "
            "This document appears to be scanned or image-only. "
            "docRag v1 requires text-based PDFs with selectable text."
        )

    metadata = PDFMetadata(
        file_name=file_name,
        total_pages=len(extracted_pages),
        file_size_bytes=len(raw_bytes),
        total_characters=total_chars,
    )

    return extracted_pages, metadata


def pages_to_langchain_documents(pages: list[PDFPage]) -> list[Document]:
    """
    Convert a list of PDFPage objects into LangChain Document instances,
    preserving page and source metadata for chunking and vector storage.
    """
    docs: list[Document] = []
    for page in pages:
        if not page.text.strip():
            continue  # Skip entirely blank pages from chunking
        docs.append(
            Document(
                page_content=page.text,
                metadata={
                    "source": page.source_name,
                    "page": page.page_number,
                },
            )
        )
    return docs
