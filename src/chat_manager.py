"""
docRag Multi-Chat & Document Session Manager
============================================
Manages independent chat sessions, multi-document indexing with FAISS,
and message histories. Fully in-memory and isolated per session.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

from langchain_core.documents import Document

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, RETRIEVAL_TOP_K
from src.pdf_processing import (
    PDFMetadata,
    PDFPage,
    extract_pages_from_pdf,
    pages_to_langchain_documents,
)
from src.rag_pipeline import Citation, RAGResponse, answer_question
from src.vector_store import (
    create_vector_store,
    split_documents_into_chunks,
)


@dataclass
class ChatMessage:
    id: str
    role: str  # 'user' | 'assistant'
    content: str
    timestamp: str
    citations: list[dict] = field(default_factory=list)
    is_fallback: bool = False


@dataclass
class AttachedDocument:
    file_name: str
    total_pages: int
    file_size_bytes: int
    total_characters: int
    pages: list[PDFPage]
    chunks_count: int


class ChatSession:
    def __init__(self, title: str = "New Chat", session_id: Optional[str] = None):
        self.session_id: str = session_id or str(uuid.uuid4())
        self.title: str = title
        self.created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.updated_at: str = self.created_at
        self.messages: list[ChatMessage] = []
        self.documents: dict[str, AttachedDocument] = {}  # file_name -> AttachedDocument
        self.all_chunks: list[Document] = []
        self.vector_store = None

    def add_pdf(self, file_bytes: bytes, file_name: str) -> AttachedDocument:
        """Extract, chunk, and index a PDF into this chat session's vector store."""
        pages, metadata = extract_pages_from_pdf(
            file_bytes,
            file_name=file_name,
        )

        langchain_docs = pages_to_langchain_documents(pages)
        new_chunks = split_documents_into_chunks(
            langchain_docs,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # Store document record
        attached = AttachedDocument(
            file_name=file_name,
            total_pages=metadata.total_pages,
            file_size_bytes=metadata.file_size_bytes,
            total_characters=metadata.total_characters,
            pages=pages,
            chunks_count=len(new_chunks),
        )
        self.documents[file_name] = attached

        # Rebuild or update the multi-document in-memory FAISS index
        # We accumulate chunks across all attached documents in this chat
        self.all_chunks.extend(new_chunks)
        self.vector_store = create_vector_store(self.all_chunks)

        # Auto-update title if it's the first document in a default 'New Chat'
        if self.title == "New Chat" and len(self.messages) == 0:
            clean_name = file_name.replace(".pdf", "").replace("_", " ").title()
            self.title = clean_name[:28]

        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        return attached

    def add_message(
        self,
        role: str,
        content: str,
        citations: Optional[list[Citation]] = None,
        is_fallback: bool = False,
    ) -> ChatMessage:
        """Append a message to this chat history."""
        citation_dicts = []
        if citations:
            for c in citations:
                citation_dicts.append(
                    {
                        "source_index": c.source_index,
                        "document_name": c.document_name,
                        "page_number": c.page_number,
                        "chunk_id": c.chunk_id,
                        "excerpt": c.excerpt,
                    }
                )

        msg = ChatMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now().strftime("%I:%M %p"),
            citations=citation_dicts,
            is_fallback=is_fallback,
        )
        self.messages.append(msg)
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        return msg

    def to_dict(self, include_pages: bool = False) -> dict:
        """Serialize chat session for REST API responses."""
        docs_summary = []
        for d in self.documents.values():
            doc_data = {
                "file_name": d.file_name,
                "total_pages": d.total_pages,
                "file_size_bytes": d.file_size_bytes,
                "total_characters": d.total_characters,
                "chunks_count": d.chunks_count,
            }
            if include_pages:
                doc_data["pages"] = [
                    {"page_number": p.page_number, "text": p.text}
                    for p in d.pages
                ]
            docs_summary.append(doc_data)

        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "documents_count": len(self.documents),
            "documents": docs_summary,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "citations": m.citations,
                    "is_fallback": m.is_fallback,
                }
                for m in self.messages
            ],
        }


class SessionManager:
    """Singleton-style manager holding active user chat sessions."""

    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}
        # Create an initial default chat
        default_session = self.create_session("Getting Started")
        self.active_session_id: str = default_session.session_id

    def create_session(self, title: str = "New Chat") -> ChatSession:
        new_session = ChatSession(title=title)
        self.sessions[new_session.session_id] = new_session
        self.active_session_id = new_session.session_id
        return new_session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.active_session_id == session_id:
                if self.sessions:
                    self.active_session_id = next(iter(self.sessions))
                else:
                    new_session = self.create_session("New Chat")
                    self.active_session_id = new_session.session_id
            return True
        return False

    def list_sessions(self) -> list[dict]:
        return [
            {
                "session_id": s.session_id,
                "title": s.title,
                "updated_at": s.updated_at,
                "documents_count": len(s.documents),
                "messages_count": len(s.messages),
                "is_active": s.session_id == self.active_session_id,
            }
            for s in sorted(
                self.sessions.values(), key=lambda x: x.updated_at, reverse=True
            )
        ]


# Global session manager instance
session_manager = SessionManager()
