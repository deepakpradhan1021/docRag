"""
docRag Multi-Chat & Document Session Manager
============================================
Manages independent chat sessions, multi-document indexing with FAISS,
and message histories. Supports persistent on-disk caching in data/sessions.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
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
    get_embedding_model,
    split_documents_into_chunks,
)

# Persistent storage directory
SESSIONS_DIR = Path("data") / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


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
    def __init__(
        self,
        title: str = "New Chat",
        session_id: Optional[str] = None,
        owner_id: Optional[str] = "default_user",
    ):
        self.session_id: str = session_id or str(uuid.uuid4())
        self.owner_id: str = owner_id or "default_user"
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

        # Rebuild or update the multi-document FAISS index
        # Filter out existing chunks for this file if re-uploaded to prevent duplicate indexing
        self.all_chunks = [
            c for c in self.all_chunks
            if c.metadata.get("source") != file_name
        ]
        self.all_chunks.extend(new_chunks)
        self.vector_store = create_vector_store(self.all_chunks)

        # Auto-update title if it's the first document in a default 'New Chat'
        if self.title == "New Chat" and len(self.messages) == 0:
            clean_name = file_name.replace(".pdf", "").replace("_", " ").title()
            self.title = clean_name[:28]

        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.save_to_disk()
        return attached

    def save_to_disk(self) -> None:
        """Persist session metadata, message history, and FAISS vector index to disk."""
        try:
            session_folder = SESSIONS_DIR / self.session_id
            session_folder.mkdir(parents=True, exist_ok=True)

            # 1. Save metadata and message JSON
            meta_path = session_folder / "session.json"
            meta_data = {
                "session_id": self.session_id,
                "owner_id": self.owner_id,
                "title": self.title,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
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
                "documents": [
                    {
                        "file_name": d.file_name,
                        "total_pages": d.total_pages,
                        "file_size_bytes": d.file_size_bytes,
                        "total_characters": d.total_characters,
                        "chunks_count": d.chunks_count,
                        "pages": [
                            {"page_number": p.page_number, "text": p.text}
                            for p in d.pages
                        ],
                    }
                    for d in self.documents.values()
                ],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)

            # 2. Save FAISS index
            if self.vector_store is not None:
                faiss_dir = session_folder / "faiss_index"
                self.vector_store.save_local(str(faiss_dir))
        except Exception:
            # Graceful degrade if disk write fails
            pass

    @classmethod
    def load_from_disk(cls, session_id: str) -> Optional["ChatSession"]:
        """Restore a ChatSession and its FAISS index from disk."""
        session_folder = SESSIONS_DIR / session_id
        meta_path = session_folder / "session.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            session = cls(
                title=data.get("title", "New Chat"),
                session_id=session_id,
                owner_id=data.get("owner_id", "default_user"),
            )
            session.created_at = data.get("created_at", session.created_at)
            session.updated_at = data.get("updated_at", session.updated_at)

            # Restore messages
            for m in data.get("messages", []):
                session.messages.append(
                    ChatMessage(
                        id=m["id"],
                        role=m["role"],
                        content=m["content"],
                        timestamp=m.get("timestamp", ""),
                        citations=m.get("citations", []),
                        is_fallback=m.get("is_fallback", False),
                    )
                )

            # Restore documents and reconstruct chunk documents
            restored_chunks: list[Document] = []
            for d in data.get("documents", []):
                pages = [
                    PDFPage(page_number=p["page_number"], text=p["text"], source_name=d["file_name"])
                    for p in d.get("pages", [])
                ]
                attached = AttachedDocument(
                    file_name=d["file_name"],
                    total_pages=d["total_pages"],
                    file_size_bytes=d["file_size_bytes"],
                    total_characters=d["total_characters"],
                    pages=pages,
                    chunks_count=d.get("chunks_count", 0),
                )
                session.documents[d["file_name"]] = attached

                # Convert pages to LangChain documents and chunk
                langchain_docs = pages_to_langchain_documents(pages)
                chunks = split_documents_into_chunks(
                    langchain_docs,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                restored_chunks.extend(chunks)

            session.all_chunks = restored_chunks

            # Restore FAISS index if present
            faiss_dir = session_folder / "faiss_index"
            if faiss_dir.exists():
                try:
                    from langchain_community.vectorstores import FAISS
                    session.vector_store = FAISS.load_local(
                        str(faiss_dir),
                        get_embedding_model(),
                        allow_dangerous_deserialization=True,
                    )
                except Exception:
                    if session.all_chunks:
                        session.vector_store = create_vector_store(session.all_chunks)
            elif session.all_chunks:
                session.vector_store = create_vector_store(session.all_chunks)

            return session
        except Exception:
            return None

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
        self.save_to_disk()
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
            "owner_id": self.owner_id,
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
    """Singleton-style manager holding active user chat sessions with persistent disk cache."""

    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}
        self.load_all_from_disk()

        if not self.sessions:
            # Create an initial default chat only if none restored
            default_session = self.create_session("Getting Started")
            self.active_session_id: str = default_session.session_id
        else:
            # Set active session to most recently updated
            sorted_sessions = sorted(
                self.sessions.values(), key=lambda x: x.updated_at, reverse=True
            )
            self.active_session_id = sorted_sessions[0].session_id

    def load_all_from_disk(self) -> None:
        """Scan data/sessions directory and load persisted sessions."""
        try:
            if not SESSIONS_DIR.exists():
                return
            for child in SESSIONS_DIR.iterdir():
                if child.is_dir() and (child / "session.json").exists():
                    sess = ChatSession.load_from_disk(child.name)
                    if sess:
                        self.sessions[sess.session_id] = sess
        except Exception:
            pass

    def create_session(self, title: str = "New Chat", owner_id: str = "default_user") -> ChatSession:
        new_session = ChatSession(title=title, owner_id=owner_id)
        self.sessions[new_session.session_id] = new_session
        self.active_session_id = new_session.session_id
        new_session.save_to_disk()
        return new_session

    def get_session(self, session_id: str, owner_id: Optional[str] = None) -> Optional[ChatSession]:
        session = self.sessions.get(session_id)
        if not session:
            session = ChatSession.load_from_disk(session_id)
            if session:
                self.sessions[session.session_id] = session

        if session and owner_id and session.owner_id != owner_id:
            return None
        return session

    def delete_session(self, session_id: str, owner_id: Optional[str] = None) -> bool:
        session = self.get_session(session_id, owner_id=owner_id)
        if not session:
            return False

        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Remove from disk
        try:
            import shutil
            sess_dir = SESSIONS_DIR / session_id
            if sess_dir.exists():
                shutil.rmtree(sess_dir, ignore_errors=True)
        except Exception:
            pass

        if self.active_session_id == session_id:
            if self.sessions:
                self.active_session_id = next(iter(self.sessions))
            else:
                new_session = self.create_session("New Chat", owner_id=owner_id or "default_user")
                self.active_session_id = new_session.session_id
        return True

    def list_sessions(self, owner_id: Optional[str] = None) -> list[dict]:
        user_sessions = [
            s for s in self.sessions.values()
            if owner_id is None or s.owner_id == owner_id
        ]
        return [
            {
                "session_id": s.session_id,
                "owner_id": s.owner_id,
                "title": s.title,
                "updated_at": s.updated_at,
                "documents_count": len(s.documents),
                "messages_count": len(s.messages),
                "is_active": s.session_id == self.active_session_id,
            }
            for s in sorted(
                user_sessions, key=lambda x: x.updated_at, reverse=True
            )
        ]


# Global session manager instance
session_manager = SessionManager()
