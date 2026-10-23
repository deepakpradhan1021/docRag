"""
docRag Web Server (FastAPI)
===========================
High-performance, pure Python REST API and static frontend server.
Supports multi-chat workspaces, multi-PDF uploads, citation mapping,
and document page inspection.
"""

from datetime import datetime
import os
import time
from collections import defaultdict
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.chat_manager import session_manager
from src.config import (
    APP_DESCRIPTION,
    APP_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LLM_MODEL_NAME,
    MAX_FILE_SIZE_MB,
    MAX_PAGES,
    MAX_QUESTIONS_PER_SESSION,
    RETRIEVAL_TOP_K,
    get_gemini_api_key,
)
from src.pdf_processing import (
    PDFEmptyError,
    PDFPageLimitError,
    PDFSizeLimitError,
    PDFValidationError,
)
from src.rag_pipeline import answer_question

# Create FastAPI app
app = FastAPI(
    title=f"{APP_NAME} API",
    description=APP_DESCRIPTION,
    version="1.0.0",
)

# Enable CORS for local dev or separate Netlify deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-Memory Sliding-Window Rate Limiter (Protects backend from spam & DDoS)
# ---------------------------------------------------------------------------
RATE_LIMIT_WINDOW_SECONDS: int = 60
RATE_LIMIT_MAX_REQUESTS: int = 40  # Max 40 API calls per minute per client IP
_CLIENT_REQUEST_LOG: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    # Only rate-limit API routes (/api/*), static files remain fast & unmetered
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SECONDS

        # Purge timestamps outside sliding window
        recent_timestamps = [t for t in _CLIENT_REQUEST_LOG[client_ip] if t > window_start]
        _CLIENT_REQUEST_LOG[client_ip] = recent_timestamps

        if len(recent_timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded: Maximum {RATE_LIMIT_MAX_REQUESTS} requests per minute. Please try again shortly."
                },
            )

        _CLIENT_REQUEST_LOG[client_ip].append(now)

    response = await call_next(request)
    return response


# --- Request Models ---
class CreateChatRequest(BaseModel):
    title: Optional[str] = "New Chat"


class SendMessageRequest(BaseModel):
    question: str
    api_key: Optional[str] = None


class RenameChatRequest(BaseModel):
    title: str


# --- API Routes ---

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "app": APP_NAME,
        "model": LLM_MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/config")
def get_configuration():
    """Return runtime configuration and key status."""
    default_key = get_gemini_api_key()
    return {
        "app_name": APP_NAME,
        "app_description": APP_DESCRIPTION,
        "model_name": LLM_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "max_pages": MAX_PAGES,
        "has_api_key": bool(default_key),
    }


@app.get("/api/chats")
def list_chats():
    """Return all active chat sessions."""
    return session_manager.list_sessions()


@app.post("/api/chats")
def create_chat(payload: CreateChatRequest):
    """Create a new chat session."""
    session = session_manager.create_session(title=payload.title or "New Chat")
    return session.to_dict()


@app.get("/api/chats/{session_id}")
def get_chat(session_id: str):
    """Retrieve details and messages of a specific chat session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    session_manager.active_session_id = session_id
    return session.to_dict(include_pages=True)


@app.patch("/api/chats/{session_id}")
def rename_chat(session_id: str, payload: RenameChatRequest):
    """Rename a chat session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    session.title = payload.title.strip()[:35] or "Chat"
    return {"status": "success", "title": session.title}


@app.delete("/api/chats/{session_id}")
def delete_chat(session_id: str):
    """Delete a chat session."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"status": "success", "active_session_id": session_manager.active_session_id}


@app.post("/api/chats/{session_id}/upload")
async def upload_documents(session_id: str, files: list[UploadFile] = File(...)):
    """Upload one or multiple PDF documents into a chat session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    results = []
    errors = []

    for file in files:
        try:
            content = await file.read()
            attached = session.add_pdf(content, file.filename)
            results.append({
                "file_name": attached.file_name,
                "total_pages": attached.total_pages,
                "file_size_bytes": attached.file_size_bytes,
                "chunks_count": attached.chunks_count,
            })
        except PDFSizeLimitError as e:
            errors.append(f"{file.filename}: {str(e)}")
        except PDFPageLimitError as e:
            errors.append(f"{file.filename}: {str(e)}")
        except PDFEmptyError as e:
            errors.append(f"{file.filename}: {str(e)}")
        except PDFValidationError as e:
            errors.append(f"{file.filename}: {str(e)}")
        except Exception as e:
            errors.append(f"{file.filename}: Unexpected error: {str(e)}")

    if not results and errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    return {
        "uploaded": results,
        "errors": errors,
        "chat": session.to_dict(),
    }


@app.post("/api/chats/{session_id}/message")
def post_message(session_id: str, payload: SendMessageRequest):
    """Send a user question, retrieve grounded passages, and return citations."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    if not session.vector_store:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one PDF document before asking questions.",
        )

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Record user message in history
    session.add_message(role="user", content=question)

    # Format history for contextual query reformulation
    chat_history = [
        {"role": m.role, "content": m.content}
        for m in session.messages[:-1]
    ]

    # Run grounded RAG pipeline
    rag_resp = answer_question(
        vector_store=session.vector_store,
        question=question,
        api_key=payload.api_key,
        chat_history=chat_history,
        top_k=RETRIEVAL_TOP_K,
    )

    if rag_resp.error:
        # Record error response
        assistant_msg = session.add_message(
            role="assistant",
            content=f"⚠️ {rag_resp.error}",
            is_fallback=True,
        )
        return {
            "error": rag_resp.error,
            "message": {
                "id": assistant_msg.id,
                "role": assistant_msg.role,
                "content": assistant_msg.content,
                "timestamp": assistant_msg.timestamp,
                "citations": [],
                "is_fallback": True,
            },
        }

    # Record assistant message with verified source citations
    assistant_msg = session.add_message(
        role="assistant",
        content=rag_resp.answer,
        citations=rag_resp.citations,
        is_fallback=rag_resp.is_fallback_answer,
    )

    return {
        "message": {
            "id": assistant_msg.id,
            "role": assistant_msg.role,
            "content": assistant_msg.content,
            "timestamp": assistant_msg.timestamp,
            "citations": assistant_msg.citations,
            "is_fallback": assistant_msg.is_fallback,
        },
        "retrieved_chunks_count": rag_resp.retrieved_chunks_count,
    }


@app.get("/api/chats/{session_id}/export")
def export_chat(session_id: str, format: str = "markdown"):
    """Export conversation transcript as Markdown or JSON."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    doc_names = ", ".join(session.documents.keys()) if session.documents else "None"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# {APP_NAME} — {session.title}",
        f"**Date:** {timestamp}",
        f"**Documents:** {doc_names}",
        "\n---\n",
    ]

    if not session.messages:
        lines.append("*No messages recorded in this chat session yet.*")
    else:
        for m in session.messages:
            role_label = "User" if m.role == "user" else "docRag Assistant"
            lines.append(f"### {role_label} ({m.timestamp})")
            lines.append(f"{m.content}\n")
            if m.citations:
                lines.append("**Source Citations:**")
                for c in m.citations:
                    lines.append(
                        f"- **[Source {c['source_index']}]** Page {c['page_number']} (`{c['document_name']}`): *\"{c['excerpt']}\"*"
                    )
                lines.append("")
            lines.append("---\n")

    md_content = "\n".join(lines)
    return PlainTextResponse(
        content=md_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=docRag-{session.title[:15]}.md"
        },
    )


# --- Static Files / Frontend Serving ---
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

# Mount /static for explicit /static path references
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount root static files (index.html, style.css, app.js, etc.)
# Placed AFTER all /api routes so API endpoints take absolute priority
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static_root")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print(">> Starting docRag Custom Web Server on http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
