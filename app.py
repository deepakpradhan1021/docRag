"""
docRag — Document Question Answering with Citations
===================================================
Main Streamlit application interface.
Enhanced with polished aesthetics, page inspection, live pipeline status,
exportable chat transcripts, and quick analysis prompts.
"""

from datetime import datetime
import streamlit as st

from src.config import (
    APP_DESCRIPTION,
    APP_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LLM_MODEL_NAME,
    MAX_FILE_SIZE_MB,
    MAX_PAGES,
    MAX_QUESTION_LENGTH,
    MAX_QUESTIONS_PER_SESSION,
    RETRIEVAL_TOP_K,
    get_gemini_api_key,
)
from src.pdf_processing import (
    PDFEmptyError,
    PDFPageLimitError,
    PDFProcessingError,
    PDFSizeLimitError,
    PDFValidationError,
    extract_pages_from_pdf,
    pages_to_langchain_documents,
)
from src.rag_pipeline import answer_question
from src.vector_store import (
    create_vector_store,
    split_documents_into_chunks,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} — Grounded PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def initialize_session_state() -> None:
    """Initialize session state variables for isolated user state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_metadata" not in st.session_state:
        st.session_state.pdf_metadata = None
    if "pdf_pages" not in st.session_state:
        st.session_state.pdf_pages = []
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "processed_file_id" not in st.session_state:
        st.session_state.processed_file_id = None
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "api_key_override" not in st.session_state:
        st.session_state.api_key_override = ""


def reset_session() -> None:
    """Clear all session data and state completely."""
    st.session_state.messages = []
    st.session_state.vector_store = None
    st.session_state.pdf_metadata = None
    st.session_state.pdf_pages = []
    st.session_state.chunk_count = 0
    st.session_state.processed_file_id = None
    st.session_state.question_count = 0


def clear_chat_only() -> None:
    """Clear chat messages while keeping the loaded PDF index."""
    st.session_state.messages = []
    st.session_state.question_count = 0


def generate_chat_markdown() -> str:
    """Export the conversation transcript as a clean Markdown report."""
    if not st.session_state.messages:
        return "No conversation recorded yet."

    doc_name = (
        st.session_state.pdf_metadata.file_name
        if st.session_state.pdf_metadata
        else "Document"
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# {APP_NAME} — Q&A Transcript",
        f"**Document:** {doc_name}",
        f"**Generated:** {timestamp}",
        "\n---\n",
    ]

    for idx, msg in enumerate(st.session_state.messages, start=1):
        role_title = "User Question" if msg["role"] == "user" else "docRag Answer"
        lines.append(f"### {role_title}")
        lines.append(f"{msg['content']}\n")

        citations = msg.get("citations", [])
        if citations:
            lines.append("**Source Citations:**")
            for c in citations:
                lines.append(
                    f"- **[Source {c.source_index}]** Page {c.page_number} (`{c.document_name}`): {c.excerpt}"
                )
            lines.append("")
        lines.append("---\n")

    return "\n".join(lines)


initialize_session_state()


# -----------------------------------------------------------------------------
# Sidebar: File Upload, Key Management, Settings & Actions
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title(f"📄 {APP_NAME}")
    st.caption("Grounded PDF Question Answering with Page-by-Page Citations")

    st.divider()

    # --- 1. LLM API Key Section ---
    st.subheader("1. LLM API Key")
    default_env_key = get_gemini_api_key()
    has_default_key = bool(default_env_key)

    if has_default_key:
        st.success("API key active (environment / secrets).", icon="🔑")
        api_key_input = st.text_input(
            "Override API key (optional):",
            type="password",
            value="",
            help="Leave empty to use pre-configured key, or enter a custom key.",
        )
    else:
        st.warning("No API key configured.", icon="⚠️")
        api_key_input = st.text_input(
            "Enter Google Gemini API Key:",
            type="password",
            value="",
            help="Get a free key from https://aistudio.google.com/apikey",
        )
        st.caption("💡 Free tier key available at [Google AI Studio](https://aistudio.google.com/apikey).")

    active_api_key = api_key_input.strip() if api_key_input.strip() else default_env_key

    st.divider()

    # --- 2. Document Upload Section ---
    st.subheader("2. Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        help=f"Text-based PDFs only. Maximum {MAX_FILE_SIZE_MB} MB and {MAX_PAGES} pages.",
    )

    # Process uploaded document if new file is selected
    if uploaded_file is not None:
        file_identifier = f"{uploaded_file.name}-{uploaded_file.size}"

        if st.session_state.processed_file_id != file_identifier:
            with st.spinner(f"Indexing '{uploaded_file.name}' with local embeddings..."):
                try:
                    file_bytes = uploaded_file.read()
                    pages, metadata = extract_pages_from_pdf(
                        file_bytes,
                        file_name=uploaded_file.name,
                        max_size_mb=MAX_FILE_SIZE_MB,
                        max_pages=MAX_PAGES,
                    )

                    langchain_docs = pages_to_langchain_documents(pages)
                    chunks = split_documents_into_chunks(
                        langchain_docs,
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )

                    # Build FAISS vector store in session memory
                    vs = create_vector_store(chunks)

                    # Update session state
                    st.session_state.vector_store = vs
                    st.session_state.pdf_metadata = metadata
                    st.session_state.pdf_pages = pages
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.processed_file_id = file_identifier
                    st.session_state.messages = []
                    st.session_state.question_count = 0

                    st.toast(f"Successfully processed {uploaded_file.name}!", icon="✅")

                except PDFSizeLimitError as exc:
                    st.error(f"Size limit exceeded: {exc}", icon="🚫")
                except PDFPageLimitError as exc:
                    st.error(f"Page limit exceeded: {exc}", icon="🚫")
                except PDFEmptyError as exc:
                    st.error(f"Scanned / empty PDF: {exc}", icon="⚠️")
                except PDFValidationError as exc:
                    st.error(f"Invalid PDF: {exc}", icon="❌")
                except Exception as exc:
                    st.error(f"An unexpected error occurred: {exc}", icon="❌")

    # Document summary badge
    if st.session_state.pdf_metadata is not None:
        meta = st.session_state.pdf_metadata
        st.success(f"**Loaded:** {meta.file_name}")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Pages", meta.total_pages)
        with col_m2:
            st.metric("Chunks", st.session_state.chunk_count)

        size_kb = meta.file_size_bytes / 1024
        st.caption(f"📦 Size: {size_kb:.1f} KB • 🔤 Characters: {meta.total_characters:,}")

    st.divider()

    # --- 3. Session Controls & Usage Limits ---
    st.subheader("3. Session Controls")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear conversation but keep document index"):
            clear_chat_only()
            st.rerun()
    with col_btn2:
        if st.button("🔄 Full Reset", use_container_width=True, type="secondary", help="Reset document and chat"):
            reset_session()
            st.rerun()

    remaining_questions = max(0, MAX_QUESTIONS_PER_SESSION - st.session_state.question_count)
    st.caption(f"Session Questions: **{st.session_state.question_count} / {MAX_QUESTIONS_PER_SESSION}**")

    # Export Chat Transcript button
    if st.session_state.messages:
        transcript_md = generate_chat_markdown()
        st.download_button(
            label="📥 Download Q&A Report",
            data=transcript_md,
            file_name=f"docRag-transcript-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Pipeline Settings Expander
    with st.expander("⚙️ Pipeline Architecture"):
        st.markdown(
            f"""
            - **Model:** `{LLM_MODEL_NAME}`
            - **Chunk Size:** `{CHUNK_SIZE}` characters (~200 words)
            - **Chunk Overlap:** `{CHUNK_OVERLAP}` characters (~20%)
            - **Retrieval Top-K:** `{RETRIEVAL_TOP_K}` passages
            - **Embedding Model:** `all-MiniLM-L6-v2` (Local CPU, 384 dims)
            - **Vector Index:** In-Memory FAISS (No disk persistence)
            """
        )

    # Privacy & external transmission disclosure
    with st.expander("🔒 Privacy & Security Model"):
        st.markdown(
            """
            - **Local Embeddings:** Embeddings are generated locally on CPU via Sentence Transformers.
            - **In-Memory Storage:** Vectors exist solely in RAM inside your active session.
            - **External LLM Transmission:** Only top relevant passages matching your question are sent to the Google Gemini API.
            - **Free Tier Policy:** Google may review prompt inputs to improve models on its free tier. Do not upload classified or confidential documents.
            """
        )


# -----------------------------------------------------------------------------
# Main Application Area
# -----------------------------------------------------------------------------

# Top Header Banner
col_head1, col_head2 = st.columns([0.75, 0.25])
with col_head1:
    st.title(f"📄 {APP_NAME}")
    st.markdown(
        f"**{APP_DESCRIPTION}**"
    )
with col_head2:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 10px;">
            <span style="background-color: #f0f2f6; color: #31333F; padding: 4px 8px; border-radius: 6px; font-size: 13px; font-weight: 600;">v1.0 Ready</span>
            <br>
            <span style="font-size: 12px; color: #888;">Zero-Budget Stack</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Badges indicator row
badge_cols = st.columns(4)
with badge_cols[0]:
    st.info("🧠 **Model:** Gemini 2.0 Flash", icon="🤖")
with badge_cols[1]:
    st.info("⚡ **Embeddings:** Local CPU (MiniLM)", icon="💻")
with badge_cols[2]:
    st.info("🗂️ **Storage:** In-Memory FAISS", icon="🛡️")
with badge_cols[3]:
    st.info("📌 **Citations:** Verified Pages", icon="🎯")

st.divider()

# -----------------------------------------------------------------------------
# State 1: No document uploaded yet
# -----------------------------------------------------------------------------
if st.session_state.vector_store is None:
    st.markdown("### 🚀 Getting Started in 3 Simple Steps")

    guide_col1, guide_col2, guide_col3 = st.columns(3)
    with guide_col1:
        st.markdown(
            """
            #### 1. Setup API Key
            Enter your free **Google Gemini API Key** in the left sidebar.
            *(Keys set in `.env` or Streamlit secrets are auto-detected).*
            """
        )
    with guide_col2:
        st.markdown(
            """
            #### 2. Upload your PDF
            Upload a text-based document (up to **10 MB** and **50 pages**).
            The app extracts and indexes it locally in seconds.
            """
        )
    with guide_col3:
        st.markdown(
            """
            #### 3. Ask & Verify
            Ask any question. Answers are **strictly grounded** with clickable source citations and page excerpts.
            """
        )

    st.divider()

    st.markdown("### 💡 Why Grounded RAG with docRag?")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        st.markdown("##### 🎯 Zero Hallucinations")
        st.write("Strict system prompts prevent external guesswork. If the document doesn't mention something, docRag explicitly tells you.")
    with feat_col2:
        st.markdown("##### 📄 Exact Page Tracking")
        st.write("Citations link directly to the human-readable 1-indexed document page number, not arbitrary chunk numbers.")
    with feat_col3:
        st.markdown("##### 💸 Truly Free")
        st.write("Zero subscriptions, zero paid database instances, and zero server fees. Perfect for personal use and deployment.")

# -----------------------------------------------------------------------------
# State 2: Document loaded
# -----------------------------------------------------------------------------
else:
    # Optional Document Page Inspector tab / expander
    if st.session_state.pdf_pages:
        with st.expander("📖 Inspect Extracted Document Pages"):
            page_numbers = [p.page_number for p in st.session_state.pdf_pages]
            selected_page_num = st.selectbox(
                "Select page to inspect extracted text:",
                options=page_numbers,
                format_func=lambda x: f"Page {x}",
            )
            selected_page = next((p for p in st.session_state.pdf_pages if p.page_number == selected_page_num), None)
            if selected_page:
                st.text_area(
                    f"Extracted Text (Page {selected_page.page_number})",
                    value=selected_page.text if selected_page.text else "(Blank page)",
                    height=200,
                    disabled=True,
                )

    # Suggested Prompts (displayed if conversation is empty)
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Quick Action Prompts")
        suggested_cols = st.columns(4)

        sample_prompts = [
            ("📋 Executive Summary", "Provide a comprehensive executive summary of this document."),
            ("🎯 Key Findings", "What are the most critical findings, conclusions, or takeaways in this document?"),
            ("📊 Data & Metrics", "List any key metrics, statistics, numbers, or dates mentioned in the text."),
            ("❓ Methodology & Scope", "What is the scope, background, or methodology described in this document?"),
        ]

        for i, (label, prompt_text) in enumerate(sample_prompts):
            with suggested_cols[i]:
                if st.button(label, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": prompt_text, "citations": []})
                    st.session_state.question_count += 1
                    st.rerun()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # If assistant message contains citations, render them cleanly in an expander
            if msg["role"] == "assistant" and msg.get("citations"):
                citations = msg["citations"]
                with st.expander(f"📌 Source Citations ({len(citations)} verified)"):
                    for c in citations:
                        st.markdown(
                            f"**[Source {c.source_index}]** — 📄 **Page {c.page_number}** • `{c.document_name}`"
                        )
                        st.markdown(
                            f"> *\"{c.excerpt}\"*"
                        )
                        st.divider()

    # Check session question limit
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.warning(
            f"You have reached the limit of {MAX_QUESTIONS_PER_SESSION} questions for this session. "
            "Please click 'Clear Chat' or 'Full Reset' in the sidebar to continue.",
            icon="🛑",
        )
    else:
        # Chat input
        user_query = st.chat_input("Ask a question about your PDF...")

        # Handle user query submission
        if user_query:
            if len(user_query.strip()) > MAX_QUESTION_LENGTH:
                st.error(
                    f"Question exceeds the maximum allowed length of {MAX_QUESTION_LENGTH} characters.",
                    icon="⚠️",
                )
            else:
                st.session_state.messages.append(
                    {"role": "user", "content": user_query.strip(), "citations": []}
                )
                st.session_state.question_count += 1
                st.rerun()

        # If the last message in history is from the user, generate assistant response
        if (
            len(st.session_state.messages) > 0
            and st.session_state.messages[-1]["role"] == "user"
            and (len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "assistant")
        ):
            last_user_msg = st.session_state.messages[-1]["content"]

            with st.chat_message("assistant"):
                # Use st.status for sleek, modern progress indication
                with st.status("Grounded RAG Pipeline in progress...", expanded=True) as status_box:
                    st.write("🔍 Searching local FAISS index for relevant passages...")
                    st.write("🤖 Generating answer grounded with Gemini Flash...")
                    response = answer_question(
                        vector_store=st.session_state.vector_store,
                        question=last_user_msg,
                        api_key=active_api_key,
                        chat_history=st.session_state.messages[:-1],
                        top_k=RETRIEVAL_TOP_K,
                    )
                    st.write("📌 Validating citation IDs and mapping to document pages...")
                    status_box.update(
                        label="Answer generated and citations verified!",
                        state="complete",
                        expanded=False,
                    )

                if response.error:
                    st.error(response.error, icon="⚠️")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"⚠️ **Error:** {response.error}",
                            "citations": [],
                        }
                    )
                else:
                    st.markdown(response.answer)

                    # Display citations
                    if response.citations:
                        with st.expander(f"📌 Source Citations ({len(response.citations)} verified)"):
                            for c in response.citations:
                                st.markdown(
                                    f"**[Source {c.source_index}]** — 📄 **Page {c.page_number}** • `{c.document_name}`"
                                )
                                st.markdown(
                                    f"> *\"{c.excerpt}\"*"
                                )
                                st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "citations": response.citations,
                        }
                    )
                    st.rerun()
