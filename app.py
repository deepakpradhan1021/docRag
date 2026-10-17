"""
docRag — Document Question Answering with Citations
===================================================
Main Streamlit application interface.
"""

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
    page_title=f"{APP_NAME} — PDF Q&A with Citations",
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
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "processed_file_id" not in st.session_state:
        st.session_state.processed_file_id = None
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "api_key_override" not in st.session_state:
        st.session_state.api_key_override = ""


def reset_session() -> None:
    """Clear all session data and state."""
    st.session_state.messages = []
    st.session_state.vector_store = None
    st.session_state.pdf_metadata = None
    st.session_state.chunk_count = 0
    st.session_state.processed_file_id = None
    st.session_state.question_count = 0


initialize_session_state()


# -----------------------------------------------------------------------------
# Sidebar: File Upload, Key Management, and Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title(f"📄 {APP_NAME}")
    st.caption("Upload a text-based PDF to ask questions with verified source citations.")

    st.divider()

    # --- API Key Section ---
    st.subheader("1. LLM API Key")
    default_env_key = get_gemini_api_key()
    has_default_key = bool(default_env_key)

    if has_default_key:
        st.success("API key detected from environment / secrets.", icon="🔑")
        api_key_input = st.text_input(
            "Change API key (optional):",
            type="password",
            value="",
            help="Leave empty to use the pre-configured key, or enter a custom key.",
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

    # --- Document Upload Section ---
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
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
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
        st.caption(f"Size: {size_kb:.1f} KB | Characters: {meta.total_characters:,}")

    st.divider()

    # --- Reset & Limits Section ---
    st.subheader("3. Session Controls")
    if st.button("🔄 Reset Document & Chat", use_container_width=True, type="secondary"):
        reset_session()
        st.rerun()

    remaining_questions = max(0, MAX_QUESTIONS_PER_SESSION - st.session_state.question_count)
    st.caption(f"Questions remaining in this session: **{remaining_questions} / {MAX_QUESTIONS_PER_SESSION}**")

    # Configuration inspector expander
    with st.expander("⚙️ Pipeline Settings & Trade-offs"):
        st.markdown(
            f"""
            - **Model:** `{LLM_MODEL_NAME}`
            - **Chunk Size:** `{CHUNK_SIZE}` characters (~200 words). Balances specific sentence retrieval with sufficient surrounding context.
            - **Chunk Overlap:** `{CHUNK_OVERLAP}` characters. Prevents cutting off sentences across boundaries.
            - **Retrieval Top-K:** `{RETRIEVAL_TOP_K}` passages. Balances evidence completeness against prompt size and LLM latency.
            - **Embedding Model:** `all-MiniLM-L6-v2` (Local Sentence Transformers, 384 dimensions, CPU).
            """
        )

    # Privacy & external transmission disclosure
    with st.expander("🔒 Privacy & Security Notice"):
        st.markdown(
            """
            - **Local Embeddings:** Embeddings are generated locally on the server via CPU.
            - **In-Memory Storage:** Vectors are stored in RAM within your active session and never written to disk or shared across users.
            - **External LLM Transmission:** When you ask a question, the top relevant excerpts from your PDF are sent to Google Gemini API for answer generation.
            - **Free Tier Policy:** On Google Gemini's free tier, Google may retain and review prompt submissions to improve models. Do not upload classified, highly sensitive, or strictly confidential data.
            """
        )


# -----------------------------------------------------------------------------
# Main Application Area
# -----------------------------------------------------------------------------
st.title(f"📄 {APP_NAME}")
st.write(APP_DESCRIPTION)

# State 1: No document uploaded yet
if st.session_state.vector_store is None:
    st.info(
        "👋 **Welcome! To get started:**\n"
        "1. (Optional) Provide your Google Gemini API key in the left sidebar.\n"
        "2. Upload a text-based PDF using the sidebar uploader.\n"
        "3. Ask questions in the chat box below to receive answers grounded in your document with page citations."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📌 Page Citations")
        st.write("Every answer cites exact document page numbers and includes expandable source excerpts.")
    with col2:
        st.markdown("### 🛡️ Grounded Answers")
        st.write("Answers are restricted solely to retrieved document passages. When content is absent, docRag explicitly tells you.")
    with col3:
        st.markdown("### ⚡ Zero Cost")
        st.write("Built entirely with free tools: local Sentence Transformers, FAISS, and Gemini free tier.")

else:
    # State 2: Document loaded — show suggestions if chat is empty
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Suggested Questions")
        suggested_cols = st.columns(3)

        sample_prompts = [
            "What is the executive summary of this document?",
            "What are the key points or findings discussed?",
            "What conclusions or recommendations are made?",
        ]

        for i, prompt_text in enumerate(sample_prompts):
            with suggested_cols[i]:
                if st.button(prompt_text, key=f"sugg_{i}", use_container_width=True):
                    # Inject question into chat
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
                with st.expander(f"📌 Source Citations ({len(citations)} found)"):
                    for c in citations:
                        st.markdown(
                            f"**[Source {c.source_index}]** — *Page {c.page_number}* (`{c.document_name}`)"
                        )
                        st.caption(f"> \"{c.excerpt}\"")
                        st.divider()

    # Check session question limit
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.warning(
            f"You have reached the limit of {MAX_QUESTIONS_PER_SESSION} questions for this session. "
            "Please click 'Reset Document & Chat' in the sidebar to start a new session.",
            icon="🛑",
        )
    else:
        # Chat input
        user_query = st.chat_input("Ask a question about your PDF...")

        # Handle user query submission (either from chat_input or if suggested button was clicked)
        pending_query = None
        if user_query:
            if len(user_query.strip()) > MAX_QUESTION_LENGTH:
                st.error(
                    f"Question exceeds the maximum allowed length of {MAX_QUESTION_LENGTH} characters.",
                    icon="⚠️",
                )
            else:
                pending_query = user_query.strip()
                st.session_state.messages.append({"role": "user", "content": pending_query, "citations": []})
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
                with st.spinner("Searching document passages and generating answer..."):
                    response = answer_question(
                        vector_store=st.session_state.vector_store,
                        question=last_user_msg,
                        api_key=active_api_key,
                        chat_history=st.session_state.messages[:-1],
                        top_k=RETRIEVAL_TOP_K,
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
                        with st.expander(f"📌 Source Citations ({len(response.citations)} found)"):
                            for c in response.citations:
                                st.markdown(
                                    f"**[Source {c.source_index}]** — *Page {c.page_number}* (`{c.document_name}`)"
                                )
                                st.caption(f"> \"{c.excerpt}\"")
                                st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "citations": response.citations,
                        }
                    )
                    st.rerun()
