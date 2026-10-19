/**
 * docRag Client Application
 * Pure Vanilla JavaScript — clean, readable, and easy to explain in any interview.
 */

// --- State ---
let currentSessionId = null;
let currentChatData = null;
let userApiKey = localStorage.getItem("docRag_apiKey") || "";
let customApiBase = localStorage.getItem("docRag_apiBase") || "";
const API_BASE = customApiBase || "";

// --- DOM Elements ---
const chatListContainer = document.getElementById("chatListContainer");
const sidebarDocList = document.getElementById("sidebarDocList");
const newChatBtn = document.getElementById("newChatBtn");
const chatScrollArea = document.getElementById("chatScrollArea");
const heroCenter = document.getElementById("heroCenter");
const messagesList = document.getElementById("messagesList");
const chatTextarea = document.getElementById("chatTextarea");
const sendBtn = document.getElementById("sendBtn");
const activeDocsBar = document.getElementById("activeDocsBar");

// Modals
const uploadModal = document.getElementById("uploadModal");
const attachBtn = document.getElementById("attachBtn");
const closeUploadModalBtn = document.getElementById("closeUploadModalBtn");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadStatusText = document.getElementById("uploadStatusText");

const inspectorModal = document.getElementById("inspectorModal");
const openInspectorBtn = document.getElementById("openInspectorBtn");
const quickInspectBtn = document.getElementById("quickInspectBtn");
const closeInspectorModalBtn = document.getElementById("closeInspectorModalBtn");
const inspectDocSelect = document.getElementById("inspectDocSelect");
const inspectPageSelect = document.getElementById("inspectPageSelect");
const inspectTextarea = document.getElementById("inspectTextarea");

const settingsModal = document.getElementById("settingsModal");
const settingsBtn = document.getElementById("settingsBtn");
const closeSettingsModalBtn = document.getElementById("closeSettingsModalBtn");
const apiKeyInput = document.getElementById("apiKeyInput");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const exportBtn = document.getElementById("exportBtn");

// --- Initialization ---
async function initApp() {
  setupEventListeners();
  if (userApiKey) apiKeyInput.value = userApiKey;
  await loadChatList();
}

// --- Event Listeners ---
function setupEventListeners() {
  // New chat
  newChatBtn.addEventListener("click", createNewChat);

  // Send message on click or Enter (without shift)
  sendBtn.addEventListener("click", handleSendMessage);
  chatTextarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  // Enable/disable send button based on input
  chatTextarea.addEventListener("input", () => {
    chatTextarea.style.height = "auto";
    chatTextarea.style.height = `${Math.min(chatTextarea.scrollHeight, 120)}px`;
    sendBtn.disabled = !chatTextarea.value.trim();
  });

  // Chip buttons click in Hero
  document.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const prompt = btn.getAttribute("data-prompt");
      chatTextarea.value = prompt;
      sendBtn.disabled = false;
      handleSendMessage();
    });
  });

  // Modals open/close
  attachBtn.addEventListener("click", () => openModal(uploadModal));
  closeUploadModalBtn.addEventListener("click", () => closeModal(uploadModal));

  openInspectorBtn.addEventListener("click", openInspector);
  quickInspectBtn.addEventListener("click", openInspector);
  closeInspectorModalBtn.addEventListener("click", () => closeModal(inspectorModal));

  settingsBtn.addEventListener("click", () => openModal(settingsModal));
  closeSettingsModalBtn.addEventListener("click", () => closeModal(settingsModal));

  saveSettingsBtn.addEventListener("click", () => {
    userApiKey = apiKeyInput.value.trim();
    if (userApiKey) {
      localStorage.setItem("docRag_apiKey", userApiKey);
    } else {
      localStorage.removeItem("docRag_apiKey");
    }
    closeModal(settingsModal);
    alert("Settings saved!");
  });

  // File Upload Drag & Drop
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--accent-purple)";
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "rgba(168, 85, 247, 0.4)";
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "rgba(168, 85, 247, 0.4)";
    if (e.dataTransfer.files.length) {
      handleFilesUpload(e.dataTransfer.files);
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      handleFilesUpload(fileInput.files);
    }
  });

  // Inspector dropdown changes
  inspectDocSelect.addEventListener("change", updateInspectorPages);
  inspectPageSelect.addEventListener("change", updateInspectorText);

  // Export
  exportBtn.addEventListener("click", handleExport);
}

function openModal(modal) {
  modal.classList.add("active");
}
function closeModal(modal) {
  modal.classList.remove("active");
}

// --- API Calls ---

async function loadChatList() {
  try {
    const res = await fetch(`${API_BASE}/api/chats");
    const sessions = await res.json();
    renderChatList(sessions);

    if (sessions.length > 0) {
      const active = sessions.find((s) => s.is_active) || sessions[0];
      await switchChat(active.session_id);
    } else {
      await createNewChat();
    }
  } catch (err) {
    console.error("Failed to load chat list:", err);
  }
}

function renderChatList(sessions) {
  chatListContainer.innerHTML = "";
  sessions.forEach((s) => {
    const item = document.createElement("div");
    item.className = `chat-item ${s.session_id === currentSessionId ? "active" : ""}`;
    item.innerHTML = `
      <div class="chat-item-left">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
        <span title="${s.title}">${escapeHtml(s.title)}</span>
      </div>
      <button class="chat-delete-btn" title="Delete chat" onclick="event.stopPropagation(); deleteChat('${s.session_id}')">&times;</button>
    `;
    item.addEventListener("click", () => switchChat(s.session_id));
    chatListContainer.appendChild(item);
  });
}

async function createNewChat() {
  try {
    const res = await fetch(`${API_BASE}/api/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New Chat" }),
    });
    const session = await res.json();
    await loadChatList();
    await switchChat(session.session_id);
  } catch (err) {
    console.error("Failed to create chat:", err);
  }
}

async function switchChat(sessionId) {
  currentSessionId = sessionId;
  try {
    const res = await fetch(`${API_BASE}/api/chats/${sessionId}`);
    currentChatData = await res.json();
    renderCurrentChat();
    await loadChatList(); // Updates active indicator
  } catch (err) {
    console.error("Failed to switch chat:", err);
  }
}

async function deleteChat(sessionId) {
  if (!confirm("Are you sure you want to delete this chat session?")) return;
  try {
    await fetch(`${API_BASE}/api/chats/${sessionId}`, { method: "DELETE" });
    await loadChatList();
  } catch (err) {
    console.error("Failed to delete chat:", err);
  }
}

function renderCurrentChat() {
  if (!currentChatData) return;

  // Render Documents in sidebar
  sidebarDocList.innerHTML = "";
  activeDocsBar.innerHTML = "";

  if (currentChatData.documents.length === 0) {
    sidebarDocList.innerHTML = `<div style="font-size: 12px; color: var(--text-dim); padding: 8px 10px;">No PDFs in this chat.</div>`;
    activeDocsBar.style.display = "none";
  } else {
    activeDocsBar.style.display = "flex";
    currentChatData.documents.forEach((doc) => {
      // Sidebar pill
      const docDiv = document.createElement("div");
      docDiv.className = "doc-pill";
      docDiv.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
        <span title="${doc.file_name}">${escapeHtml(doc.file_name)} (${doc.total_pages}p)</span>
      `;
      sidebarDocList.appendChild(docDiv);

      // Input bar tag
      const tagDiv = document.createElement("div");
      tagDiv.className = "active-doc-tag";
      tagDiv.innerHTML = `📄 ${escapeHtml(doc.file_name)} <span style="opacity: 0.6;">(${doc.total_pages}p)</span>`;
      activeDocsBar.appendChild(tagDiv);
    });
  }

  // Render Messages or Hero
  if (currentChatData.messages.length === 0) {
    heroCenter.style.display = "block";
    messagesList.innerHTML = "";
  } else {
    heroCenter.style.display = "none";
    messagesList.innerHTML = "";
    currentChatData.messages.forEach((msg) => renderMessageRow(msg));
  }

  scrollChatToBottom();
}

function renderMessageRow(msg) {
  const row = document.createElement("div");
  row.className = `message-row ${msg.role}`;

  let citationsHtml = "";
  if (msg.role === "assistant" && msg.citations && msg.citations.length > 0) {
    citationsHtml = `
      <div class="citations-container">
        <div class="citations-header">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>
          Verified Source Citations (${msg.citations.length})
        </div>
        ${msg.citations
          .map(
            (c) => `
          <div class="citation-card">
            <div class="citation-meta">
              <span class="citation-badge">[Source ${c.source_index}]</span>
              <span>📄 Page ${c.page_number}</span>
              <span style="color: var(--text-dim); font-size: 11px;">(${escapeHtml(c.document_name)})</span>
            </div>
            <div class="citation-quote">"${escapeHtml(c.excerpt)}"</div>
          </div>
        `
          )
          .join("")}
      </div>
    `;
  }

  if (msg.role === "user") {
    row.innerHTML = `
      <div class="message-bubble">${escapeHtml(msg.content)}</div>
      <div class="avatar user-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
      </div>
    `;
  } else {
    row.innerHTML = `
      <div class="avatar ai-avatar">
        <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
      </div>
      <div class="message-bubble">
        <div>${formatMarkdownBasic(msg.content)}</div>
        ${citationsHtml}
      </div>
    `;
  }

  messagesList.appendChild(row);
}

// --- Sending Messages ---
async function handleSendMessage() {
  const text = chatTextarea.value.trim();
  if (!text || !currentSessionId) return;

  if (currentChatData.documents.length === 0) {
    alert("Please attach at least one PDF document before asking questions.");
    openModal(uploadModal);
    return;
  }

  // Clear input
  chatTextarea.value = "";
  chatTextarea.style.height = "auto";
  sendBtn.disabled = true;

  // Optimistically show user message
  heroCenter.style.display = "none";
  renderMessageRow({ role: "user", content: text });
  scrollChatToBottom();

  // Show live loading bubble
  const loadingRow = document.createElement("div");
  loadingRow.className = "message-row assistant";
  loadingRow.id = "tempLoadingRow";
  loadingRow.innerHTML = `
    <div class="avatar ai-avatar">✦</div>
    <div class="message-bubble" style="display: flex; align-items: center; gap: 8px;">
      <span style="font-size: 13px; color: var(--accent-purple);">Searching FAISS index and generating grounded answer...</span>
    </div>
  `;
  messagesList.appendChild(loadingRow);
  scrollChatToBottom();

  try {
    const res = await fetch(`${API_BASE}/api/chats/${currentSessionId}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, api_key: userApiKey }),
    });

    const data = await res.json();
    loadingRow.remove();

    if (res.ok && data.message) {
      renderMessageRow(data.message);
      // Refresh chat list to update title
      await loadChatList();
    } else {
      renderMessageRow({
        role: "assistant",
        content: `⚠️ Error: ${data.detail || data.error || "Generation failed"}`,
      });
    }
  } catch (err) {
    loadingRow.remove();
    renderMessageRow({
      role: "assistant",
      content: `⚠️ Network error: Could not reach docRag server.`,
    });
  }

  scrollChatToBottom();
}

// --- Uploading Files ---
async function handleFilesUpload(files) {
  if (!files || !files.length || !currentSessionId) return;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  uploadStatusText.innerText = "Indexing PDF(s) with local embeddings... Please wait...";

  try {
    const res = await fetch(`${API_BASE}/api/chats/${currentSessionId}/upload`, {
      method: "POST",
      body: formData,
    });

    const result = await res.json();
    if (res.ok) {
      uploadStatusText.innerText = `Successfully indexed ${result.uploaded.length} document(s)!`;
      setTimeout(() => {
        closeModal(uploadModal);
        uploadStatusText.innerText = "";
      }, 1000);
      await switchChat(currentSessionId);
    } else {
      uploadStatusText.innerText = `Error: ${result.detail || "Upload failed."}`;
    }
  } catch (err) {
    uploadStatusText.innerText = "Failed to upload: Network error.";
  }
}

// --- Document Page Inspector ---
function openInspector() {
  if (!currentChatData || !currentChatData.documents || currentChatData.documents.length === 0) {
    alert("No PDF documents attached in this chat yet. Please attach a PDF first.");
    return;
  }

  inspectDocSelect.innerHTML = "";
  currentChatData.documents.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d.file_name;
    opt.innerText = `${d.file_name} (${d.total_pages} pages)`;
    inspectDocSelect.appendChild(opt);
  });

  updateInspectorPages();
  openModal(inspectorModal);
}

function updateInspectorPages() {
  const selectedDocName = inspectDocSelect.value;
  const doc = currentChatData.documents.find((d) => d.file_name === selectedDocName);
  if (!doc) return;

  inspectPageSelect.innerHTML = "";
  if (doc.pages && doc.pages.length) {
    doc.pages.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.page_number;
      opt.innerText = `Page ${p.page_number}`;
      inspectPageSelect.appendChild(opt);
    });
  }
  updateInspectorText();
}

function updateInspectorText() {
  const selectedDocName = inspectDocSelect.value;
  const pageNum = parseInt(inspectPageSelect.value);
  const doc = currentChatData.documents.find((d) => d.file_name === selectedDocName);
  if (!doc || !doc.pages) return;

  const page = doc.pages.find((p) => p.page_number === pageNum);
  inspectTextarea.value = page ? page.text : "(No content on this page)";
}

// --- Export ---
function handleExport() {
  if (!currentSessionId) return;
  window.open(`/api/chats/${currentSessionId}/export`, "_blank");
}

// --- Helpers ---
function scrollChatToBottom() {
  setTimeout(() => {
    chatScrollArea.scrollTop = chatScrollArea.scrollHeight;
  }, 50);
}

function escapeHtml(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatMarkdownBasic(text) {
  if (!text) return "";
  let formatted = escapeHtml(text);
  // Bold
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  // Italic
  formatted = formatted.replace(/\*(.*?)\*/g, "<em>$1</em>");
  // Source tags
  formatted = formatted.replace(
    /\[Source\s*(\d+)\]/g,
    '<span class="badge-pill" style="font-size: 11px;">Source $1</span>'
  );
  // Line breaks
  formatted = formatted.replace(/\n/g, "<br>");
  return formatted;
}

// Start
document.addEventListener("DOMContentLoaded", initApp);
