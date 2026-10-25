/**
 * docRag Client Application
 * Pure Vanilla JavaScript — clean, responsive, robust, and accessible.
 */

// --- Dynamic API Base Resolution ---
const isLocalFile = window.location.protocol === "file:";
const isNetlify = window.location.hostname.includes("netlify.app");
let customApiBase = localStorage.getItem("docRag_apiBase") || "";
const API_BASE = customApiBase || ((isLocalFile || isNetlify) ? "https://docrag-3gb4.onrender.com" : "");

// --- State ---
let currentSessionId = null;
let currentChatData = null;
let userApiKey = sessionStorage.getItem("docRag_apiKey") || ""; // In-memory session persistence
let isServerOnline = false;
let isHealthChecking = false;
let isSending = false;
let isUploading = false;
let activeChatSwitchRequestId = 0;

// Per-browser persistent unique owner ID to isolate chat histories across visitors
let browserOwnerId = localStorage.getItem("docRag_ownerId");
if (!browserOwnerId) {
  browserOwnerId = "usr_" + Math.random().toString(36).substring(2, 11) + Date.now().toString(36);
  localStorage.setItem("docRag_ownerId", browserOwnerId);
}

// --- DOM Elements ---
const chatListContainer = document.getElementById("chatListContainer");
const sidebarDocList = document.getElementById("sidebarDocList");
const newChatBtn = document.getElementById("newChatBtn");
const toggleSidebarBtn = document.getElementById("toggleSidebarBtn");
const sidebar = document.getElementById("sidebar");

const chatScrollArea = document.getElementById("chatScrollArea");
const heroCenter = document.getElementById("heroCenter");
const messagesList = document.getElementById("messagesList");
const chatTextarea = document.getElementById("chatTextarea");
const sendBtn = document.getElementById("sendBtn");
const activeDocsBar = document.getElementById("activeDocsBar");
const backendStatusPill = document.getElementById("backendStatusPill");
const currentModelLabel = document.getElementById("currentModelLabel");
const backendConnectionNote = document.getElementById("backendConnectionNote");

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
const apiBaseInput = document.getElementById("apiBaseInput");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const exportBtn = document.getElementById("exportBtn");

// Track modal trigger element for focus restoration
let lastFocusedElement = null;

// --- Initialization ---
async function initApp() {
  setupEventListeners();
  if (userApiKey && apiKeyInput) apiKeyInput.value = userApiKey;
  if (customApiBase && apiBaseInput) apiBaseInput.value = customApiBase;

  updateConnectionStatusUI("Connecting...");
  await checkServerHealth();
  await loadServerConfig();
  await loadChatList();
}

// --- Live Status UI Helpers ---
function updateConnectionStatusUI(statusText, isReady = false) {
  if (backendStatusPill) {
    backendStatusPill.innerText = statusText;
    backendStatusPill.style.background = isReady ? "rgba(34, 197, 94, 0.2)" : "rgba(234, 88, 12, 0.2)";
    backendStatusPill.style.color = isReady ? "#4ade80" : "#fb923c";
    backendStatusPill.style.borderColor = isReady ? "rgba(74, 222, 128, 0.4)" : "rgba(251, 146, 60, 0.4)";
  }
  if (backendConnectionNote) {
    backendConnectionNote.innerText = `Status: ${statusText}`;
  }
}

// --- Server Connection Health Check ---
async function checkServerHealth() {
  if (isHealthChecking) return;
  isHealthChecking = true;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 12000);
    const res = await fetch(`${API_BASE}/api/health`, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (res.ok) {
      isServerOnline = true;
      removeOfflineBanner();
      updateConnectionStatusUI("Ready", true);
    } else {
      isServerOnline = false;
      showOfflineBanner();
      updateConnectionStatusUI("Waking up...");
    }
  } catch (err) {
    isServerOnline = false;
    showOfflineBanner();
    updateConnectionStatusUI("Waking up...");
  } finally {
    isHealthChecking = false;
  }
}

async function loadServerConfig() {
  if (!isServerOnline) return;
  try {
    const res = await fetch(`${API_BASE}/api/config`);
    if (res.ok) {
      const data = await res.json();
      if (currentModelLabel && data.model_name) {
        currentModelLabel.innerText = `${data.model_name} (Grounded)`;
      }
    }
  } catch (err) {
    console.warn("Could not load backend config:", err);
  }
}

function showOfflineBanner() {
  let banner = document.getElementById("offlineBanner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "offlineBanner";
    banner.style.cssText = `
      background: linear-gradient(90deg, #ea580c 0%, #c2410c 100%);
      color: white;
      padding: 10px 20px;
      font-size: 13px;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
      z-index: 1000;
      position: sticky;
      top: 0;
    `;
    banner.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #fef08a;"></span>
        <span>The document service is waking up (~30-45s cold start)... Please wait a moment.</span>
      </div>
      <button id="retryConnBtn" style="background: white; color: #c2410c; border: none; padding: 4px 12px; border-radius: 6px; font-weight: 700; cursor: pointer;">Check Again</button>
    `;
    document.body.prepend(banner);

    document.getElementById("retryConnBtn").addEventListener("click", async () => {
      const btn = document.getElementById("retryConnBtn");
      btn.innerText = "Checking...";
      await checkServerHealth();
      if (isServerOnline) {
        showToast("Connected to document intelligence backend!", "success");
        await loadServerConfig();
        await loadChatList();
      } else {
        showToast("Service is still waking up... Please wait a few seconds and retry.", "info");
      }
      btn.innerText = "Check Again";
    });
  }
}

function removeOfflineBanner() {
  const banner = document.getElementById("offlineBanner");
  if (banner) banner.remove();
}

// --- Notification Toasts ---
function showToast(message, type = "info") {
  let toast = document.createElement("div");
  toast.style.cssText = `
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: ${type === "error" ? "rgba(239, 68, 68, 0.95)" : type === "success" ? "rgba(34, 197, 94, 0.95)" : "rgba(30, 41, 59, 0.95)"};
    border: 1px solid ${type === "error" ? "#f87171" : type === "success" ? "#4ade80" : "rgba(255,255,255,0.2)"};
    color: white;
    padding: 12px 18px;
    border-radius: 12px;
    font-size: 13.5px;
    font-weight: 500;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    z-index: 9999;
    animation: fadeIn 0.25s ease;
  `;
  toast.innerText = message;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// --- Accessible Dialog Functions ---
function openModal(modal) {
  if (!modal) return;
  lastFocusedElement = document.activeElement;
  modal.classList.add("active");
  const focusable = modal.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
  if (focusable) focusable.focus();
}

function closeModal(modal) {
  if (!modal) return;
  modal.classList.remove("active");
  if (lastFocusedElement && typeof lastFocusedElement.focus === "function") {
    lastFocusedElement.focus();
  }
}

// --- Event Listeners Setup ---
function setupEventListeners() {
  // Toggle Sidebar
  if (toggleSidebarBtn && sidebar) {
    toggleSidebarBtn.addEventListener("click", () => {
      sidebar.classList.toggle("collapsed");
    });
  }

  // New chat
  if (newChatBtn) {
    newChatBtn.addEventListener("click", createNewChat);
  }

  // Send message
  if (sendBtn) {
    sendBtn.addEventListener("click", handleSendMessage);
  }

  if (chatTextarea) {
    chatTextarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    chatTextarea.addEventListener("input", () => {
      chatTextarea.style.height = "auto";
      chatTextarea.style.height = `${Math.min(chatTextarea.scrollHeight, 120)}px`;
      updateSendButtonState();
    });
  }

  // Chip buttons click in Hero
  document.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const prompt = btn.getAttribute("data-prompt");
      if (chatTextarea) {
        chatTextarea.value = prompt;
        updateSendButtonState();
        handleSendMessage();
      }
    });
  });

  // Attach PDF Buttons (opens modal)
  if (attachBtn) {
    attachBtn.addEventListener("click", () => openModal(uploadModal));
  }
  const sidebarAttachBtn = document.getElementById("sidebarAttachBtn");
  if (sidebarAttachBtn) {
    sidebarAttachBtn.addEventListener("click", () => openModal(uploadModal));
  }
  const cardAttachPdf = document.getElementById("cardAttachPdf");
  if (cardAttachPdf) {
    cardAttachPdf.addEventListener("click", () => openModal(uploadModal));
  }
  if (closeUploadModalBtn) {
    closeUploadModalBtn.addEventListener("click", () => closeModal(uploadModal));
  }

  // Document Page Inspector Buttons
  if (openInspectorBtn) openInspectorBtn.addEventListener("click", openInspector);
  if (quickInspectBtn) quickInspectBtn.addEventListener("click", openInspector);
  if (closeInspectorModalBtn) closeInspectorModalBtn.addEventListener("click", () => closeModal(inspectorModal));

  // Settings Modal Buttons
  if (settingsBtn) settingsBtn.addEventListener("click", () => openModal(settingsModal));
  if (closeSettingsModalBtn) closeSettingsModalBtn.addEventListener("click", () => closeModal(settingsModal));

  if (saveSettingsBtn) {
    saveSettingsBtn.addEventListener("click", () => {
      userApiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
      if (userApiKey) {
        sessionStorage.setItem("docRag_apiKey", userApiKey);
      } else {
        sessionStorage.removeItem("docRag_apiKey");
      }

      if (apiBaseInput) {
        const newApiBase = apiBaseInput.value.trim().replace(/\/+$/, "");
        if (newApiBase) {
          if (!newApiBase.startsWith("https://") && !newApiBase.startsWith("http://localhost") && !newApiBase.startsWith("http://127.0.0.1")) {
            showToast("Backend URL must use HTTPS for remote hosts.", "error");
            return;
          }
          localStorage.setItem("docRag_apiBase", newApiBase);
        } else {
          localStorage.removeItem("docRag_apiBase");
        }
      }

      closeModal(settingsModal);
      showToast("Configuration saved! Reconnecting...", "success");
      setTimeout(() => window.location.reload(), 600);
    });
  }

  // Modal Backdrop Click & ESC to close
  document.querySelectorAll(".modal-overlay").forEach((modal) => {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal(modal);
    });
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      document.querySelectorAll(".modal-overlay.active").forEach(closeModal);
    }
  });

  // File Upload Drag & Drop & Browse
  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("click", (e) => e.stopPropagation());
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
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
        handleFilesUpload(e.dataTransfer.files);
      }
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length) {
        handleFilesUpload(fileInput.files);
      }
    });
  }

  // Inspector dropdown changes
  if (inspectDocSelect) inspectDocSelect.addEventListener("change", updateInspectorPages);
  if (inspectPageSelect) inspectPageSelect.addEventListener("change", updateInspectorText);

  // Export Button
  if (exportBtn) {
    exportBtn.addEventListener("click", handleExport);
  }
}

function updateSendButtonState() {
  if (!sendBtn) return;
  const hasText = chatTextarea && chatTextarea.value.trim().length > 0;
  sendBtn.disabled = !hasText || isSending;
}

// --- API Calls ---

async function loadChatList() {
  try {
    const res = await fetch(`${API_BASE}/api/chats`, {
      headers: { "x-owner-id": browserOwnerId },
    });
    if (!res.ok) throw new Error("Server returned non-200");
    const sessions = await res.json();
    renderChatList(sessions);

    if (sessions.length > 0) {
      const active = sessions.find((s) => s.is_active) || sessions[0];
      await switchChat(active.session_id);
    } else {
      await createNewChat();
    }
    isServerOnline = true;
    removeOfflineBanner();
  } catch (err) {
    console.warn("Could not load chat list. Backend may be offline:", err);
    isServerOnline = false;
    showOfflineBanner();
  }
}

// Safe DOM construction preventing HTML injection in title attributes
function renderChatList(sessions) {
  if (!chatListContainer) return;
  chatListContainer.innerHTML = "";
  sessions.forEach((s) => {
    const item = document.createElement("div");
    item.className = `chat-item ${s.session_id === currentSessionId ? "active" : ""}`;

    const left = document.createElement("div");
    left.className = "chat-item-left";

    const svgIcon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgIcon.setAttribute("viewBox", "0 0 24 24");
    svgIcon.setAttribute("fill", "none");
    svgIcon.setAttribute("stroke", "currentColor");
    svgIcon.setAttribute("stroke-width", "2");
    svgIcon.innerHTML = '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>';

    const titleSpan = document.createElement("span");
    titleSpan.title = String(s.title || "Chat");
    titleSpan.textContent = String(s.title || "Chat");

    left.appendChild(svgIcon);
    left.appendChild(titleSpan);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "chat-delete-btn";
    deleteBtn.title = "Delete chat";
    deleteBtn.setAttribute("aria-label", `Delete chat ${s.title}`);
    deleteBtn.innerHTML = "&times;";
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteChat(s.session_id);
    });

    item.appendChild(left);
    item.appendChild(deleteBtn);
    item.addEventListener("click", () => switchChat(s.session_id));
    chatListContainer.appendChild(item);
  });
}

async function createNewChat() {
  try {
    const res = await fetch(`${API_BASE}/api/chats`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-owner-id": browserOwnerId,
      },
      body: JSON.stringify({ title: "New Chat", owner_id: browserOwnerId }),
    });
    if (!res.ok) throw new Error("Could not create chat");
    const session = await res.json();
    currentSessionId = session.session_id;
    currentChatData = session;
    await loadChatList();
    renderCurrentChat();
    showToast("Started new chat session", "info");
  } catch (err) {
    showToast("The document service is temporarily unavailable. Please retry shortly.", "error");
  }
}

// Race-condition guarded chat switching
async function switchChat(sessionId) {
  const switchRequestId = ++activeChatSwitchRequestId;
  currentSessionId = sessionId;

  try {
    const res = await fetch(`${API_BASE}/api/chats/${sessionId}`, {
      headers: { "x-owner-id": browserOwnerId },
    });
    if (!res.ok) throw new Error("Failed to load session");
    const chatData = await res.json();

    // Discard stale responses if user switched chats again while request was in-flight
    if (switchRequestId !== activeChatSwitchRequestId) return;

    currentChatData = chatData;
    renderCurrentChat();

    // Update active highlight in sidebar
    document.querySelectorAll(".chat-item").forEach((el) => el.classList.remove("active"));
    const activeItem = [...chatListContainer.children].find((el) =>
      el.textContent.includes(chatData.title)
    );
    if (activeItem) activeItem.classList.add("active");
  } catch (err) {
    console.error("Failed to switch chat:", err);
  }
}

// Explicit status code check on chat deletion
async function deleteChat(sessionId) {
  if (!confirm("Are you sure you want to delete this chat session?")) return;
  try {
    const res = await fetch(`${API_BASE}/api/chats/${sessionId}`, {
      method: "DELETE",
      headers: { "x-owner-id": browserOwnerId },
    });
    if (!res.ok) {
      throw new Error("Server rejected deletion");
    }
    showToast("Chat deleted", "info");
    await loadChatList();
  } catch (err) {
    showToast("Could not delete chat session.", "error");
  }
}

function renderCurrentChat() {
  if (!currentChatData) return;

  // Render Documents in sidebar and active docs pill bar
  if (sidebarDocList) sidebarDocList.innerHTML = "";
  if (activeDocsBar) activeDocsBar.innerHTML = "";

  if (!currentChatData.documents || currentChatData.documents.length === 0) {
    if (sidebarDocList) {
      sidebarDocList.innerHTML = `<div style="font-size: 12px; color: var(--text-dim); padding: 8px 10px;">No PDFs in this chat.</div>`;
    }
    if (activeDocsBar) activeDocsBar.style.display = "none";
  } else {
    if (activeDocsBar) activeDocsBar.style.display = "flex";
    currentChatData.documents.forEach((doc) => {
      // Sidebar pill
      if (sidebarDocList) {
        const docDiv = document.createElement("div");
        docDiv.className = "doc-pill";
        docDiv.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
          <span title="${escapeHtml(doc.file_name)}">${escapeHtml(doc.file_name)} (${doc.total_pages}p)</span>
        `;
        sidebarDocList.appendChild(docDiv);
      }

      // Input bar tag
      if (activeDocsBar) {
        const tagDiv = document.createElement("div");
        tagDiv.className = "active-doc-tag";
        tagDiv.innerHTML = `📄 ${escapeHtml(doc.file_name)} <span style="opacity: 0.6;">(${doc.total_pages}p)</span>`;
        activeDocsBar.appendChild(tagDiv);
      }
    });
  }

  // Render Messages or Hero
  if (!currentChatData.messages || currentChatData.messages.length === 0) {
    if (heroCenter) heroCenter.style.display = "block";
    if (messagesList) messagesList.innerHTML = "";
  } else {
    if (heroCenter) heroCenter.style.display = "none";
    if (messagesList) {
      messagesList.innerHTML = "";
      currentChatData.messages.forEach((msg) => renderMessageRow(msg));
    }
  }

  scrollChatToBottom();
}

function renderMessageRow(msg) {
  if (!messagesList) return;
  const row = document.createElement("div");
  row.className = `message-row ${msg.role}`;

  let citationsHtml = "";
  if (msg.role === "assistant" && msg.citations && msg.citations.length > 0) {
    citationsHtml = `
      <div class="citations-container">
        <div class="citations-header">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>
          Source References (${msg.citations.length})
        </div>
        ${msg.citations
          .map(
            (c) => `
          <div class="citation-card" style="cursor: pointer;" onclick="inspectSpecificPage('${escapeHtml(c.document_name)}', ${parseInt(c.page_number) || 1})" title="Click to view this page in the Inspector">
            <div class="citation-meta">
              <span class="citation-badge">[Source ${parseInt(c.source_index) || 1}]</span>
              <span>📄 Page ${parseInt(c.page_number) || 1}</span>
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
        <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
      </div>
      <div class="message-bubble">
        <div>${formatMarkdownBasic(msg.content)}</div>
        ${citationsHtml}
      </div>
    `;
  }

  messagesList.appendChild(row);
}

// Interactive helper: Clicking a citation card opens the page inspector directly
window.inspectSpecificPage = function (docName, pageNum) {
  openInspector();
  if (inspectDocSelect && docName) {
    inspectDocSelect.value = docName;
    updateInspectorPages();
  }
  if (inspectPageSelect && pageNum) {
    inspectPageSelect.value = pageNum;
    updateInspectorText();
  }
};

// --- Sending Messages (With in-flight guard & session-race prevention) ---
async function handleSendMessage() {
  if (isSending) return;
  if (!chatTextarea) return;
  const text = chatTextarea.value.trim();
  if (!text) return;

  if (!currentSessionId) {
    await createNewChat();
  }

  if (!currentChatData || !currentChatData.documents || currentChatData.documents.length === 0) {
    showToast("Please attach at least one PDF document first!", "error");
    openModal(uploadModal);
    return;
  }

  const targetSessionId = currentSessionId;
  isSending = true;
  updateSendButtonState();

  // Clear input
  chatTextarea.value = "";
  chatTextarea.style.height = "auto";

  // Optimistically show user message
  if (heroCenter) heroCenter.style.display = "none";
  renderMessageRow({ role: "user", content: text });
  scrollChatToBottom();

  // Show live loading bubble
  const loadingRow = document.createElement("div");
  loadingRow.className = "message-row assistant";
  loadingRow.id = "tempLoadingRow";
  loadingRow.innerHTML = `
    <div class="avatar ai-avatar">✦</div>
    <div class="message-bubble" style="display: flex; align-items: center; gap: 8px;">
      <span style="font-size: 13px; color: var(--accent-purple);">Retrieving passages and generating answer...</span>
    </div>
  `;
  if (messagesList) messagesList.appendChild(loadingRow);
  scrollChatToBottom();

  try {
    const res = await fetch(`${API_BASE}/api/chats/${targetSessionId}/message`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-owner-id": browserOwnerId,
      },
      body: JSON.stringify({ question: text, api_key: userApiKey }),
    });

    const data = await res.json();
    loadingRow.remove();

    // Check if user is still on the same chat
    if (currentSessionId === targetSessionId) {
      if (res.ok && data.message) {
        renderMessageRow(data.message);
        await loadChatList();
      } else {
        renderMessageRow({
          role: "assistant",
          content: `⚠️ Error: ${data.detail || data.error || "Generation could not complete."}`,
        });
      }
    }
  } catch (err) {
    loadingRow.remove();
    if (currentSessionId === targetSessionId) {
      renderMessageRow({
        role: "assistant",
        content: `⚠️ The document service is temporarily unavailable. Please retry shortly.`,
      });
    }
  } finally {
    isSending = false;
    updateSendButtonState();
    scrollChatToBottom();
  }
}

// --- Uploading Files (With client-side validation & in-flight guard) ---
async function handleFilesUpload(files) {
  if (isUploading) return;
  if (!files || !files.length) return;

  const validFiles = [];
  const maxBytes = 10 * 1024 * 1024; // 10 MB

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      showToast(`Skipped '${file.name}': Only PDF files are supported.`, "error");
      continue;
    }
    if (file.size > maxBytes) {
      showToast(`Skipped '${file.name}': File exceeds 10 MB limit.`, "error");
      continue;
    }
    validFiles.push(file);
  }

  if (validFiles.length === 0) {
    if (fileInput) fileInput.value = "";
    return;
  }

  if (!currentSessionId) {
    await createNewChat();
  }

  const uploadSessionId = currentSessionId;
  isUploading = true;

  const formData = new FormData();
  for (let i = 0; i < validFiles.length; i++) {
    formData.append("files", validFiles[i]);
  }

  if (uploadStatusText) {
    uploadStatusText.innerText = "Indexing document(s)... Please wait...";
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout

  try {
    const res = await fetch(`${API_BASE}/api/chats/${uploadSessionId}/upload`, {
      method: "POST",
      headers: {
        "x-owner-id": browserOwnerId,
      },
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    const result = await res.json();
    if (res.ok) {
      if (uploadStatusText) {
        uploadStatusText.innerText = `Successfully indexed ${result.uploaded.length} document(s)!`;
      }
      showToast(`Attached ${result.uploaded.length} document(s) successfully!`, "success");
      setTimeout(() => {
        closeModal(uploadModal);
        if (uploadStatusText) uploadStatusText.innerText = "";
        if (fileInput) fileInput.value = "";
      }, 1000);

      // Only switch if user hasn't actively switched to another chat
      if (currentSessionId === uploadSessionId) {
        await switchChat(uploadSessionId);
      }
    } else {
      if (uploadStatusText) {
        uploadStatusText.innerText = `Error: ${result.detail || "Upload could not be processed."}`;
      }
      showToast(result.detail || "Upload could not be processed", "error");
    }
  } catch (err) {
    clearTimeout(timeoutId);
    const msg = err.name === "AbortError"
      ? "Upload timed out (60s). If the free tier container was waking up, please retry."
      : "The document service is temporarily unavailable. Please retry shortly.";
    if (uploadStatusText) {
      uploadStatusText.innerText = msg;
    }
    showToast(msg, "error");
  } finally {
    isUploading = false;
  }
}

// --- Document Page Inspector ---
function openInspector() {
  if (!currentChatData || !currentChatData.documents || currentChatData.documents.length === 0) {
    showToast("No PDFs attached in this chat yet. Attach a PDF first!", "error");
    openModal(uploadModal);
    return;
  }

  if (inspectDocSelect) {
    inspectDocSelect.innerHTML = "";
    currentChatData.documents.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = d.file_name;
      opt.innerText = `${d.file_name} (${d.total_pages} pages)`;
      inspectDocSelect.appendChild(opt);
    });
  }

  updateInspectorPages();
  openModal(inspectorModal);
}

function updateInspectorPages() {
  if (!inspectDocSelect || !inspectPageSelect) return;
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
  } else {
    for (let i = 1; i <= doc.total_pages; i++) {
      const opt = document.createElement("option");
      opt.value = i;
      opt.innerText = `Page ${i}`;
      inspectPageSelect.appendChild(opt);
    }
  }
  updateInspectorText();
}

function updateInspectorText() {
  if (!inspectDocSelect || !inspectPageSelect || !inspectTextarea) return;
  const selectedDocName = inspectDocSelect.value;
  const pageNum = parseInt(inspectPageSelect.value);
  const doc = currentChatData.documents.find((d) => d.file_name === selectedDocName);
  if (!doc || !doc.pages) {
    inspectTextarea.value = "(Text extracted and indexed in FAISS vector store)";
    return;
  }

  const page = doc.pages.find((p) => p.page_number === pageNum);
  inspectTextarea.value = page ? page.text : "(No content on this page)";
}

// --- Export Conversation (With explicit x-owner-id header & Blob download) ---
async function handleExport() {
  if (!currentSessionId) {
    showToast("No active chat to export. Click '+ New Chat' first.", "error");
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/chats/${currentSessionId}/export`, {
      headers: {
        "x-owner-id": browserOwnerId,
      },
    });

    if (!res.ok) {
      throw new Error("Export request failed");
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const rawTitle = (currentChatData && currentChatData.title) ? currentChatData.title : "transcript";
    const safeTitle = rawTitle.replace(/[^a-zA-Z0-9_-]/g, "_").substring(0, 30);

    const a = document.createElement("a");
    a.href = url;
    a.download = `docRag-${safeTitle}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);

    showToast("Downloaded Q&A transcript report.", "success");
  } catch (err) {
    showToast("Could not export chat transcript.", "error");
  }
}

// --- Helpers ---
function scrollChatToBottom() {
  if (chatScrollArea) {
    setTimeout(() => {
      chatScrollArea.scrollTop = chatScrollArea.scrollHeight;
    }, 50);
  }
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
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  formatted = formatted.replace(/\*(.*?)\*/g, "<em>$1</em>");
  formatted = formatted.replace(
    /\[Source\s*(\d+)\]/g,
    '<span class="badge-pill" style="font-size: 11px;">Source $1</span>'
  );
  formatted = formatted.replace(/\n/g, "<br>");
  return formatted;
}

// Start
document.addEventListener("DOMContentLoaded", initApp);
