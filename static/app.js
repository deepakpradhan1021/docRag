/**
 * DocRAG Client Application
 * Premium AI Document Assistant Frontend
 * Pure Vanilla JavaScript — fast, responsive, robust, and accessible.
 */

// --- Dynamic API Base Resolution ---
const isLocalFile = window.location.protocol === "file:";
const isNetlify = window.location.hostname.includes("netlify.app");
let customApiBase = localStorage.getItem("docRag_apiBase") || "";
const API_BASE = customApiBase || ((isLocalFile || isNetlify) ? "https://docrag-3gb4.onrender.com" : "");

// --- State ---
let currentSessionId = null;
let currentChatData = null;
let cachedSessions = [];

// Session-only storage: In memory for this browser tab only (cleared on tab close)
if (typeof localStorage !== "undefined" && localStorage.getItem("docRag_apiKey")) {
  localStorage.removeItem("docRag_apiKey");
}
let userApiKey = typeof sessionStorage !== "undefined" ? (sessionStorage.getItem("docRag_apiKey") || "") : "";

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
const sidebar = document.getElementById("sidebar");
const sidebarBackdrop = document.getElementById("sidebarBackdrop");
const toggleSidebarBtn = document.getElementById("toggleSidebarBtn");
const mobileMenuBtn = document.getElementById("mobileMenuBtn");
const newChatBtn = document.getElementById("newChatBtn");
const chatListContainer = document.getElementById("chatListContainer");
const sidebarDocList = document.getElementById("sidebarDocList");
const sidebarAttachBtn = document.getElementById("sidebarAttachBtn");
const sidebarStatusDot = document.getElementById("sidebarStatusDot");
const backendStatusPill = document.getElementById("backendStatusPill");

const navChatTitle = document.getElementById("navChatTitle");
const headerStatusBadge = document.getElementById("headerStatusBadge");
const headerStatusDot = document.getElementById("headerStatusDot");
const headerStatusText = document.getElementById("headerStatusText");

const threeDotBtn = document.getElementById("threeDotBtn");
const threeDotMenu = document.getElementById("threeDotMenu");
const menuInspectorBtn = document.getElementById("menuInspectorBtn");
const menuExportBtn = document.getElementById("menuExportBtn");
const menuClearChatBtn = document.getElementById("menuClearChatBtn");
const menuSettingsBtn = document.getElementById("menuSettingsBtn");
const menuAboutBtn = document.getElementById("menuAboutBtn");

const chatScrollArea = document.getElementById("chatScrollArea");
const heroCenter = document.getElementById("heroCenter");
const hero3DCanvas = document.getElementById("hero3DCanvas");
const heroStartChatBtn = document.getElementById("heroStartChatBtn");
const heroUploadPdfBtn = document.getElementById("heroUploadPdfBtn");
const cardAttachPdf = document.getElementById("cardAttachPdf");
const messagesList = document.getElementById("messagesList");

const activeDocsBar = document.getElementById("activeDocsBar");
const chatTextarea = document.getElementById("chatTextarea");
const attachBtn = document.getElementById("attachBtn");
const sendBtn = document.getElementById("sendBtn");

// Modals
const uploadModal = document.getElementById("uploadModal");
const closeUploadModalBtn = document.getElementById("closeUploadModalBtn");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseFilesBtn = document.getElementById("browseFilesBtn");
const uploadStatusText = document.getElementById("uploadStatusText");

const inspectorModal = document.getElementById("inspectorModal");
const closeInspectorModalBtn = document.getElementById("closeInspectorModalBtn");
const inspectDocSelect = document.getElementById("inspectDocSelect");
const inspectPageSelect = document.getElementById("inspectPageSelect");
const inspectTextarea = document.getElementById("inspectTextarea");

const settingsModal = document.getElementById("settingsModal");
const closeSettingsModalBtn = document.getElementById("closeSettingsModalBtn");
const apiKeyInput = document.getElementById("apiKeyInput");
const apiBaseInput = document.getElementById("apiBaseInput");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const clearApiKeyBtn = document.getElementById("clearApiKeyBtn");
const backendConnectionNote = document.getElementById("backendConnectionNote");

const aboutModal = document.getElementById("aboutModal");
const closeAboutModalBtn = document.getElementById("closeAboutModalBtn");

let lastFocusedElement = null;

// --- 3D Futuristic Hero Canvas Animation ---
let heroAnimationId = null;
let heroAnimRunning = false;

function init3DHeroCanvas() {
  if (!hero3DCanvas) return;
  const ctx = hero3DCanvas.getContext("2d");
  if (!ctx) return;

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let width = 480;
  let height = 150;
  let dpr = window.devicePixelRatio || 1;

  function resizeCanvas() {
    const parent = hero3DCanvas.parentElement;
    const parentW = parent ? parent.clientWidth : 0;
    const parentH = parent ? parent.clientHeight : 0;
    width = Math.max(parentW || 0, 320);
    height = Math.max(parentH || 0, 140);
    hero3DCanvas.width = width * dpr;
    hero3DCanvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvas();
  setTimeout(resizeCanvas, 50);
  setTimeout(resizeCanvas, 200);

  window.addEventListener("resize", () => {
    dpr = window.devicePixelRatio || 1;
    resizeCanvas();
    if (prefersReducedMotion) renderFrame();
  });

  // 3D Polyhedron geometry: 12 vertices of an icosahedron
  const phi = (1 + Math.sqrt(5)) / 2;
  const rawVertices = [
    [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
    [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
    [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
  ];

  // Normalize vertices to radius R
  const R = 34;
  const baseVertices = rawVertices.map(v => {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return [ (v[0] / len) * R, (v[1] / len) * R, (v[2] / len) * R ];
  });

  // Edges connecting nearby vertices
  const edges = [];
  for (let i = 0; i < baseVertices.length; i++) {
    for (let j = i + 1; j < baseVertices.length; j++) {
      const dx = baseVertices[i][0] - baseVertices[j][0];
      const dy = baseVertices[i][1] - baseVertices[j][1];
      const dz = baseVertices[i][2] - baseVertices[j][2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < R * 1.15) {
        edges.push([i, j]);
      }
    }
  }

  // Floating orbiting particles / shards
  const numParticles = 24;
  const particles = [];
  for (let p = 0; p < numParticles; p++) {
    const theta = Math.random() * Math.PI * 2;
    const rad = 45 + Math.random() * 70;
    particles.push({
      x: Math.cos(theta) * rad,
      y: (Math.random() - 0.5) * 50,
      z: Math.sin(theta) * rad,
      radius: 1.4 + Math.random() * 2.0,
      speed: 0.005 + Math.random() * 0.009,
      floatOffset: Math.random() * Math.PI * 2,
      floatSpeed: 0.02 + Math.random() * 0.02,
      color: Math.random() > 0.4 ? "#3ecf8e" : "#10b981"
    });
  }

  let angleX = 0.2;
  let angleY = 0;
  let targetAngleX = 0.2;
  let targetAngleY = 0;

  // Gentle mouse parallax
  window.addEventListener("mousemove", (e) => {
    const nx = (e.clientX / window.innerWidth) - 0.5;
    const ny = (e.clientY / window.innerHeight) - 0.5;
    targetAngleY = nx * 0.8;
    targetAngleX = 0.2 + ny * 0.5;
  }, { passive: true });

  let time = 0;

  function renderFrame() {
    if (!heroAnimRunning && !prefersReducedMotion) return;

    // Check if hero is hidden or tab in background
    if (document.hidden || (heroCenter && heroCenter.style.display === "none")) {
      heroAnimRunning = false;
      return;
    }

    ctx.clearRect(0, 0, width, height);

    const cx = width / 2;
    const cy = height / 2;
    const fl = 240; // Focal length

    if (!prefersReducedMotion) {
      time += 0.016;
      angleY += 0.008;
      // Damped lerp toward mouse target
      angleX += (targetAngleX - angleX) * 0.05;
      angleY += (targetAngleY - 0) * 0.02;
    }

    const cosX = Math.cos(angleX);
    const sinX = Math.sin(angleX);
    const cosY = Math.cos(angleY);
    const sinY = Math.sin(angleY);

    // Subtle breathing / floating scale
    const breath = 1 + Math.sin(time * 1.5) * 0.04;

    // Central glowing core
    const coreGrad = ctx.createRadialGradient(cx, cy, 2, cx, cy, 38 * breath);
    coreGrad.addColorStop(0, "rgba(220, 252, 231, 0.95)");
    coreGrad.addColorStop(0.3, "rgba(62, 207, 142, 0.55)");
    coreGrad.addColorStop(0.7, "rgba(16, 185, 129, 0.2)");
    coreGrad.addColorStop(1, "rgba(12, 13, 14, 0)");
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, 38 * breath, 0, Math.PI * 2);
    ctx.fill();

    // Project polyhedron vertices
    const projected = baseVertices.map(([bx, by, bz]) => {
      // Scale
      const sx = bx * breath;
      const sy = by * breath;
      const sz = bz * breath;

      // Rotate Y
      const x1 = sx * cosY + sz * sinY;
      const y1 = sy;
      const z1 = -sx * sinY + sz * cosY;

      // Rotate X
      const x2 = x1;
      const y2 = y1 * cosX - z1 * sinX;
      const z2 = y1 * sinX + z1 * cosX;

      // Project
      const scale = fl / (fl + z2 + 100);
      return {
        x: cx + x2 * scale,
        y: cy + y2 * scale,
        z: z2,
        scale
      };
    });

    // Draw polyhedron edges
    ctx.lineWidth = 1.3;
    edges.forEach(([i, j]) => {
      const p1 = projected[i];
      const p2 = projected[j];
      const avgZ = (p1.z + p2.z) / 2;
      const alpha = Math.max(0.15, Math.min(0.75, (avgZ + R) / (2 * R)));

      const edgeGrad = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
      edgeGrad.addColorStop(0, `rgba(62, 207, 142, ${alpha})`);
      edgeGrad.addColorStop(1, `rgba(16, 185, 129, ${alpha * 0.85})`);

      ctx.strokeStyle = edgeGrad;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    });

    // Draw vertices glowing points
    projected.forEach(p => {
      const pointAlpha = Math.max(0.3, Math.min(1, (p.z + R) / (2 * R)));
      ctx.fillStyle = `rgba(220, 252, 231, ${pointAlpha})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.4 * p.scale, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw floating orbiting particles & dust
    particles.forEach(pt => {
      if (!prefersReducedMotion) {
        pt.floatOffset += pt.floatSpeed;
      }
      const py = pt.y + Math.sin(pt.floatOffset) * 10;

      // Rotate Y
      const rx = pt.x * cosY + pt.z * sinY;
      const rz = -pt.x * sinY + pt.z * cosY;

      // Rotate X
      const ry = py * cosX - rz * sinX;
      const rzFinal = py * sinX + rz * cosX;

      const scale = fl / (fl + rzFinal + 120);
      const px = cx + rx * scale;
      const pyFinal = cy + ry * scale;

      const pAlpha = Math.max(0.2, Math.min(0.85, (rzFinal + 120) / 240));
      ctx.fillStyle = pt.color;
      ctx.globalAlpha = pAlpha;
      ctx.beginPath();
      ctx.arc(px, pyFinal, pt.radius * scale, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1.0;
    });

    if (!prefersReducedMotion) {
      heroAnimationId = requestAnimationFrame(renderFrame);
    }
  }

  heroAnimRunning = true;
  renderFrame();
}

function startHeroAnimation() {
  if (heroAnimationId) cancelAnimationFrame(heroAnimationId);
  heroAnimRunning = true;
  init3DHeroCanvas();
}

function stopHeroAnimation() {
  heroAnimRunning = false;
  if (heroAnimationId) {
    cancelAnimationFrame(heroAnimationId);
    heroAnimationId = null;
  }
}

// --- App Initialization ---
async function initApp() {
  setupEventListeners();

  if (userApiKey && apiKeyInput) apiKeyInput.value = userApiKey;
  if (customApiBase && apiBaseInput) apiBaseInput.value = customApiBase;

  updateConnectionStatusUI("Connecting...", false);

  // Start 3D canvas
  startHeroAnimation();

  await checkServerHealth();
  if (isServerOnline) {
    await loadChatList();
  } else {
    // Render initial empty chat view optimistically so user never stares at blank screen
    renderEmptyHeroState();
  }

  // Automatic background polling: check every 3.5s so when Render wakes up, it connects instantly!
  setInterval(async () => {
    if (!isServerOnline) {
      await checkServerHealth();
      if (isServerOnline) {
        await loadChatList();
      }
    }
  }, 3500);

  // Resume 3D canvas when tab becomes visible
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && heroCenter && heroCenter.style.display !== "none") {
      startHeroAnimation();
    } else {
      stopHeroAnimation();
    }
  });
}

// --- Connection Status UI ---
function updateConnectionStatusUI(statusText, isReady = false) {
  if (backendStatusPill) {
    backendStatusPill.innerText = statusText;
  }
  if (sidebarStatusDot) {
    sidebarStatusDot.className = `status-dot ${isReady ? "online" : ""}`;
  }
  if (headerStatusDot) {
    headerStatusDot.className = `status-dot ${isReady ? "online" : ""}`;
  }
  if (headerStatusText) {
    headerStatusText.innerText = isReady ? "Connected" : "Waking up...";
  }
  if (backendConnectionNote) {
    backendConnectionNote.innerText = `Status: ${statusText}`;
  }
}

// --- Health Check ---
async function checkServerHealth() {
  if (isHealthChecking) return;
  isHealthChecking = true;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);
    const res = await fetch(`${API_BASE}/api/health`, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      isServerOnline = true;
      removeOfflineBanner();
      updateConnectionStatusUI("Connected (Ready)", true);
    } else {
      isServerOnline = false;
      showOfflineBanner();
      updateConnectionStatusUI("Free-tier backend waking up (~30s)", false);
    }
  } catch (err) {
    isServerOnline = false;
    showOfflineBanner();
    updateConnectionStatusUI("Free-tier backend waking up (~30s)", false);
  } finally {
    isHealthChecking = false;
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
      padding: 9px 16px;
      font-size: 12.5px;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
      z-index: 200;
      position: sticky;
      top: 0;
    `;
    banner.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #fef08a;"></span>
        <span>Render free-tier backend is spinning up (~30-45 s cold start). Auto-connecting...</span>
      </div>
      <button id="retryConnBtn" style="background: white; color: #c2410c; border: none; padding: 3px 10px; border-radius: 5px; font-weight: 700; cursor: pointer; font-size: 11.5px;">Check</button>
    `;
    document.body.prepend(banner);

    const retryBtn = document.getElementById("retryConnBtn");
    if (retryBtn) {
      retryBtn.addEventListener("click", async () => {
        retryBtn.innerText = "Checking...";
        await checkServerHealth();
        if (isServerOnline) {
          showToast("Connected to DocRAG backend!", "success");
          await loadChatList();
        } else {
          showToast("Service still warming up... Please give it a few seconds.", "info");
        }
        retryBtn.innerText = "Check";
      });
    }
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
    background: ${type === "error" ? "rgba(239, 68, 68, 0.95)" : type === "success" ? "rgba(16, 185, 129, 0.95)" : "rgba(18, 20, 23, 0.95)"};
    border: 1px solid ${type === "error" ? "#f87171" : type === "success" ? "#3ecf8e" : "rgba(62, 207, 142, 0.4)"};
    color: white;
    padding: 10px 16px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    z-index: 9999;
    animation: fadeIn 0.2s ease;
  `;
  toast.innerText = message;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.25s";
    setTimeout(() => toast.remove(), 250);
  }, 3200);
}

// --- Modal Helpers ---
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
  // Mobile drawer toggle
  if (mobileMenuBtn && sidebar) {
    mobileMenuBtn.addEventListener("click", () => {
      sidebar.classList.toggle("open");
      document.body.classList.toggle("sidebar-open");
    });
  }

  if (toggleSidebarBtn && sidebar) {
    toggleSidebarBtn.addEventListener("click", () => {
      sidebar.classList.remove("open");
      document.body.classList.remove("sidebar-open");
    });
  }

  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener("click", () => {
      sidebar.classList.remove("open");
      document.body.classList.remove("sidebar-open");
    });
  }

  // New Chat button (Instant optimistic response!)
  if (newChatBtn) {
    newChatBtn.addEventListener("click", () => {
      createNewChat();
      closeMobileSidebar();
    });
  }

  // Hero CTAs
  if (heroStartChatBtn) {
    heroStartChatBtn.addEventListener("click", () => {
      if (chatTextarea) chatTextarea.focus();
    });
  }

  if (heroUploadPdfBtn) {
    heroUploadPdfBtn.addEventListener("click", () => openModal(uploadModal));
  }

  if (cardAttachPdf) {
    cardAttachPdf.addEventListener("click", () => openModal(uploadModal));
  }

  // Three-dot options menu
  if (threeDotBtn && threeDotMenu) {
    threeDotBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const isOpen = threeDotMenu.classList.contains("open");
      threeDotMenu.classList.toggle("open", !isOpen);
      threeDotBtn.setAttribute("aria-expanded", String(!isOpen));
    });

    document.addEventListener("click", (e) => {
      if (!threeDotMenu.contains(e.target) && e.target !== threeDotBtn) {
        threeDotMenu.classList.remove("open");
        threeDotBtn.setAttribute("aria-expanded", "false");
      }
    });
  }

  // Three-dot menu actions
  if (menuInspectorBtn) {
    menuInspectorBtn.addEventListener("click", () => {
      threeDotMenu.classList.remove("open");
      openInspector();
    });
  }

  if (menuExportBtn) {
    menuExportBtn.addEventListener("click", () => {
      threeDotMenu.classList.remove("open");
      handleExport();
    });
  }

  if (menuClearChatBtn) {
    menuClearChatBtn.addEventListener("click", () => {
      threeDotMenu.classList.remove("open");
      handleClearChat();
    });
  }

  if (menuSettingsBtn) {
    menuSettingsBtn.addEventListener("click", () => {
      threeDotMenu.classList.remove("open");
      openModal(settingsModal);
    });
  }

  if (menuAboutBtn) {
    menuAboutBtn.addEventListener("click", () => {
      threeDotMenu.classList.remove("open");
      openModal(aboutModal);
    });
  }

  if (closeAboutModalBtn) {
    closeAboutModalBtn.addEventListener("click", () => closeModal(aboutModal));
  }

  // Composer Input
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
      chatTextarea.style.height = `${Math.min(chatTextarea.scrollHeight, 140)}px`;
      updateSendButtonState();
    });
  }

  // Universal Click delegation for "Try asking" Suggestion Chips
  document.addEventListener("click", (e) => {
    const chip = e.target.closest(".suggestion-chip");
    if (!chip) return;
    const prompt = chip.getAttribute("data-prompt");
    if (prompt && chatTextarea) {
      chatTextarea.value = prompt;
      updateSendButtonState();
      handleSendMessage();
    }
  });

  // Attach PDF buttons
  if (attachBtn) {
    attachBtn.addEventListener("click", () => openModal(uploadModal));
  }
  if (sidebarAttachBtn) {
    sidebarAttachBtn.addEventListener("click", () => openModal(uploadModal));
  }
  if (closeUploadModalBtn) {
    closeUploadModalBtn.addEventListener("click", () => closeModal(uploadModal));
  }

  // Inspector Modal
  if (closeInspectorModalBtn) {
    closeInspectorModalBtn.addEventListener("click", () => closeModal(inspectorModal));
  }
  if (inspectDocSelect) inspectDocSelect.addEventListener("change", updateInspectorPages);
  if (inspectPageSelect) inspectPageSelect.addEventListener("change", updateInspectorText);

  // Settings Modal
  if (closeSettingsModalBtn) {
    closeSettingsModalBtn.addEventListener("click", () => closeModal(settingsModal));
  }

  if (clearApiKeyBtn) {
    clearApiKeyBtn.addEventListener("click", () => {
      sessionStorage.removeItem("docRag_apiKey");
      localStorage.removeItem("docRag_apiKey");
      userApiKey = "";
      if (apiKeyInput) apiKeyInput.value = "";
      showToast("API key cleared from this browser session.", "info");
    });
  }

  if (saveSettingsBtn) {
    saveSettingsBtn.addEventListener("click", () => {
      userApiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
      if (userApiKey) {
        sessionStorage.setItem("docRag_apiKey", userApiKey);
        showToast("Key saved in memory for this session only.", "success");
      } else {
        sessionStorage.removeItem("docRag_apiKey");
        showToast("Using default server environment key.", "info");
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
      setTimeout(() => checkServerHealth(), 300);
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
      if (threeDotMenu) threeDotMenu.classList.remove("open");
      closeMobileSidebar();
    }
  });

  // File Upload Drag & Drop & Browse
  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("click", (e) => e.stopPropagation());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--accent-green)";
    });
    dropzone.addEventListener("dragleave", () => {
      dropzone.style.borderColor = "rgba(62, 207, 142, 0.35)";
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "rgba(62, 207, 142, 0.35)";
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
}

function closeMobileSidebar() {
  if (sidebar) sidebar.classList.remove("open");
  document.body.classList.remove("sidebar-open");
}

function updateSendButtonState() {
  if (!sendBtn) return;
  const hasText = chatTextarea && chatTextarea.value.trim().length > 0;
  sendBtn.disabled = !hasText || isSending;
}

// --- Chat List & Session Management ---

async function loadChatList() {
  try {
    const res = await fetch(`${API_BASE}/api/chats`, {
      headers: { "x-owner-id": browserOwnerId },
    });
    if (!res.ok) throw new Error("Server returned non-200");
    const sessions = await res.json();
    cachedSessions = sessions;
    renderChatList(sessions);

    if (sessions.length > 0) {
      // Find active or first session
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
    renderEmptyHeroState();
  }
}

function renderChatList(sessions) {
  if (!chatListContainer) return;
  chatListContainer.innerHTML = "";

  if (!sessions || sessions.length === 0) {
    chatListContainer.innerHTML = `<div style="font-size: 12px; color: var(--text-dim); padding: 8px 10px;">No chats yet.</div>`;
    return;
  }

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
    item.addEventListener("click", () => {
      switchChat(s.session_id);
      closeMobileSidebar();
    });
    chatListContainer.appendChild(item);
  });
}

// Optimized Instant "New Chat" (0ms perceived UI latency!)
async function createNewChat() {
  // 1. Immediately switch UI to fresh empty state
  currentSessionId = null;
  currentChatData = {
    session_id: null,
    title: "New Chat",
    messages: [],
    documents: [],
  };
  renderEmptyHeroState();
  if (navChatTitle) navChatTitle.innerText = "New Chat";
  if (chatTextarea) chatTextarea.focus();

  // 2. Perform backend creation
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

    // Prepend to cached sessions and update sidebar without extra round-trips!
    cachedSessions = [
      {
        session_id: session.session_id,
        owner_id: session.owner_id,
        title: session.title,
        updated_at: session.updated_at,
        documents_count: 0,
        messages_count: 0,
        is_active: true,
      },
      ...cachedSessions.map(s => ({ ...s, is_active: false }))
    ];
    renderChatList(cachedSessions);
  } catch (err) {
    showToast("Server is waking up... You can start typing or upload a PDF.", "info");
  }
}

// Guarded chat switching
async function switchChat(sessionId) {
  if (!sessionId) return;
  const switchRequestId = ++activeChatSwitchRequestId;
  currentSessionId = sessionId;

  // Immediate sidebar UI highlight
  document.querySelectorAll(".chat-item").forEach((el) => el.classList.remove("active"));
  const items = chatListContainer ? chatListContainer.querySelectorAll(".chat-item") : [];
  items.forEach((item) => {
    if (item.querySelector(".chat-item-left span")?.textContent === currentChatData?.title) {
      item.classList.add("active");
    }
  });

  try {
    const res = await fetch(`${API_BASE}/api/chats/${sessionId}`, {
      headers: { "x-owner-id": browserOwnerId },
    });
    if (!res.ok) throw new Error("Failed to load session");
    const chatData = await res.json();

    if (switchRequestId !== activeChatSwitchRequestId) return;

    currentChatData = chatData;
    if (navChatTitle) navChatTitle.innerText = chatData.title || "DocRAG Workspace";
    renderCurrentChat();
    renderChatList(cachedSessions.map(s => ({ ...s, is_active: s.session_id === sessionId })));
  } catch (err) {
    console.error("Failed to switch chat:", err);
  }
}

// Delete chat
async function deleteChat(sessionId) {
  if (!confirm("Are you sure you want to delete this chat session?")) return;
  try {
    const res = await fetch(`${API_BASE}/api/chats/${sessionId}`, {
      method: "DELETE",
      headers: { "x-owner-id": browserOwnerId },
    });
    if (!res.ok) throw new Error("Server rejected deletion");

    showToast("Chat deleted", "info");
    cachedSessions = cachedSessions.filter(s => s.session_id !== sessionId);

    if (currentSessionId === sessionId) {
      if (cachedSessions.length > 0) {
        await switchChat(cachedSessions[0].session_id);
      } else {
        await createNewChat();
      }
    } else {
      renderChatList(cachedSessions);
    }
  } catch (err) {
    showToast("Could not delete chat session.", "error");
  }
}

// Clear messages in current conversation
async function handleClearChat() {
  if (!currentSessionId) {
    showToast("No active conversation to clear.", "info");
    return;
  }
  if (!confirm("Clear message history for this chat? Attached documents will be kept.")) return;

  try {
    const res = await fetch(`${API_BASE}/api/chats/${currentSessionId}/clear`, {
      method: "POST",
      headers: { "x-owner-id": browserOwnerId },
    });
    if (res.ok) {
      const data = await res.json();
      currentChatData = data.chat;
      renderCurrentChat();
      showToast("Conversation cleared", "info");
    } else {
      showToast("Could not clear conversation.", "error");
    }
  } catch (err) {
    showToast("Service temporarily unavailable.", "error");
  }
}

// --- Render Chat State ---

function renderEmptyHeroState() {
  if (heroCenter) heroCenter.style.display = "flex";
  startHeroAnimation();
  if (messagesList) messagesList.innerHTML = "";
  if (activeDocsBar) activeDocsBar.style.display = "none";
  if (sidebarDocList) {
    sidebarDocList.innerHTML = `<div class="sidebar-empty-docs">No PDFs in this chat.</div>`;
  }
}

function renderCurrentChat() {
  if (!currentChatData) {
    renderEmptyHeroState();
    return;
  }

  // Render Documents in sidebar and active docs pill bar
  if (sidebarDocList) sidebarDocList.innerHTML = "";
  if (activeDocsBar) activeDocsBar.innerHTML = "";

  const docs = currentChatData.documents || [];
  if (docs.length === 0) {
    if (sidebarDocList) {
      sidebarDocList.innerHTML = `<div class="sidebar-empty-docs">No PDFs in this chat.</div>`;
    }
    if (activeDocsBar) activeDocsBar.style.display = "none";
  } else {
    if (activeDocsBar) activeDocsBar.style.display = "flex";
    docs.forEach((doc) => {
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
        tagDiv.innerHTML = `📄 ${escapeHtml(doc.file_name)} <span style="opacity: 0.65;">(${doc.total_pages}p)</span>`;
        activeDocsBar.appendChild(tagDiv);
      }
    });
  }

  // Render Messages, Document Ready State, or Hero
  const messages = currentChatData.messages || [];
  if (messages.length === 0) {
    if (docs.length > 0) {
      // Document is indexed and ready!
      if (heroCenter) heroCenter.style.display = "none";
      stopHeroAnimation();
      if (messagesList) {
        messagesList.innerHTML = "";
        const docNames = docs.map(d => `${d.file_name} (${d.total_pages}p)`).join(", ");
        const welcomeCard = document.createElement("div");
        welcomeCard.className = "doc-ready-card";
        welcomeCard.innerHTML = `
          <div style="font-size: 26px; margin-bottom: 8px;">📄</div>
          <h3 style="font-size: 16px; margin: 0 0 6px 0; color: #ffffff;">Document Ready</h3>
          <p style="font-size: 13px; color: var(--text-muted); margin: 0 0 16px 0;">
            Indexed <strong style="color: var(--accent-lavender);">${escapeHtml(docNames)}</strong>. Click any prompt below to query immediately:
          </p>
          <div style="font-size: 12.5px; font-weight: 700; color: var(--accent-lavender); margin-bottom: 12px; display: flex; align-items: center; justify-content: center; gap: 6px;">
            <span>✨</span> Try asking:
          </div>
          <div class="chip-grid" style="margin-bottom: 0;">
            <button class="suggestion-chip" data-prompt="Summarize this document."><span class="chip-icon">📄</span> <span class="chip-text">Summarize this document</span></button>
            <button class="suggestion-chip" data-prompt="What are the main topics?"><span class="chip-icon">📌</span> <span class="chip-text">What are the main topics?</span></button>
            <button class="suggestion-chip" data-prompt="Explain this document in simple words."><span class="chip-icon">💡</span> <span class="chip-text">Explain in simple words</span></button>
            <button class="suggestion-chip" data-prompt="What are the important definitions?"><span class="chip-icon">📖</span> <span class="chip-text">Important definitions</span></button>
            <button class="suggestion-chip" data-prompt="Give me the key points."><span class="chip-icon">🎯</span> <span class="chip-text">Key points</span></button>
            <button class="suggestion-chip" data-prompt="Explain the most important concepts."><span class="chip-icon">🧠</span> <span class="chip-text">Important concepts</span></button>
            <button class="suggestion-chip" data-prompt="What are the advantages and disadvantages?"><span class="chip-icon">⚖️</span> <span class="chip-text">Pros & cons</span></button>
            <button class="suggestion-chip" data-prompt="Generate short study notes."><span class="chip-icon">📝</span> <span class="chip-text">Generate study notes</span></button>
          </div>
        `;
        messagesList.appendChild(welcomeCard);
      }
    } else {
      renderEmptyHeroState();
    }
  } else {
    if (heroCenter) heroCenter.style.display = "none";
    stopHeroAnimation();
    if (messagesList) {
      messagesList.innerHTML = "";
      messages.forEach((msg) => renderMessageRow(msg));
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
          <div style="display: flex; align-items: center; gap: 6px;">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>
            <span>Source References (${msg.citations.length})</span>
          </div>
          <button class="citations-toggle-btn" type="button" onclick="toggleCitations(this)">Collapse ▴</button>
        </div>
        <div class="citations-list">
          ${msg.citations
            .map(
              (c) => `
            <div class="citation-card" style="cursor: pointer;" onclick="inspectSpecificPage('${escapeHtml(c.document_name)}', ${parseInt(c.page_number) || 1})" title="Click to view full page in the Inspector">
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
      </div>
    `;
  }

  if (msg.role === "user") {
    row.innerHTML = `
      <div class="message-bubble">${escapeHtml(msg.content)}</div>
      <div class="avatar user-avatar" title="You">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
      </div>
    `;
  } else {
    row.innerHTML = `
      <div class="avatar ai-avatar" title="DocRAG">
        <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
      </div>
      <div class="message-bubble">
        <div>${formatMarkdownRich(msg.content)}</div>
        ${citationsHtml}
      </div>
    `;
  }

  messagesList.appendChild(row);
}

// Citations toggle helper
window.toggleCitations = function (btn) {
  const container = btn.closest(".citations-container");
  if (!container) return;
  const list = container.querySelector(".citations-list");
  if (!list) return;
  if (list.style.display === "none") {
    list.style.display = "flex";
    btn.textContent = "Collapse ▴";
  } else {
    list.style.display = "none";
    btn.textContent = "Expand ▾";
  }
};

// Inspector page helper
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

// --- Sending Messages ---

async function handleSendMessage() {
  if (isSending) return;
  if (!chatTextarea) return;
  const text = chatTextarea.value.trim();
  if (!text) return;

  if (!currentSessionId) {
    await createNewChat();
  }

  // Ensure documents exist
  if (!currentChatData || !currentChatData.documents || currentChatData.documents.length === 0) {
    showToast("Please attach a PDF document first!", "info");
    openModal(uploadModal);
    return;
  }

  const targetSessionId = currentSessionId;
  isSending = true;
  updateSendButtonState();

  // Reset textarea height
  chatTextarea.value = "";
  chatTextarea.style.height = "auto";

  // Hide hero
  if (heroCenter) heroCenter.style.display = "none";
  stopHeroAnimation();

  // Optimistically append user message
  renderMessageRow({ role: "user", content: text });
  scrollChatToBottom();

  // Generating indicator bubble
  const loadingRow = document.createElement("div");
  loadingRow.className = "message-row assistant";
  loadingRow.id = "tempLoadingRow";
  loadingRow.innerHTML = `
    <div class="avatar ai-avatar">✦</div>
    <div class="message-bubble" style="display: flex; align-items: center; gap: 10px;">
      <div class="generating-dots">
        <div class="generating-dot"></div>
        <div class="generating-dot"></div>
        <div class="generating-dot"></div>
      </div>
      <span style="font-size: 13px; color: var(--accent-lavender);">Analyzing document and preparing grounded response...</span>
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

    if (currentSessionId === targetSessionId) {
      if (res.ok && data.message) {
        renderMessageRow(data.message);
        // Update cached session message count
        const cached = cachedSessions.find(s => s.session_id === targetSessionId);
        if (cached) cached.messages_count = (cached.messages_count || 0) + 2;
      } else {
        renderMessageRow({
          role: "assistant",
          content: `⚠️ ${data.detail || data.error || "Generation could not complete. Please retry."}`,
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

// --- Uploading Files ---

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
  const timeoutId = setTimeout(() => controller.abort(), 60000);

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
      }, 900);

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
      ? "Upload timed out (60s). Please retry."
      : "The document service is temporarily unavailable. Please retry shortly.";
    if (uploadStatusText) uploadStatusText.innerText = msg;
    showToast(msg, "error");
  } finally {
    isUploading = false;
  }
}

// --- Document Page Inspector ---

function openInspector() {
  if (!currentChatData || !currentChatData.documents || currentChatData.documents.length === 0) {
    showToast("No PDFs attached in this chat yet. Attach a PDF first!", "info");
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
    inspectTextarea.value = "(Text extracted and indexed in FAISS / BM25 index)";
    return;
  }

  const page = doc.pages.find((p) => p.page_number === pageNum);
  inspectTextarea.value = page ? page.text : "(No content on this page)";
}

// --- Export Conversation ---

async function handleExport() {
  if (!currentSessionId) {
    showToast("No active conversation to export.", "info");
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/chats/${currentSessionId}/export`, {
      headers: { "x-owner-id": browserOwnerId },
    });
    if (!res.ok) throw new Error("Export failed");

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const rawTitle = (currentChatData && currentChatData.title) ? currentChatData.title : "transcript";
    const safeTitle = rawTitle.replace(/[^a-zA-Z0-9_-]/g, "_").substring(0, 30);

    const a = document.createElement("a");
    a.href = url;
    a.download = `DocRAG-${safeTitle}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);

    showToast("Downloaded Q&A transcript report.", "success");
  } catch (err) {
    showToast("Could not export chat transcript.", "error");
  }
}

// --- Helper Functions ---

function scrollChatToBottom() {
  if (!chatScrollArea) return;
  setTimeout(() => {
    const lastChild = messagesList ? messagesList.lastElementChild : null;
    if (lastChild) {
      lastChild.scrollIntoView({ behavior: "smooth", block: "end" });
    } else {
      chatScrollArea.scrollTop = chatScrollArea.scrollHeight;
    }
  }, 50);
}

function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function copyCodeToClipboard(button, codeText) {
  navigator.clipboard.writeText(codeText).then(() => {
    const original = button.innerText;
    button.innerText = "Copied!";
    setTimeout(() => {
      button.innerText = original;
    }, 2000);
  }).catch(() => {
    button.innerText = "Copied!";
    setTimeout(() => { button.innerText = "Copy"; }, 2000);
  });
}
window.copyCodeToClipboard = copyCodeToClipboard;

function formatMarkdownRich(text) {
  if (!text) return "";
  let formatted = escapeHtml(text);

  // Fenced Code Blocks with copy button
  formatted = formatted.replace(/```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const displayLang = lang ? lang.toUpperCase() : "CODE";
    const encodedCode = encodeURIComponent(code.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">"));
    return `
      <div class="code-block-wrapper" style="margin: 10px 0;">
        <div class="code-block-header">
          <span>${displayLang}</span>
          <button class="code-copy-btn" type="button" onclick="copyCodeToClipboard(this, decodeURIComponent('${encodedCode}'))">Copy</button>
        </div>
        <pre><code>${code.trim()}</code></pre>
      </div>
    `;
  });

  // Inline code
  formatted = formatted.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Headings
  formatted = formatted.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  formatted = formatted.replace(/^## (.*$)/gim, "<h2>$1</h2>");
  formatted = formatted.replace(/^# (.*$)/gim, "<h2>$1</h2>");

  // Bold & Italics
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  formatted = formatted.replace(/\*(.*?)\*/g, "<em>$1</em>");

  // Source badges
  formatted = formatted.replace(
    /\[Source\s*(\d+)\]/g,
    '<span class="citation-badge">Source $1</span>'
  );

  // Line breaks
  formatted = formatted.replace(/\n\n/g, "<br><br>");
  formatted = formatted.replace(/\n/g, "<br>");

  return formatted;
}

// Start on DOM ready
document.addEventListener("DOMContentLoaded", initApp);
