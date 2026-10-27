// ==========================================================================
// NeuroLearn AI Studio - Client Controller
// ==========================================================================

// State Management
const state = {
  activeTab: 'chat',
  activeDocId: null,
  documents: [],
  currentProvider: 'gemini',
  chatHistory: [],
  currentFlashcards: [],
  flashcardIndex: 0,
  generatedQAs: [],
  quizScore: { correct: 0, total: 0 },
  currentMcqs: [],
  quizIndex: 0,
  quizAnswers: {},
  quizMode: 'focus',
  chatBusy: false,
  citationRegistry: {},
  citationSequence: 0,
  bookmarkedQAs: new Set(JSON.parse(localStorage.getItem('neurolearn_bookmarks') || '[]')),
  testPaper: null
};

// DOM References
const docCountEl = document.getElementById('docCount');
const documentListEl = document.getElementById('documentList');
const currentProviderNameEl = document.getElementById('currentProviderName');
const activeDocTitleDisplay = document.getElementById('activeDocTitleDisplay');
const dropzone = document.getElementById('dropzone');
const pdfFileInput = document.getElementById('pdfFileInput');
const toastLoading = document.getElementById('toastLoading');
const toastLoadingText = document.getElementById('toastLoadingText');
const appToast = document.getElementById('appToast');
const documentSearchInput = document.getElementById('documentSearchInput');
const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
const sidebarBackdrop = document.getElementById('sidebarBackdrop');

// Chat DOM
const chatMessagesEl = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendChatBtn = document.getElementById('sendChatBtn');
const scrollLatestBtn = document.getElementById('scrollLatestBtn');

// Q&A DOM
const generateQaBtn = document.getElementById('generateQaBtn');
const qaResultsEl = document.getElementById('qaResults');
const exportQaJsonBtn = document.getElementById('exportQaJson');
const exportQaCsvBtn = document.getElementById('exportQaCsv');
const exportQaMdBtn = document.getElementById('exportQaMd');
const generatePaperBtn = document.getElementById('generatePaperBtn');
const testPaperResults = document.getElementById('testPaperResults');
const paperSummary = document.getElementById('paperSummary');

// MCQ DOM
const generateMcqBtn = document.getElementById('generateMcqBtn');
const mcqResultsEl = document.getElementById('mcqResults');
const quizScoreDisplay = document.getElementById('quizScoreDisplay');
const quizScoreVal = document.getElementById('quizScoreVal');
const quizModeToggle = document.getElementById('quizModeToggle');

// Flashcards DOM
const generateFlashcardsBtn = document.getElementById('generateFlashcardsBtn');
const flashcardsWrapperEl = document.getElementById('flashcardsWrapper');

// Summary DOM
const generateSummaryBtn = document.getElementById('generateSummaryBtn');
const summaryResultsEl = document.getElementById('summaryResults');

// Settings Modal DOM
const openSettingsBtn = document.getElementById('openSettingsBtn');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');
const settingsModal = document.getElementById('settingsModal');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');
const modalLlmProvider = document.getElementById('modalLlmProvider');
const modalGeminiKey = document.getElementById('modalGeminiKey');
const modalGroqKey = document.getElementById('modalGroqKey');
const modalOpenAiKey = document.getElementById('modalOpenAiKey');
const modalEmbProvider = document.getElementById('modalEmbProvider');
const sidebarEmbProvider = document.getElementById('sidebarEmbProvider');
const modalSalienceMethod = document.getElementById('modalSalienceMethod');
const sidebarSalienceMethod = document.getElementById('sidebarSalienceMethod');

// Helper: Show/Hide Loading Toast
function showLoading(msg = 'Processing...') {
  toastLoadingText.textContent = msg;
  toastLoading.style.display = 'flex';
}

function hideLoading() {
  toastLoading.style.display = 'none';
}

async function parseResponse(res) {
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) { /* handled below */ }
  if (!res.ok) {
    const detail = data?.detail || data?.message || text.replace(/\s+/g, ' ').trim();
    throw new Error(detail || `Request failed (${res.status})`);
  }
  if (!data) throw new Error('The server returned an invalid response. Please retry.');
  return data;
}

// Initialization
document.addEventListener('DOMContentLoaded', async () => {
  // Prefer a useful large-document default while still allowing the learner
  // to choose a smaller count from the expanded control.
  const qaCountControl = document.getElementById('qaCountSelect');
  const mcqCountControl = document.getElementById('mcqCountSelect');
  if (qaCountControl) qaCountControl.value = '10';
  if (mcqCountControl) mcqCountControl.value = '10';
  setupThemeToggle();
  setupTabs();
  setupIngestSubtabs();
  setupUploadHandlers();
  setupSettingsModal();
  setupKeyboardShortcuts();
  setupAutoResizeTextarea();
  setupResponsiveShell();
  setupSourceDrawer();
  setupDocumentFilter();
  setupChatScroll();
  setupTestPaper();
  await refreshConfig();
  await refreshDocuments();
});

function escapeHtml(value = '') {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

let toastTimer;
function showToast(message, type = 'success') {
  if (!appToast) return;
  clearTimeout(toastTimer);
  appToast.className = `app-toast ${type} visible`;
  appToast.innerHTML = `<span class="toast-dot"></span><span>${escapeHtml(message)}</span>`;
  toastTimer = setTimeout(() => appToast.classList.remove('visible'), 3200);
}

function renderErrorState(target, title, message) {
  target.innerHTML = `
    <div class="empty-state error-state" role="alert">
      <div class="empty-icon-wrap">!</div>
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(message)}</p>
    </div>`;
}

function renderSkeletons(target, count = 3) {
  target.innerHTML = Array.from({ length: count }, () => `
    <div class="skeleton-card" aria-hidden="true">
      <div class="skeleton-row"><span></span><span></span></div>
      <div class="skeleton-line wide"></div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line short"></div>
    </div>`).join('');
}

// Theme Management (Light / Dark)
function setupThemeToggle() {
  const toggleBtn = document.getElementById('themeToggleBtn');
  const sunIcon = document.getElementById('themeIconSun');
  const moonIcon = document.getElementById('themeIconMoon');

  const savedTheme = localStorage.getItem('neurolearn_theme') || 'dark';
  applyTheme(savedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      localStorage.setItem('neurolearn_theme', next);
    });
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (theme === 'light') {
      if (sunIcon) sunIcon.style.display = 'none';
      if (moonIcon) moonIcon.style.display = 'block';
    } else {
      if (sunIcon) sunIcon.style.display = 'block';
      if (moonIcon) moonIcon.style.display = 'none';
    }
  }
}

// Setup Main Tabs
function setupTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const target = tab.dataset.tab;
      state.activeTab = target;
      const targetPane = document.getElementById(`pane-${target}`);
      if (targetPane) targetPane.classList.add('active');
    });
  });
}

// Ingest Subtabs (PDF vs URL vs Raw)
function setupIngestSubtabs() {
  const subtabs = document.querySelectorAll('.subtab-btn');
  subtabs.forEach(btn => {
    btn.addEventListener('click', () => {
      subtabs.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.subtab-content').forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const contentEl = document.getElementById(`subtab-${btn.dataset.subtab}`);
      if (contentEl) contentEl.classList.add('active');
    });
  });
}

// Upload & Ingest Handlers
function setupUploadHandlers() {
  if (!dropzone || !pdfFileInput) {
    console.error('Upload controls are missing from the page.');
    return;
  }
  dropzone.addEventListener('click', () => pdfFileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      handlePdfUpload(e.dataTransfer.files[0]);
    }
  });

  pdfFileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handlePdfUpload(e.target.files[0]);
    }
  });

  // URL Ingest
  document.getElementById('scrapeUrlBtn').addEventListener('click', async () => {
    const url = document.getElementById('urlInput').value.trim();
    if (!url) return alert('Please enter a valid URL.');
    const salienceMethod = sidebarSalienceMethod ? sidebarSalienceMethod.value : 'tfidf';
    const embeddingProvider = sidebarEmbProvider ? sidebarEmbProvider.value : 'bert';
    showLoading(`Scraping & indexing webpage (${salienceMethod}, ${embeddingProvider})...`);
    try {
      const res = await fetch('/api/documents/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, salience_method: salienceMethod, embedding_provider: embeddingProvider })
      });
      const data = await parseResponse(res);
      document.getElementById('urlInput').value = '';
      await refreshDocuments();
      selectDocument(data.document.doc_id);
      appendChatMessage('assistant', `Webpage **${data.document.title}** indexed with ${salienceMethod.toUpperCase()} (${embeddingProvider})! (${data.document.total_chunks} chunks). Ready for questions.`);
    } catch (err) {
      alert(err.message);
    } finally {
      hideLoading();
    }
  });

  // Raw Text Ingest
  document.getElementById('indexTextBtn').addEventListener('click', async () => {
    const title = document.getElementById('rawTitleInput').value.trim() || 'Pasted Notes';
    const text = document.getElementById('rawTextInput').value.trim();
    if (!text) return alert('Please enter some text to index.');
    const salienceMethod = sidebarSalienceMethod ? sidebarSalienceMethod.value : 'tfidf';
    const embeddingProvider = sidebarEmbProvider ? sidebarEmbProvider.value : 'bert';
    showLoading(`Chunking and embedding text (${salienceMethod}, ${embeddingProvider})...`);
    try {
      const res = await fetch('/api/documents/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, text, salience_method: salienceMethod, embedding_provider: embeddingProvider })
      });
      const data = await parseResponse(res);
      document.getElementById('rawTextInput').value = '';
      await refreshDocuments();
      selectDocument(data.document.doc_id);
      appendChatMessage('assistant', `Document **${title}** indexed with ${salienceMethod.toUpperCase()} (${embeddingProvider})! (${data.document.total_chunks} chunks). Ready for study.`);
    } catch (err) {
      alert(err.message);
    } finally {
      hideLoading();
    }
  });
}

async function handlePdfUpload(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    return alert('Only PDF files are supported.');
  }
  const salienceMethod = sidebarSalienceMethod ? sidebarSalienceMethod.value : 'tfidf';
  const embeddingProvider = sidebarEmbProvider ? sidebarEmbProvider.value : 'bert';
  const formData = new FormData();
  formData.append('file', file);
  formData.append('salience_method', salienceMethod);
  formData.append('embedding_provider', embeddingProvider);

  showLoading(`Extracting & indexing ${file.name} (${salienceMethod}, ${embeddingProvider})...`);
  dropzone.classList.add('is-uploading');
  try {
    const res = await fetch('/api/documents/upload', {
      method: 'POST',
      body: formData
    });
    const data = await parseResponse(res);
    await refreshDocuments();
    selectDocument(data.document.doc_id);
    appendChatMessage('assistant', `PDF **${data.document.title}** uploaded successfully using ${salienceMethod.toUpperCase()} concept extraction! Analyzed ${data.document.total_pages} page(s) into ${data.document.total_chunks} chunks.`);
  } catch (err) {
    alert('Upload error: ' + err.message);
  } finally {
    hideLoading();
    dropzone.classList.remove('is-uploading');
    pdfFileInput.value = '';
  }
}

// Fetch and render document list
async function refreshDocuments() {
  try {
    const res = await fetch('/api/documents');
    const data = await res.json();
    state.documents = data.documents || [];
    docCountEl.textContent = state.documents.length;
    renderDocumentList();

    // Default select first doc if none selected
    if (!state.activeDocId && state.documents.length > 0) {
      selectDocument(state.documents[0].doc_id);
    }
  } catch (err) {
    console.error('Failed to list documents:', err);
  }
}

function renderDocumentList() {
  const query = (documentSearchInput?.value || '').trim().toLowerCase();
  const documents = state.documents.filter(doc => (doc.title || '').toLowerCase().includes(query));

  if (state.documents.length === 0) {
    documentListEl.innerHTML = `
      <div class="empty-docs">
        <div class="empty-docs-icon">📄</div>
        <p>No documents indexed yet</p>
        <span>Add a PDF, webpage, or notes to begin.</span>
      </div>`;
    state.activeDocId = null;
    if (activeDocTitleDisplay) activeDocTitleDisplay.textContent = 'No document selected';
    return;
  }

  if (documents.length === 0) {
    documentListEl.innerHTML = `<div class="empty-docs compact"><p>No matching documents</p><span>Try a different search term.</span></div>`;
    return;
  }

  documentListEl.innerHTML = documents.map(doc => `
    <button class="doc-card ${state.activeDocId === doc.doc_id ? 'selected' : ''}" onclick="selectDocument('${doc.doc_id}')" title="${escapeHtml(doc.title)}">
      <span class="doc-file-icon" aria-hidden="true">${doc.source_type === 'url' ? '↗' : doc.source_type === 'text' ? 'T' : 'P'}</span>
      <span class="doc-info">
        <span class="doc-title">${escapeHtml(doc.title)}</span>
        <span class="doc-meta">${doc.total_pages} ${doc.total_pages === 1 ? 'page' : 'pages'} · ${doc.total_chunks} chunks</span>
      </span>
      <span class="delete-doc-btn" role="button" tabindex="0" onclick="event.stopPropagation(); deleteDocument('${doc.doc_id}')" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();event.stopPropagation();deleteDocument('${doc.doc_id}')}" aria-label="Delete ${escapeHtml(doc.title)}" title="Delete document">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
      </span>
    </button>`).join('');
}

function setupDocumentFilter() {
  documentSearchInput?.addEventListener('input', renderDocumentList);
}

function selectDocument(docId) {
  state.activeDocId = docId;
  const cards = document.querySelectorAll('.doc-card');
  cards.forEach(c => c.classList.remove('selected'));
  const selectedDoc = state.documents.find(d => d.doc_id === docId);

  if (activeDocTitleDisplay && selectedDoc) {
    activeDocTitleDisplay.textContent = `${selectedDoc.title} (${selectedDoc.total_pages} pg, ${selectedDoc.total_chunks} chunks)`;
  }

  // Update selected class without rebuilding full DOM
  const matchingCard = Array.from(cards).find(c => c.getAttribute('onclick')?.includes(docId));
  if (matchingCard) matchingCard.classList.add('selected');
  if (window.innerWidth <= 860) closeMobileSidebar();
}

async function deleteDocument(docId) {
  if (!confirm('Are you sure you want to remove this document from the vector store?')) return;
  try {
    await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
    if (state.activeDocId === docId) {
      state.activeDocId = null;
      if (activeDocTitleDisplay) activeDocTitleDisplay.textContent = 'No document selected';
    }
    await refreshDocuments();
  } catch (err) {
    alert('Delete error: ' + err.message);
  }
}

// RAG Chat System
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
sendChatBtn.addEventListener('click', sendMessage);

// Auto-resizing textarea
function setupAutoResizeTextarea() {
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
  });
}

function setupResponsiveShell() {
  const sidebar = document.querySelector('.sidebar');
  const open = () => {
    sidebar?.classList.add('mobile-open');
    sidebarBackdrop?.classList.add('visible');
    sidebarToggleBtn?.setAttribute('aria-expanded', 'true');
  };
  sidebarToggleBtn?.addEventListener('click', () => {
    if (sidebar?.classList.contains('mobile-open')) closeMobileSidebar();
    else open();
  });
  sidebarBackdrop?.addEventListener('click', closeMobileSidebar);
  window.addEventListener('resize', () => {
    if (window.innerWidth > 860) closeMobileSidebar();
  });
}

function closeMobileSidebar() {
  document.querySelector('.sidebar')?.classList.remove('mobile-open');
  sidebarBackdrop?.classList.remove('visible');
  sidebarToggleBtn?.setAttribute('aria-expanded', 'false');
}

function setupSourceDrawer() {
  const drawer = document.getElementById('sourceDrawer');
  if (!drawer) return;
  drawer.hidden = true;
  document.getElementById('closeSourceDrawerBtn')?.addEventListener('click', closeSourceDrawer);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && drawer?.classList.contains('open')) closeSourceDrawer();
  });
}

function closeSourceDrawer() {
  const drawer = document.getElementById('sourceDrawer');
  if (!drawer) return;
  drawer.classList.remove('open');
  drawer.setAttribute('aria-hidden', 'true');
  drawer.hidden = true;
}

function setupChatScroll() {
  if (!chatMessagesEl || !scrollLatestBtn) return;
  const update = () => {
    const gap = chatMessagesEl.scrollHeight - chatMessagesEl.scrollTop - chatMessagesEl.clientHeight;
    scrollLatestBtn.classList.toggle('visible', gap > 120);
  };
  chatMessagesEl.addEventListener('scroll', update, { passive: true });
  scrollLatestBtn.addEventListener('click', () => {
    chatMessagesEl.scrollTo({ top: chatMessagesEl.scrollHeight, behavior: 'smooth' });
  });
}

// Quick Prompt action
window.quickPrompt = function(text) {
  chatInput.value = text;
  chatInput.focus();
  sendMessage();
};

async function sendMessage() {
  const query = chatInput.value.trim();
  if (!query) return;
  if (!state.activeDocId) {
    showToast('Select or upload a document before chatting.', 'error');
    return;
  }
  if (state.chatBusy) return;
  state.chatBusy = true;
  chatInput.disabled = true;
  sendChatBtn.disabled = true;

  // Clear welcome hero if present on first message
  const hero = chatMessagesEl.querySelector('.welcome-hero');
  if (hero) hero.remove();

  appendChatMessage('user', query);
  state.chatHistory.push({ role: 'user', content: query });
  chatInput.value = '';
  chatInput.style.height = 'auto';

  const assistantBubble = appendChatMessage('assistant', 'Searching verified document chunks with BERT...');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        doc_id: state.activeDocId,
        chat_history: state.chatHistory,
        top_k: 5
      })
    });
    const data = await parseResponse(res);

    state.chatHistory.push({ role: 'assistant', content: data.answer });
    updateAssistantMessage(assistantBubble, data.answer, data.citations, data.confidence_level);
  } catch (err) {
    const errorText = `Error: ${err.message || 'Chat request failed. Please retry.'}`;
    assistantBubble.innerHTML = `<p class="chat-error">${escapeHtml(errorText)}</p>`;
    showToast(err.message || 'Chat request failed. Please retry.', 'error');
  } finally {
    state.chatBusy = false;
    chatInput.disabled = false;
    sendChatBtn.disabled = false;
    chatInput.focus();
  }
}

function appendChatMessage(role, content) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.innerHTML = `<p>${formatMarkdown(content)}</p>`;
  msgDiv.appendChild(bubble);
  chatMessagesEl.appendChild(msgDiv);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  return bubble;
}

function updateAssistantMessage(bubble, answer, citations, confidence) {
  let html = `<p>${formatMarkdown(answer)}</p>`;
  if (citations && citations.length > 0) {
    html += `<div class="citations-panel">`;
    citations.forEach(c => {
      const pct = Math.round((c.relevance_score || 0.85) * 100);
      const excerpt = (c.excerpt || '').replace(/"/g, '&quot;');
      html += `<span class="citation-badge" title="Match: ${pct}% | Excerpt: ${excerpt}">Page ${c.page_number} (${pct}%)</span>`;
    });
    html += `</div>`;
  }
  bubble.innerHTML = html;
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

// Markdown formatting helper
function formatMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

// Q&A Generator
generateQaBtn.addEventListener('click', async () => {
  if (!state.activeDocId) return alert('Please upload or select a document first.');

  const count = parseInt(document.getElementById('qaCountSelect').value, 10);
  const difficulty = document.getElementById('qaDifficultySelect').value || null;
  const bloom = document.getElementById('qaBloomSelect').value || null;

  showLoading('Generating Bloom-classified Q&A pairs with BERT context...');
  try {
    const res = await fetch('/api/generate/qa', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_id: state.activeDocId,
        num_questions: count,
        target_difficulty: difficulty,
        target_bloom: bloom
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Q&A generation failed');

    state.generatedQAs = data.qa_pairs || [];
    renderQAPairs(state.generatedQAs);
  } catch (err) {
    alert(err.message);
  } finally {
    hideLoading();
  }
});

function renderQAPairs(pairs) {
  if (!pairs || pairs.length === 0) {
    qaResultsEl.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon-wrap">
          <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" stroke-width="1.8">
            <circle cx="12" cy="12" r="10"/>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </div>
        <h3>No questions generated</h3>
      </div>`;
    return;
  }

  qaResultsEl.innerHTML = pairs.map((qa, idx) => `
    <div class="qa-card">
      <div class="qa-card-header">
        <span class="badge bloom-${(qa.blooms_level || 'understand').toLowerCase()}">${qa.blooms_level || 'Understand'}</span>
        <span class="badge diff-${(qa.difficulty || 'medium').toLowerCase()}">${qa.difficulty || 'Medium'}</span>
        <span class="page-tag">Page ${qa.page_number}</span>
        <button class="icon-btn" style="width:26px; height:26px; margin-left: 6px;" onclick="copyQAPair(${idx})" title="Copy Q&A">
          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
          </svg>
        </button>
      </div>
      <div class="qa-question">Q${idx + 1}: ${escapeHtml(qa.question)}</div>
      <div class="qa-answer"><strong>Reference answer:</strong> ${escapeHtml(qa.answer)}</div>
      ${qa.context_quote ? `<div class="qa-quote">&ldquo;${escapeHtml(qa.context_quote)}&rdquo;</div>` : ''}
      <div class="answer-evaluator">
        <label for="studentAnswer-${idx}">Your answer</label>
        <textarea id="studentAnswer-${idx}" class="student-answer-input" rows="4" placeholder="Write your answer in your own words..."></textarea>
        <div class="answer-evaluator-actions"><label class="marks-input">Max marks <input id="maxMarks-${idx}" type="number" min="0.5" max="100" step="0.5" value="10"></label><button class="primary-btn sm" onclick="evaluateStudentAnswer(${idx})">Submit answer</button></div>
        <div id="evaluation-${idx}" class="evaluation-result" aria-live="polite"></div>
      </div>
    </div>
  `).join('');
}

window.copyQAPair = function(idx) {
  const qa = state.generatedQAs[idx];
  if (!qa) return;
  const text = `Question: ${qa.question}\nAnswer: ${qa.answer}\n(Page ${qa.page_number} | ${qa.blooms_level} | ${qa.difficulty})`;
  navigator.clipboard.writeText(text).then(() => {
    alert('Copied to clipboard!');
  });
};

window.evaluateStudentAnswer = async function(idx) {
  const qa = state.generatedQAs[idx];
  const input = document.getElementById(`studentAnswer-${idx}`);
  const output = document.getElementById(`evaluation-${idx}`);
  const marksInput = document.getElementById(`maxMarks-${idx}`);
  if (!qa || !input || !output || !input.value.trim()) {
    showToast('Write an answer before submitting.', 'error');
    return;
  }
  output.innerHTML = '<div class="evaluation-loading">Evaluating concepts and meaning…</div>';
  try {
    const res = await fetch('/api/evaluate/answer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
      doc_id: state.activeDocId, question: qa.question, reference_answer: qa.answer,
      student_answer: input.value.trim(), page_number: qa.page_number,
      max_marks: Number(marksInput?.value || 10)
    }) });
    const data = await parseResponse(res);
    output.innerHTML = `<div class="evaluation-score"><strong>${escapeHtml(data.marks)} / ${escapeHtml(data.max_marks)}</strong><span>${escapeHtml(data.percentage)}%</span></div>
      <p>${escapeHtml(data.explanation || '')}</p>
      <div class="evaluation-columns"><div><strong>What you got right</strong><ul>${(data.strengths || []).map(item => `<li>${escapeHtml(item)}</li>`).join('') || '<li>—</li>'}</ul></div><div><strong>Missing or incorrect</strong><ul>${(data.missing_concepts || []).concat(data.incorrect_points || []).map(item => `<li>${escapeHtml(item)}</li>`).join('') || '<li>—</li>'}</ul></div></div>
      <p><strong>Improve:</strong> ${escapeHtml(data.improvement || 'Review the reference answer and source evidence.')}</p>
      <details><summary>Show ideal answer</summary><p>${escapeHtml(data.ideal_answer || qa.answer)}</p></details>`;
  } catch (err) {
    output.innerHTML = `<p class="chat-error">${escapeHtml(err.message)}</p>`;
  }
};

// Export Q&A handlers
exportQaJsonBtn.addEventListener('click', () => {
  if (!state.generatedQAs.length) return alert('No Q&A pairs to export.');
  downloadFile('qa_pairs.json', JSON.stringify(state.generatedQAs, null, 2), 'application/json');
});

exportQaCsvBtn.addEventListener('click', () => {
  if (!state.generatedQAs.length) return alert('No Q&A pairs to export.');
  let csv = 'Question,Answer,Difficulty,BloomsLevel,Page\n';
  state.generatedQAs.forEach(qa => {
    const q = `"${qa.question.replace(/"/g, '""')}"`;
    const a = `"${qa.answer.replace(/"/g, '""')}"`;
    csv += `${q},${a},${qa.difficulty},${qa.blooms_level},${qa.page_number}\n`;
  });
  downloadFile('qa_pairs.csv', csv, 'text/csv');
});

exportQaMdBtn.addEventListener('click', () => {
  if (!state.generatedQAs.length) return alert('No Q&A pairs to export.');
  let md = '# Generated Q&A Pairs\n\n';
  state.generatedQAs.forEach((qa, i) => {
    md += `### Q${i+1}: ${qa.question}\n`;
    md += `**Answer:** ${qa.answer}\n\n`;
    md += `*Difficulty:* ${qa.difficulty} | *Bloom's Level:* ${qa.blooms_level} | *Page:* ${qa.page_number}\n\n---\n\n`;
  });
  downloadFile('qa_pairs.md', md, 'text/markdown');
});

function downloadFile(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Custom test-paper builder
function setupTestPaper() {
  const typeInputs = document.querySelectorAll('[data-paper-type]');
  const markInputs = document.querySelectorAll('[data-paper-mark-type]');
  const totalControl = document.getElementById('paperTotalSelect');
  const updateCount = () => {
    const total = [...typeInputs].reduce((sum, input) => sum + Math.max(0, Number(input.value || 0)), 0);
    const requested = totalControl?.value === 'custom' ? 'custom' : Number(totalControl?.value || 10);
    const valid = requested === 'custom' || requested === total;
    if (paperSummary) paperSummary.textContent = `${total} questions · ${requested === 'custom' ? 'custom total' : `target ${requested}`} · ${total > 50 ? 'Maximum is 50' : valid ? 'Ready to generate' : 'Adjust the type counts to match the target'}`;
  };
  typeInputs.forEach(input => input.addEventListener('input', updateCount));
  totalControl?.addEventListener('change', updateCount);
  updateCount();
  generatePaperBtn?.addEventListener('click', async () => {
    if (!state.activeDocId) return showToast('Select or upload a document first.', 'error');
    const question_counts = {};
    const marks = {};
    typeInputs.forEach(input => { const value = Number(input.value || 0); if (value > 0) question_counts[input.dataset.paperType] = value; });
    markInputs.forEach(input => { marks[input.dataset.paperMarkType] = Number(input.value || 1); });
    const total = Object.values(question_counts).reduce((sum, value) => sum + value, 0);
    if (!total || total > 50) return showToast('Choose between 1 and 50 total questions.', 'error');
    const requested = totalControl?.value === 'custom' ? 'custom' : Number(totalControl?.value || 10);
    if (requested !== 'custom' && requested !== total) return showToast(`Question counts must add up to ${requested}.`, 'error');
    renderSkeletons(testPaperResults, 3);
    showLoading('Building a source-grounded test paper…');
    try {
      const res = await fetch('/api/generate/test-paper', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
        doc_id: state.activeDocId,
        title: document.getElementById('paperTitleInput')?.value.trim() || 'Custom Test Paper',
        subject: document.getElementById('paperSubjectInput')?.value.trim() || '',
        topic: document.getElementById('paperTopicInput')?.value.trim() || 'All topics',
        difficulty: document.getElementById('paperDifficultySelect')?.value || 'Mixed',
        duration_minutes: Number(document.getElementById('paperDurationInput')?.value || 60),
        question_counts, marks,
        include_answer_key: true, include_explanations: true, include_rubrics: true
      }) });
      state.testPaper = await parseResponse(res);
      renderTestPaper(state.testPaper);
      showToast('Test paper generated.');
    } catch (err) {
      renderErrorState(testPaperResults, 'Could not generate test paper', err.message);
    } finally { hideLoading(); }
  });
}

function paperCorrectAnswer(question) {
  const answer = String(question.answer || '—');
  if (question.type === 'mcq' && Array.isArray(question.options)) {
    const option = question.options.find(item => new RegExp(`^\\s*${answer}\\s*[).]`, 'i').test(String(item)));
    return option ? `${answer} — ${option}` : answer;
  }
  return answer;
}

function paperEvaluationReport(paper) {
  const questions = paper.questions || [];
  if (!questions.some(question => question.evaluation)) return '';
  const attempted = questions.filter(question => question.evaluation?.status !== 'not_attempted').length;
  const earned = questions.reduce((sum, question) => sum + Number(question.evaluation?.marks || 0), 0);
  return `<section class="paper-evaluation-report" aria-label="Test evaluation report"><div class="paper-report-header"><div><span class="eyebrow">Evaluation report</span><h3>Attempt review</h3><p>${attempted} of ${questions.length} attempted · ${earned.toFixed(2)} / ${Number(paper.total_marks || 0).toFixed(2)} marks</p></div><span class="paper-report-score">${earned.toFixed(2)} / ${Number(paper.total_marks || 0).toFixed(2)}</span></div><div class="paper-report-list">${questions.map((question, index) => { const evaluation = question.evaluation || {}; const notAttempted = evaluation.status === 'not_attempted'; const status = notAttempted ? 'Not attempted' : evaluation.status === 'correct' ? 'Correct' : evaluation.status === 'incorrect' ? 'Incorrect' : 'Graded'; return `<article class="paper-report-item ${notAttempted ? 'not-attempted' : status.toLowerCase()}"><div class="paper-report-item-head"><strong>Q${index + 1}</strong><span class="paper-result-status">${status}</span><span>${escapeHtml(evaluation.marks ?? 0)} / ${escapeHtml(evaluation.max_marks ?? question.marks)} marks</span></div><p><strong>Your answer:</strong> ${escapeHtml(question.student_answer || 'Not attempted')}</p><p><strong>Correct answer:</strong> ${escapeHtml(paperCorrectAnswer(question))}</p><p class="paper-report-feedback">${escapeHtml(evaluation.explanation || (notAttempted ? 'No answer was submitted. 0 marks awarded.' : 'Evaluated against the document and marking criteria.'))}</p></article>`; }).join('')}</div></section>`;
}

function renderTestPaper(paper) {
  if (!testPaperResults) return;
  const warning = paper.warning ? `<div class="paper-warning">${escapeHtml(paper.warning)}</div>` : '';
  const hasReport = (paper.questions || []).some(question => question.evaluation);
  const earned = (paper.questions || []).reduce((total, question) => total + Number(question.evaluation?.marks || 0), 0);
  const score = hasReport ? ` · Score ${earned.toFixed(2)} / ${Number(paper.total_marks || 0).toFixed(2)}` : '';
  testPaperResults.innerHTML = `<div class="paper-header"><div><span class="eyebrow">${escapeHtml(paper.subject || 'Assessment')}</span><h2>${escapeHtml(paper.title)}</h2><p>${paper.total_questions} questions · ${paper.total_marks} marks · ${paper.duration_minutes} minutes${score}</p></div><div class="paper-header-actions"><button class="primary-btn sm" onclick="startPaperAttempt()">${hasReport ? 'Retake test' : 'Start solving test'}</button><button class="ghost-btn sm" onclick="downloadTestPaper()">Download JSON</button></div></div>${warning}${paperEvaluationReport(paper)}<div class="paper-question-list">${(paper.questions || []).map((question, index) => `<article class="paper-question-card"><div class="paper-question-meta"><span>${index + 1}. ${escapeHtml(String(question.type).replace('_', ' '))}</span><span>${escapeHtml(question.marks)} marks · Page ${escapeHtml(question.page_number)}</span></div><h3>${escapeHtml(question.question)}</h3>${paperResponsePreview(question)}<div class="paper-question-actions"><button class="ghost-btn sm" onclick="editPaperQuestion(${index})">Edit</button><button class="ghost-btn sm" onclick="movePaperQuestion(${index},-1)">↑</button><button class="ghost-btn sm" onclick="movePaperQuestion(${index},1)">↓</button><button class="ghost-btn sm" onclick="deletePaperQuestion(${index})">Delete</button></div><details><summary>Show answer key${question.type === 'descriptive' ? ' & rubric' : ''}</summary>${question.student_answer ? `<p><strong>Your answer:</strong> ${escapeHtml(question.student_answer)}</p>` : ''}${question.evaluation ? `<p><strong>Evaluation:</strong> ${escapeHtml(question.evaluation.marks ?? '')} / ${escapeHtml(question.evaluation.max_marks ?? question.marks)} (${escapeHtml(question.evaluation.percentage ?? '')}%) — ${escapeHtml(question.evaluation.explanation || '')}</p>` : ''}<p><strong>Answer:</strong> ${escapeHtml(paperCorrectAnswer(question))}</p>${question.explanation ? `<p><strong>Explanation:</strong> ${escapeHtml(question.explanation)}</p>` : ''}${question.rubric ? `<p><strong>Rubric:</strong> ${escapeHtml(question.rubric)}</p>` : ''}</details></article>`).join('')}</div>`;
}

function paperResponsePreview(question) {
  if (Array.isArray(question.options) && question.options.length) {
    return `<ol class="paper-options">${question.options.map(option => `<li>${escapeHtml(option)}</li>`).join('')}</ol>`;
  }
  if (question.type === 'true_false') return '<div class="paper-objective-options"><span>True</span><span>False</span></div>';
  if (question.type === 'fill_blank') return '<div class="paper-blank-hint">Write the missing word or phrase.</div>';
  return '';
}

window.downloadTestPaper = function() {
  if (state.testPaper) downloadFile('test-paper.json', JSON.stringify(state.testPaper, null, 2), 'application/json');
};
window.editPaperQuestion = function(index) {
  const question = state.testPaper?.questions?.[index];
  if (!question) return;
  const edited = window.prompt('Edit question', question.question);
  if (edited && edited.trim()) { question.question = edited.trim(); renderTestPaper(state.testPaper); }
};
window.deletePaperQuestion = function(index) {
  if (!state.testPaper?.questions?.[index]) return;
  state.testPaper.questions.splice(index, 1);
  state.testPaper.total_questions = state.testPaper.questions.length;
  state.testPaper.total_marks = state.testPaper.questions.reduce((sum, item) => sum + Number(item.marks || 0), 0);
  renderTestPaper(state.testPaper);
};
window.movePaperQuestion = function(index, direction) {
  const questions = state.testPaper?.questions;
  const target = index + direction;
  if (!questions || target < 0 || target >= questions.length) return;
  [questions[index], questions[target]] = [questions[target], questions[index]];
  renderTestPaper(state.testPaper);
};

function paperAttemptInput(question, index) {
  if (question.type === 'mcq') {
    return `<div class="paper-attempt-options">${(question.options || []).map((option, optionIndex) => {
      const letter = (String(option).match(/^\s*([A-D])\)/i)?.[1] || String.fromCharCode(65 + optionIndex)).toUpperCase();
      return `<label><input type="radio" name="paper-answer-${index}" value="${letter}"><span>${escapeHtml(option)}</span></label>`;
    }).join('')}</div>`;
  }
  if (question.type === 'true_false') {
    return `<div class="paper-attempt-options horizontal"><label><input type="radio" name="paper-answer-${index}" value="True"><span>True</span></label><label><input type="radio" name="paper-answer-${index}" value="False"><span>False</span></label></div>`;
  }
  const placeholder = question.type === 'fill_blank' ? 'Type the missing word or phrase…' : 'Write your answer in your own words…';
  return `<textarea class="paper-student-answer" data-paper-answer="${index}" rows="${question.type === 'descriptive' ? 6 : 3}" placeholder="${placeholder}"></textarea>`;
}

function normaliseObjectiveAnswer(value) {
  const text = String(value || '').trim().toUpperCase();
  const letter = text.match(/^([A-D])(?:\)|\.|\s|$)/);
  if (letter) return letter[1];
  if (text === 'TRUE' || text === 'FALSE') return text[0] + text.slice(1).toLowerCase();
  return text;
}

window.startPaperAttempt = function() {
  if (!state.testPaper || !testPaperResults) return;
  const paper = state.testPaper;
  testPaperResults.innerHTML = `<div class="paper-header"><div><span class="eyebrow">Attempt mode</span><h2>${escapeHtml(paper.title)}</h2><p>${paper.total_questions} questions · ${paper.total_marks} marks · ${paper.duration_minutes} minutes</p></div><button class="ghost-btn sm" onclick="renderTestPaper(state.testPaper)">Back to preview</button></div><div class="paper-question-list">${paper.questions.map((question, index) => `<article class="paper-question-card"><div class="paper-question-meta"><span>${index + 1}. ${escapeHtml(String(question.type).replace('_', ' '))}</span><span>${escapeHtml(question.marks)} marks</span></div><h3>${escapeHtml(question.question)}</h3>${paperAttemptInput(question, index)}</article>`).join('')}</div><button class="primary-btn" onclick="submitPaperAttempt()">Submit & grade test</button>`;
  testPaperResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

window.submitPaperAttempt = async function() {
  if (!state.testPaper) return;
  state.testPaper.questions.forEach((question, index) => {
    const textInput = document.querySelector(`[data-paper-answer="${index}"]`);
    const selected = document.querySelector(`input[name="paper-answer-${index}"]:checked`);
    question.student_answer = (selected?.value || textInput?.value || '').trim();
    if (question.type === 'mcq' || question.type === 'true_false') {
      const correct = normaliseObjectiveAnswer(question.answer);
      const isCorrect = normaliseObjectiveAnswer(question.student_answer) === correct;
      if (!question.student_answer) {
        question.evaluation = { status: 'not_attempted', marks: 0, max_marks: question.marks, percentage: 0, explanation: 'No answer was submitted. 0 marks awarded.' };
      } else {
        question.evaluation = { status: isCorrect ? 'correct' : 'incorrect', marks: isCorrect ? question.marks : 0, max_marks: question.marks, percentage: isCorrect ? 100 : 0, explanation: isCorrect ? 'Correct answer.' : `Incorrect. The correct answer is ${question.answer}.` };
      }
    }
  });
  const writtenQuestions = state.testPaper.questions.filter(question => !['mcq', 'true_false'].includes(question.type));
  writtenQuestions.filter(question => !question.student_answer).forEach(question => {
    question.evaluation = { status: 'not_attempted', marks: 0, max_marks: question.marks, percentage: 0, explanation: 'No answer was submitted. 0 marks awarded.' };
  });
  const written = writtenQuestions.filter(question => question.student_answer);
  if (written.length && state.activeDocId) {
    showLoading(`Evaluating ${written.length} written response${written.length === 1 ? '' : 's'}…`);
    for (const question of written) {
      try {
        const res = await fetch('/api/evaluate/answer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ doc_id: state.activeDocId, question: question.question, reference_answer: question.answer, student_answer: question.student_answer, page_number: question.page_number, max_marks: question.marks, rubric: question.rubric }) });
        question.evaluation = await parseResponse(res);
        question.evaluation.status = 'graded';
      } catch (error) { question.evaluation = { status: 'graded', marks: 0, max_marks: question.marks, percentage: 0, explanation: `Evaluation failed: ${error.message}` }; }
    }
    hideLoading();
  }
  renderTestPaper(state.testPaper);
  showToast('Attempt saved. Open the answer key to review your responses.');
};

// MCQ Quiz System
generateMcqBtn.addEventListener('click', async () => {
  if (!state.activeDocId) return alert('Please upload or select a document first.');

  const count = parseInt(document.getElementById('mcqCountSelect').value, 10);
  const difficulty = document.getElementById('mcqDifficultySelect').value || null;

  showLoading('Generating interactive quiz questions...');
  try {
    const res = await fetch('/api/generate/mcq', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_id: state.activeDocId,
        num_questions: count,
        target_difficulty: difficulty
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Quiz generation failed');

    renderMCQs(data.mcqs || []);
  } catch (err) {
    alert(err.message);
  } finally {
    hideLoading();
  }
});

function renderMCQs(mcqs) {
  state.quizScore = { correct: 0, total: mcqs.length };
  quizScoreDisplay.style.display = 'flex';
  quizScoreVal.textContent = `0/${mcqs.length}`;

  if (!mcqs || mcqs.length === 0) {
    mcqResultsEl.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon-wrap">
          <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" stroke-width="1.8">
            <path d="m9 11 3 3L22 4"/>
            <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
          </svg>
        </div>
        <h3>No questions generated</h3>
      </div>`;
    return;
  }

  mcqResultsEl.innerHTML = mcqs.map((mcq, qIdx) => `
    <div class="mcq-card" id="mcq-card-${qIdx}">
      <div class="qa-card-header">
        <span class="badge diff-${(mcq.difficulty || 'medium').toLowerCase()}">${mcq.difficulty || 'Medium'}</span>
        <span class="page-tag">Page ${mcq.page_number}</span>
      </div>
      <div class="mcq-question">Q${qIdx + 1}: ${mcq.question}</div>
      <div class="mcq-options">
        ${mcq.options.map((opt, oIdx) => {
          const letter = String.fromCharCode(65 + oIdx);
          return `<button class="mcq-option-btn" onclick="handleOptionSelect(${qIdx}, '${letter}', '${mcq.correct_answer}')">${opt}</button>`;
        }).join('')}
      </div>
      <div class="mcq-explanation" id="mcq-expl-${qIdx}">
        <strong>Explanation:</strong> ${mcq.explanation}
      </div>
    </div>
  `).join('');
}

window.handleOptionSelect = function(qIdx, selectedLetter, correctLetter) {
  const card = document.getElementById(`mcq-card-${qIdx}`);
  const buttons = card.querySelectorAll('.mcq-option-btn');
  const expl = document.getElementById(`mcq-expl-${qIdx}`);

  buttons.forEach(btn => {
    btn.disabled = true;
    if (btn.textContent.trim().startsWith(correctLetter)) {
      btn.classList.add('correct');
    }
  });

  if (selectedLetter === correctLetter) {
    state.quizScore.correct += 1;
  } else {
    buttons.forEach(btn => {
      if (btn.textContent.trim().startsWith(selectedLetter)) {
        btn.classList.add('wrong');
      }
    });
  }

  expl.style.display = 'block';
  quizScoreVal.textContent = `${state.quizScore.correct}/${state.quizScore.total}`;
};

// Flashcards Review
generateFlashcardsBtn.addEventListener('click', async () => {
  if (!state.activeDocId) return alert('Please upload or select a document first.');

  const count = parseInt(document.getElementById('flashcardCountSelect').value, 10);

  showLoading('Creating active-recall flashcards...');
  try {
    const res = await fetch('/api/generate/flashcards', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_id: state.activeDocId,
        num_cards: count
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Flashcard generation failed');

    state.currentFlashcards = data.flashcards || [];
    state.flashcardIndex = 0;
    renderFlashcard();
  } catch (err) {
    alert(err.message);
  } finally {
    hideLoading();
  }
});

function renderFlashcard() {
  if (!state.currentFlashcards.length) return;
  const card = state.currentFlashcards[state.flashcardIndex];

  flashcardsWrapperEl.innerHTML = `
    <div class="flashcards-container">
      <div class="flashcard-stage" onclick="this.querySelector('.flashcard-inner').classList.toggle('flipped')">
        <div class="flashcard-inner">
          <div class="flashcard-front">
            <span class="badge bloom-understand" style="margin-bottom: 14px;">${card.concept}</span>
            <h3>${card.front}</h3>
            <p class="file-hint" style="margin-top: 24px;">Click or press Space to flip &bull; &rarr; to advance</p>
          </div>
          <div class="flashcard-back">
            <span class="page-tag" style="margin-bottom: 14px;">Page ${card.page_number}</span>
            <p>${card.back}</p>
          </div>
        </div>
      </div>
      <div class="flashcard-nav">
        <button class="ghost-btn" onclick="prevFlashcard()" ${state.flashcardIndex === 0 ? 'disabled' : ''}>&larr; Previous</button>
        <span style="font-size: 0.84rem; font-weight: 600; color: var(--text-muted);">Card ${state.flashcardIndex + 1} of ${state.currentFlashcards.length}</span>
        <button class="ghost-btn" onclick="nextFlashcard()" ${state.flashcardIndex === state.currentFlashcards.length - 1 ? 'disabled' : ''}>Next &rarr;</button>
      </div>
    </div>
  `;
}

window.prevFlashcard = function() {
  if (state.flashcardIndex > 0) {
    state.flashcardIndex--;
    renderFlashcard();
  }
};

window.nextFlashcard = function() {
  if (state.flashcardIndex < state.currentFlashcards.length - 1) {
    state.flashcardIndex++;
    renderFlashcard();
  }
};

function setupKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    if (state.activeTab === 'flashcards' && state.currentFlashcards.length > 0) {
      if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault();
        const stage = document.querySelector('.flashcard-stage');
        if (stage) stage.querySelector('.flashcard-inner').classList.toggle('flipped');
      } else if (e.code === 'ArrowRight' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        nextFlashcard();
      } else if (e.code === 'ArrowLeft' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        prevFlashcard();
      }
    }
  });
}

// Summary & Study Guide
generateSummaryBtn.addEventListener('click', async () => {
  if (!state.activeDocId) return alert('Please upload or select a document first.');

  showLoading('Synthesizing executive summary and study guide...');
  try {
    const res = await fetch('/api/generate/summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_id: state.activeDocId })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Summary generation failed');

    renderSummary(data.summary);
  } catch (err) {
    alert(err.message);
  } finally {
    hideLoading();
  }
});

function renderSummary(summary) {
  if (!summary) return;

  summaryResultsEl.innerHTML = `
    <div class="summary-container">
      <div class="summary-section">
        <h4>Executive Summary</h4>
        <p>${formatMarkdown(summary.executive_summary)}</p>
      </div>

      <div class="summary-section">
        <h4>Key Takeaways</h4>
        <ul class="summary-list">
          ${(summary.key_takeaways || []).map(item => `<li>${item}</li>`).join('')}
        </ul>
      </div>

      <div class="summary-section">
        <h4>Core Concepts & Terminology</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 10px;">
          ${(summary.core_concepts || []).map(c => `
            <div style="background: var(--bg-input); padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
              <strong style="color: var(--primary); display:block; margin-bottom: 2px;">${c.term}</strong>
              <span style="font-size: 0.82rem; color: var(--text-secondary);">${c.description}</span>
            </div>
          `).join('')}
        </div>
      </div>

      <div class="summary-section">
        <h4>Recommended Study Plan</h4>
        <ul class="summary-list">
          ${(summary.recommended_study_plan || []).map(step => `<li>${step}</li>`).join('')}
        </ul>
      </div>
    </div>
  `;
}

// Settings Modal & API Keys
function setupSettingsModal() {
  openSettingsBtn.addEventListener('click', () => {
    settingsModal.classList.add('active');
  });

  closeSettingsBtn.addEventListener('click', () => {
    settingsModal.classList.remove('active');
  });

  settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) settingsModal.classList.remove('active');
  });

  saveSettingsBtn.addEventListener('click', async () => {
    const payload = {
      llm_provider: modalLlmProvider.value,
      llm_api_key: getSelectedKey(),
      embedding_provider: modalEmbProvider.value,
      salience_method: modalSalienceMethod ? modalSalienceMethod.value : undefined
    };

    showLoading('Updating AI settings...');
    try {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to update settings');

      settingsModal.classList.remove('active');
      await refreshConfig();
      alert('Settings updated successfully!');
    } catch (err) {
      alert(err.message);
    } finally {
      hideLoading();
    }
  });

  // Sync selectors
  if (sidebarSalienceMethod && modalSalienceMethod) {
    sidebarSalienceMethod.addEventListener('change', () => {
      modalSalienceMethod.value = sidebarSalienceMethod.value;
    });
    modalSalienceMethod.addEventListener('change', () => {
      sidebarSalienceMethod.value = modalSalienceMethod.value;
    });
  }

  if (sidebarEmbProvider && modalEmbProvider) {
    sidebarEmbProvider.addEventListener('change', async () => {
      modalEmbProvider.value = sidebarEmbProvider.value;
      try {
        await fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ embedding_provider: sidebarEmbProvider.value })
        });
      } catch (e) {
        console.error('Failed to update embedding provider:', e);
      }
    });
    modalEmbProvider.addEventListener('change', () => {
      sidebarEmbProvider.value = modalEmbProvider.value;
    });
  }
}

function getSelectedKey() {
  const provider = modalLlmProvider.value;
  if (provider === 'gemini') return modalGeminiKey.value.trim() || undefined;
  if (provider === 'groq') return modalGroqKey.value.trim() || undefined;
  if (provider === 'openai') return modalOpenAiKey.value.trim() || undefined;
  return undefined;
}

async function refreshConfig() {
  try {
    const res = await fetch('/api/config');
    const cfg = await res.json();
    state.currentProvider = cfg.llm_provider;

    let display = cfg.llm_provider.toUpperCase();
    if (cfg.llm_provider === 'gemini') display = 'Gemini (Free)';
    else if (cfg.llm_provider === 'groq') display = 'Groq (Free)';
    else if (cfg.llm_provider === 'mock') display = 'Offline Simulator';

    currentProviderNameEl.textContent = display;
    modalLlmProvider.value = cfg.llm_provider;

    if (cfg.embedding_provider) {
      if (modalEmbProvider) modalEmbProvider.value = cfg.embedding_provider;
      if (sidebarEmbProvider) sidebarEmbProvider.value = cfg.embedding_provider;
    }

    if (cfg.salience_method) {
      if (modalSalienceMethod) modalSalienceMethod.value = cfg.salience_method;
      if (sidebarSalienceMethod) sidebarSalienceMethod.value = cfg.salience_method;
    }
  } catch (err) {
    console.error('Failed to get config:', err);
  }
}
