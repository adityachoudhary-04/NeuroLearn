const BACKEND_URL = 'http://127.0.0.1:8000';

let currentTab = null;
let currentDocId = null;

const pageTitleEl = document.getElementById('pageTitle');
const pageUrlEl = document.getElementById('pageUrl');
const statusBar = document.getElementById('statusBar');
const ingestPageBtn = document.getElementById('ingestPageBtn');
const quickSummaryBtn = document.getElementById('quickSummaryBtn');
const quickQuizBtn = document.getElementById('quickQuizBtn');
const quickCardsBtn = document.getElementById('quickCardsBtn');
const resultsArea = document.getElementById('resultsArea');
const resultsContent = document.getElementById('resultsContent');

document.addEventListener('DOMContentLoaded', async () => {
  // Query active tab in Chrome
  if (chrome.tabs) {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs.length > 0) {
      currentTab = tabs[0];
      pageTitleEl.textContent = currentTab.title || 'Untitled Page';
      pageUrlEl.textContent = currentTab.url || '';
    }
  } else {
    pageTitleEl.textContent = 'Development Tab';
    pageUrlEl.textContent = 'https://example.com/test';
  }

  setupEventListeners();
});

function setupEventListeners() {
  ingestPageBtn.addEventListener('click', async () => {
    if (!currentTab || !currentTab.url) return;
    setStatus('Ingesting page into vector store...');
    try {
      const res = await fetch(`${BACKEND_URL}/api/documents/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: currentTab.url })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Ingestion failed');

      currentDocId = data.document.doc_id;
      setStatus(`Indexed! ${data.document.total_chunks} chunks ready.`);
      quickSummaryBtn.disabled = false;
      quickQuizBtn.disabled = false;
      quickCardsBtn.disabled = false;
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  });

  quickSummaryBtn.addEventListener('click', async () => {
    if (!currentDocId) return;
    setStatus('Synthesizing summary...');
    try {
      const res = await fetch(`${BACKEND_URL}/api/generate/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: currentDocId })
      });
      const data = await res.json();
      showResults(`<strong>Summary:</strong><br>${data.summary.executive_summary}`);
      setStatus('Summary ready!');
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  });

  quickQuizBtn.addEventListener('click', async () => {
    if (!currentDocId) return;
    setStatus('Generating 2 quick quiz questions...');
    try {
      const res = await fetch(`${BACKEND_URL}/api/generate/mcq`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: currentDocId, num_questions: 2 })
      });
      const data = await res.json();
      let html = '<strong>Quick Quiz:</strong><br>';
      (data.mcqs || []).forEach((m, i) => {
        html += `<p style="margin-top:6px;"><strong>Q${i+1}:</strong> ${m.question}<br><em>Answer: ${m.correct_answer}</em></p>`;
      });
      showResults(html);
      setStatus('Quiz generated!');
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  });

  quickCardsBtn.addEventListener('click', async () => {
    if (!currentDocId) return;
    setStatus('Creating flashcards...');
    try {
      const res = await fetch(`${BACKEND_URL}/api/generate/flashcards`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: currentDocId, num_cards: 3 })
      });
      const data = await res.json();
      let html = '<strong>Flashcards:</strong><br>';
      (data.flashcards || []).forEach(f => {
        html += `<p style="margin-top:6px;"><strong>[${f.concept}]</strong> ${f.front}<br>&rarr; ${f.back}</p>`;
      });
      showResults(html);
      setStatus('Flashcards ready!');
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  });
}

function setStatus(text) {
  statusBar.textContent = text;
}

function showResults(html) {
  resultsArea.style.display = 'block';
  resultsContent.innerHTML = html;
}
