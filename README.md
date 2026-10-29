# NeuroLearn: PDF RAG & Q/A Learning Assistant

> A complete, production-ready RAG (Retrieval-Augmented Generation) system that ingests PDFs or Webpages and generates pedagogical Q&A pairs, interactive MCQ quizzes, flashcards, summaries, and real-time chat with page citations.

---

## Architecture Overview

```text
       PDF / Webpage / Text
                ↓
    [1. Document Extraction]     (PyMuPDF fitz / fallback pypdf, HTML cleaner)
                ↓
    [2. Preprocessing & Chunking] (Recursive character split, overlap, page provenance)
                ↓
    [3. Concept & Salience Extr.] (TF-IDF & TextRank importance scoring)
                ↓
    [4. Dense Embeddings]        (Local BERT/Sentence-BERT by default / Gemini / OpenAI / Local fallback)
                ↓
    [5. Vector Database]         (ChromaDB persistent collection / In-memory cosine k-NN)
                ↓
    [6. Hybrid RAG Retrieval]    (Top-k semantic search + metadata filtering)
                ↓
    [7. Multi-Provider LLM]      (Gemini 2.5 Flash [Free] | Groq Llama 3 [Free] | OpenAI | Claude)
                ↓
 ┌─────────────────────────────────────────────────────────┐
 │ • Pedagogical Q&A (Bloom's Taxonomy & Difficulty Rated) │
 │ • Interactive MCQs & Quizzes (Auto-grading & Explains)   │
 │ • 3D Active-Recall Flashcards                           │
 │ • Executive Summary & Study Plan                        │
 │ • Conversational RAG Chat (Page Citations)              │
 └─────────────────────────────────────────────────────────┘
```

---

## Key Features

1. **Free Tier API Integration First**:
   - **Google Gemini API**: Free tier (`gemini-2.5-flash`, `text-embedding-004`) via Google AI Studio.
   - **Groq Cloud API**: Free tier with ultra-fast inference (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`).
   - **Offline / Local Fallback**: Deterministic pseudo-dense projection + mock generator for instant offline testing without any API key required. If the local Sentence-BERT model is unavailable, the embedding wrapper falls back to this mode.

2. **Easy Pluggability for Paid Models**:
   - Seamlessly switch to **OpenAI** (`gpt-4o-mini`, `text-embedding-3-small`) or **Anthropic Claude** (`claude-3-5-sonnet`) with a single click in the UI settings or `.env`.

3. **Pedagogical Q&A Generation**:
   - Classifies questions according to **Bloom's Taxonomy**: *Remember, Understand, Apply, Analyze, Evaluate, Create*.
   - Predicts and tags difficulty (*Easy, Medium, Hard*).
   - Jaccard and semantic deduplication to eliminate redundant questions.
   - Grounded context quote and exact page number citations.
   - Export to **JSON**, **CSV**, or **Markdown**.

4. **Interactive Quiz / MCQ Mode**:
   - 4-choice questions with 1 verified correct answer and plausible distractors.
   - Instant visual feedback (green/red) and educational explanations upon selection.
   - Live score tracker.

5. **Active-Recall 3D Flashcards**:
   - Concept/term on front, definition on back with smooth 3D flip animation.

6. **Web Dashboard & Chrome Extension**:
   - Modern, responsive dark-mode Web Dashboard.
   - Full **Chrome Extension (Manifest V3)** to ingest web pages and trigger instant quizzes directly from any tab.

---

## Directory Structure

```text
minor proj/
├── app/
│   ├── api/
│   │   ├── router.py          # FastAPI REST endpoints
│   │   └── schemas.py         # Pydantic request/response schemas
│   ├── static/
│   │   ├── app.js             # Client-side UI logic (Chat, Q&A, Quizzes, Flashcards)
│   │   ├── index.html         # Modern dashboard
│   │   └── style.css          # Responsive design & 3D cards
│   └── main.py                # FastAPI entry point & CORS configuration
├── chrome_extension/
│   ├── background.js          # Service worker (Manifest V3)
│   ├── manifest.json          # Chrome Extension configuration
│   ├── popup.css              # Extension popup styling
│   ├── popup.html             # Extension popup UI
│   └── popup.js               # Webpage scraper & backend API connector
├── core/
│   ├── generators/
│   │   ├── chat_engine.py     # Grounded chat with citations
│   │   ├── flashcard_generator.py
│   │   ├── mcq_generator.py   # 4-choice quiz generator
│   │   ├── qa_generator.py    # Bloom's taxonomy Q&A generator
│   │   └── summary_generator.py
│   ├── config.py              # Settings & environment variables
│   ├── embedding_provider.py  # Gemini, OpenAI, & Local embedding abstractions
│   ├── extractor.py           # PyMuPDF / pypdf & URL HTML scraper
│   ├── llm_provider.py        # Gemini, Groq, OpenAI, Claude & Mock providers
│   ├── preprocessor.py        # Recursive chunking & TF-IDF salience scoring
│   ├── rag_pipeline.py        # Master RAG pipeline coordinator
│   └── vector_store.py        # ChromaDB persistent store + fallback
├── data/
│   ├── chroma_db/             # Vector database storage
│   └── uploads/               # Stored PDF documents
├── tests/
│   ├── test_api_endpoints.py  # FastAPI TestClient test suite
│   └── test_rag_pipeline.py   # End-to-end pipeline verification test
├── .env.example               # Environment variables template
├── requirements.txt           # Project dependencies
└── run.py                     # One-click start script
```

---

## Quick Start

### 1. Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

*(Or if using cmd: `venv\Scripts\activate.bat`)*

### 2. Configure Free API Keys

Open `.env` (or configure via the Web UI Settings tab):

```env
# Free Google AI Studio API Key: https://aistudio.google.com/
GEMINI_API_KEY=your_gemini_key_here

# Free Groq Cloud API Key: https://console.groq.com/
GROQ_API_KEY=your_groq_key_here
```

> **Note**: Even without an API key, the system runs in **Mock/Offline mode**, allowing you to test the entire ingestion, vector indexing, retrieval, and UI flow immediately!

### 3. Launch the Application

```powershell
python run.py
```

Open your browser at:
- **Web Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Using the Chrome Extension (Manifest V3)

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Toggle on **Developer mode** in the top-right corner.
3. Click **Load unpacked**.
4. Select the `chrome_extension` folder in this project:
   `C:\Users\harshit\Desktop\minor proj\chrome_extension`
5. Pin the NeuroLearn icon to your Chrome toolbar.
6. Open any article or PDF in Chrome, click the extension, and click **Ingest Current Page** to generate instant quizzes and flashcards!

---

## How to Switch to OpenAI / Claude

1. In `.env` or in the **Web UI Settings dialog** (gear icon in the bottom left):
   - Set `OPENAI_API_KEY=sk-...` or `ANTHROPIC_API_KEY=sk-ant-...`
   - Select **OpenAI** or **Claude** from the active provider dropdown.
   - For embeddings, select **OpenAI text-embedding-3-small**.
2. Click **Save & Apply Changes**. The pipeline hot-swaps to the new provider without restarting the server!

---

## Running Automated Tests

Run the full end-to-end integration and API tests:

```powershell
python tests/test_rag_pipeline.py
python tests/test_api_endpoints.py
```
