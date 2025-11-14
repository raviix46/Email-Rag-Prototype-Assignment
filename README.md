# 📧 Email Thread RAG Assistant

This project is a **hybrid RAG (Retrieval-Augmented Generation) system** for exploring a subset of the **Enron email corpus** and a few **PDF attachments**.

The assistant lets you:

- Pick an **email thread**.
- Ask natural-language questions about **that thread**.
- Retrieve evidence from both:
  - **Emails** (message bodies)
  - **Attachments** (PDF pages, AI-generated)
- Get answers with **inline citations** like:

  - From email: `[msg: M-000207]`
  - From attachment: `[msg: M-000207, page: 2]`

The system includes:

- **Hybrid retrieval**: BM25 keywords + sentence-transformer semantic search.
- **Thread-scoped sessions** with simple **entity memory**.
- **Timeline view** of who said what, when.
- A **Gradio UI** and a **FastAPI API**.
- Docker support via `Dockerfile` + `docker-compose.yml`.

---

## 1. Repository Structure

```text
Email-Rag-Prototype-Assignment/
│
├── app.py                 # Gradio UI
├── api.py                 # FastAPI backend (REST API)
├── ingest.py              # (optional) original ingest pipeline entry point
├── Dockerfile             # Container image for the app
├── docker-compose.yml     # Compose file to run the app + API
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── email_rag/
│   ├── __init__.py
│   ├── rag_config.py      # Paths and config for data and runs
│   ├── rag_data.py        # Loads chunks, BM25 index, embeddings, thread list
│   ├── rag_retrieval.py   # Query rewrite, hybrid retrieval, answer builder
│   ├── rag_sessions.py    # Session store, entity memory, timeline helpers
│   └── rag_timeline.py    # Timeline formatting helpers (used by UI)
│
└── data/
    ├── emails_subset_with_ids.csv  # Small Enron subset used for this project
    ├── messages.json               # Normalized email messages
    ├── threads.json                # Threads (group of messages)
    ├── chunks.jsonl                # Retrieval chunks (emails + attachment pages)
    ├── chunk_ids.json              # Ordered chunk ids (aligned with embeddings.npy)
    ├── embeddings.npy              # Dense embeddings for all chunks
    ├── fake_attachments.jsonl      # Text + metadata for AI-generated PDFs
    ├── add_fake_attachments.py     # Script to merge fake attachments into chunks.jsonl
    ├── build_embeddings.py         # Script to rebuild embeddings.npy & chunk_ids.json
    └── attachments/
        ├── NGI_draft_terms.pdf     # Draft PDF (1 page)
        └── NGI_final_approval.pdf  # Final approval PDF (2 pages)
```

---

## 2. High-Level Architecture

### 2.1 Components

1. **Data layer (`data/`)**
   - Holds the preprocessed Enron subset, chunk index, embeddings, and synthetic PDFs.
   - `chunks.jsonl` is the core retrieval index:
     - Each line is one chunk (either an email or one page of a PDF attachment).

2. **Core RAG logic (`email_rag/`)**

   - `rag_data.py`
     - Loads `chunks.jsonl`.
     - Loads `embeddings.npy`.
     - Builds a BM25 index over chunk texts.
     - Exposes `chunks`, `bm25`, `embeddings`, and `THREAD_OPTIONS`.

   - `rag_retrieval.py`
     - `rewrite_query`: injects thread id + entity memory into the query.
     - `retrieve_chunks`: hybrid BM25 + semantic retrieval, optionally thread-scoped.
     - `build_answer`: assembles markdown answer with citations.
     - `extract_entities_for_turn`: regex-based entity extraction (people, files, dates, amounts).
     - `log_trace`: logs every turn into `runs/trace.jsonl`.

   - `rag_sessions.py`
     - In-memory session store: `start_session`, `get_session`, `reset_session`.
     - Manages:
       - `thread_id` for each session.
       - `recent_turns` (short history).
       - `entity_memory` (people, files, amounts, dates).
     - Timeline helpers used by the UI.

3. **Application layer**

   - `app.py` – Gradio-based UI:
     - Dropdown to choose thread.
     - Buttons: **Start Session**, **Ask**, **Reset Session**, **Show Timeline**.
     - Textbox for questions.
     - Checkbox **“Search outside selected thread”**.
     - Debug section showing rewritten query + retrieved chunk scores.

   - `api.py` – FastAPI service:
     - `POST /start_session`
     - `POST /ask`
     - `POST /switch_thread`
     - `POST /reset_session`
     - Uses the same RAG logic as the Gradio UI.

4. **Logging**

   - `runs/trace.jsonl` – JSONL log with:
     - `trace_id`, `session_id`, `thread_id`
     - original question
     - rewritten query
     - retrieved chunk IDs + scores
     - answer + citations

---

## 3. Dataset and Indexing

### 3.1 Source Dataset

- Based on the public Enron email corpus.
- For this assignment, a small subset was created locally:
  - Load first ~5000 rows of the full CSV.
  - Filter to threads with 3–25 emails.
- Resulting subset:
  - ~207 emails
  - grouped into ~15 threads
- This subset is stored as:
  - `data/emails_subset_with_ids.csv`

### 3.2 Normalization

The ingest pipeline (run once, locally) derived:

- `data/messages.json` – one record per email:
```json
  {
    "message_id": "M-000207",
    "thread_id": "T-0002",
    "date": "2001-07-26 11:56:00",
    "from": "phillip.allen@enron.com",
    "to": ["dexter@intelligencepress.com"],
    "cc": ["..."],
    "subject": "NGI access to eol",
    "body_text": "Dexter, Hopefully Griff Gray has sent you the information ..."
  }
```
- `data/threads.json` – one record per thread:
```json
  {
    "thread_id": "T-0002",
    "subject": "NGI access to eol",
    "messages": ["M-000119", "M-000120", "...", "M-000207"],
    "first_date": "...",
    "last_date": "...",
    "participants": [
      "phillip.allen@enron.com",
      "dexter@intelligencepress.com",
      "..."
    ]
  }
```
### 3.3 Chunking

The main index file is `data/chunks.jsonl` (JSONL: one JSON object per line).

Two kinds of chunks:

1. Email chunks (per message)
```json
   {
     "chunk_id": "M-000207",
     "thread_id": "T-0002",
     "message_id": "M-000207",
     "source": "email",
     "text": "Dexter, Hopefully Griff Gray has sent you the information on your id and password by now..."
   }
```
2. Attachment chunks (per PDF page)

   The NGI thread has two synthetic PDFs:

   - `NGI_draft_terms.pdf` – 1 page, linked to `M-000206`
   - `NGI_final_approval.pdf` – 2 pages, linked to `M-000207`

   Text and metadata are defined in `data/fake_attachments.jsonl` and merged into `chunks.jsonl` via `data/add_fake_attachments.py`, resulting in entries like:
```json
   {
     "chunk_id": "att_M-000207_p2",
     "thread_id": "T-0002",
     "message_id": "M-000207",
     "page_no": 2,
     "source": "attachment",
     "filename": "NGI_final_approval.pdf",
     "text": "DETAILS – The guest ID for Dexter Steis is valid through January of the following year..."
   }
```
### 3.4 Embeddings

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- All chunk texts are encoded into dense vectors:
  - Stored as `data/embeddings.npy` (float32).
  - `data/chunk_ids.json` stores the corresponding `chunk_id` order.
 
---

## 4. Retrieval Approach

### 4.1 Lexical Retrieval (BM25)

- Library: `rank-bm25`
- Index built over **chunk texts** (emails + attachment pages).
- For a tokenized query `tokens`, BM25 scores are computed as:
```python
bm25_scores = bm25.get_scores(tokens)  # 1 score per chunk
```

- Scores are normalized to [0, 1] by dividing by the maximum:
 ```python
if bm25_scores.max() > 0:
    bm25_norm = bm25_scores / bm25_scores.max()
else:
    bm25_norm = bm25_scores  # remains all zeros if everything is zero
```

### 4.2 Semantic Retrieval (Embeddings)

- The query is rewritten and encoded using the same sentence-transformer model:
```python
q_vec = sem_model.encode([rewrite], normalize_embeddings=True)[0]  # shape (D,)
sem_scores = embeddings @ q_vec  # cosine similarity (one score per chunk)
sem_norm = (sem_scores + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
```

- Here:
  - embeddings is the (num_chunks, D) matrix loaded from data/embeddings.npy.
  - q_vec is the query embedding.

### 4.3 Hybrid Scoring

- Combined score for each chunk:
```python
combined = 0.6 * bm25_norm + 0.4 * sem_norm
```

- Then:
```python
# sort indices by combined score descending
order = np.argsort(-combined)

top_k = 8  # default number of chunks to return
top_indices = indices[order[:top_k]]

# later used to pick top chunks from `chunks`
top_chunks = [chunks[i] for i in top_indices]
```

- Steps:
  - Sort chunks by combined score in descending order.
  - Take top k chunks (default top_k = 8).
  - Optionally filter by thread (see below).
 
### 4.4 Thread-Scoped vs Global Search

- Each session is bound to a specific thread_id.
- By default, retrieval only considers chunks within that thread:
```python
thread_id = session["thread_id"]
N = len(chunks)
indices = np.arange(N)

if not search_outside_thread:
    mask = np.array([chunks[i]["thread_id"] == thread_id for i in range(N)])
    indices = indices[mask]
    bm25_norm = bm25_norm[mask]
    sem_norm = sem_norm[mask]
```

- If search_outside_thread = True (via UI checkbox or API query parameter),
this filter is skipped, and retrieval covers all chunks across all threads.

---

## 5. Conversation & Memory

### 5.1 Sessions

- Sessions are managed in `email_rag/rag_sessions.py`.
- Each session is always tied to:
  - a specific `thread_id`
  - a small rolling chat history (`recent_turns`)
  - a lightweight `entity_memory` (people, files, dates, amounts)

Conceptually, a session looks like:

```json
{
  "session_id": "a5b94f4a-8e4e-4b7e-9a8d-0c7c9a4f4d01",
  "thread_id": "T-0002",
  "recent_turns": [
    { "user": "...", "answer": "..." }
  ],
  "entity_memory": {
    "people": ["phillip.allen@enron.com", "dexter@intelligencepress.com"],
    "files": ["NGI_final_approval.pdf"],
    "amounts": [],
    "dates": ["01/26/2001"]
  }
}
```
At a high level:
- start_session(thread_id) → creates a new session.
- get_session(session_id) → returns the in-memory record.
- reset_session(session_id) → clears history + entity memory for that session.

These functions are used by both:
	•	the Gradio UI (app.py)
	•	the FastAPI API (api.py)
  
### 5.2 Entity Memory
- Implemented in rag_retrieval.extract_entities_for_turn(user_text, retrieved).
- Uses regex patterns to capture:
  - people  → email-like strings.
  - files   → filenames (*.pdf, *.docx, *.xlsx, *.pptx, *.txt).
  - amounts → simple money/number patterns.
  - dates   → simple d/m/yyyy-style dates.

- Shape of extracted entities:
```json
{
  "people": [
    "dexter@intelligencepress.com",
    "phillip.allen@enron.com"
  ],
  "files": [
    "NGI_draft_terms.pdf",
    "NGI_final_approval.pdf"
  ],
  "amounts": [],
  "dates": [
    "01/26/2001"
  ]
}
```

---

## 6. Answering, Grounding & Citations

- Answering logic lives in `email_rag/rag_retrieval.py` → `build_answer(...)`.
- Flow:
  1. Check retrieved chunks:
     - If nothing relevant → return a **polite “no clear answer”** instead of hallucinating.
  2. Otherwise:
     - Echo the user question.
     - List **relevant snippets** as bullet points.
     - Add **inline citations**:
       - Email body → `[msg: M-000207]`
       - Attachment page → `[msg: M-000207, page: 2]`.

Example answer shape:

```markdown
**Question:** When was that approval sent?

**Relevant information:**
- Dexter, Hopefully Griff Gray has sent you the information on your id and password by now. It should be good through January. [msg: M-000207]
- DETAILS – The guest ID for Dexter Steis is valid through January of the following year. [msg: M-000207, page: 2]
```

Along with the markdown answer, the function returns a structured list of citations like:
```json
[
  { "message_id": "M-000207", "page_no": null, "chunk_id": "M-000207" },
  { "message_id": "M-000207", "page_no": 2,    "chunk_id": "att_M-000207_p2" }
]
```

---

## 7. Timeline View

- Implemented in email_rag/rag_timeline.py.
- Uses threads.json + messages.json to show who said what, when within a thread.

Example output:
```text
Timeline for thread T-0002
2001-01-26 01:55 — phillip.allen@enron.com — NGI access to eol [msg: M-000119]
2001-01-26 01:57 — phillip.allen@enron.com — Re: NGI access to eol [msg: M-000123]
2001-03-26 09:05 — phillip.allen@enron.com — Re: NGI access to eol [msg: M-000146]
2001-07-26 11:56 — phillip.allen@enron.com — RE: NGI access to eol [msg: M-000207]
```
In the Gradio app, a “Show Timeline” button:
- Looks up the current session’s thread_id.
- Calls the timeline helper.
- Displays this text as a quick per-thread summary.

---

## 10. Setup & Running

This section explains how to run the project:

- Locally on your machine (recommended for development).
- Using Docker + docker-compose (for reproducible runs).

> Note: All data artifacts (`chunks.jsonl`, `embeddings.npy`, `fake_attachments.jsonl`, etc.) are already included in the repo, so you **do not** need to re-run the ingest pipeline unless you want to change the data.

### 10.1 Prerequisites

- **Python**: 3.9+ (3.10/3.11 also fine)
- **Git**
- (Optional, but recommended) **virtualenv / venv**
- For Docker mode:
  - **Docker** and **docker-compose** installed

### 10.2 Clone the Repository
```bash
git clone https://github.com/raviix46/Email-Rag-Prototype-Assignment.git
cd Email-Rag-Prototype-Assignment
```

### 10.3 Create and Activate Virtual Environment (Local Mode)

Create a virtual environment so this project’s dependencies don’t clash with global Python packages.
```bash
# Create venv
python -m venv .venv

# Activate (Linux / macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
# .venv\Scripts\Activate.ps1
```

### 10.4 Install Dependencies

Install all required Python packages from requirements.txt:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- gradio – UI
- fastapi, uvicorn, pydantic – API backend
- rank-bm25 – BM25 lexical retrieval
- sentence-transformers – MiniLM embeddings
- numpy and other supporting libraries

### 10.5 Data Artifacts (Already Provided)

The repo already includes:
	•	data/messages.json          – normalized emails
	•	data/threads.json           – per-thread metadata
	•	data/chunks.jsonl           – email + attachment chunks
	•	data/fake_attachments.jsonl – text for synthetic PDFs
	•	data/embeddings.npy         – dense embeddings for all chunks
	•	data/chunk_ids.json         – order of chunk_id for embeddings
	•	data/attachments/           – the PDF files

You only need to re-run the data scripts if you:
	•	Add new attachments.
	•	Change fake_attachments.jsonl.
	•	Want to rebuild the index from scratch.
