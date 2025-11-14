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
