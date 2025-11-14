import json
import time
import uuid
import numpy as np
import re
from datetime import datetime

from .rag_config import RUNS_DIR, ROOT_DIR
from .rag_data import chunks, bm25, embeddings, sem_model, THREAD_OPTIONS
from .rag_sessions import get_session

RUNS_DIR.mkdir(exist_ok=True)

# --- simple regex patterns for entities ---
FILE_PAT = re.compile(r"\b[\w\-.]+\.(?:pdf|docx?|xls[xm]?|pptx?|txt)\b", re.IGNORECASE)
EMAIL_PAT = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
AMOUNT_PAT = re.compile(r"\b(?:\$|USD\s*)?\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")
DATE_PAT = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")  # very simple date pattern


def rewrite_query(user_text: str, session: dict) -> str:
    """
    Rewrite user query by injecting thread ID and a light summary
    of known entities from entity_memory.
    """
    tid = session["thread_id"]
    mem = session.get("entity_memory") or {}

    key_bits = []

    people = mem.get("people") or []
    if people:
        key_bits.append(f"people: {', '.join(people[:3])}")

    files = mem.get("files") or []
    if files:
        key_bits.append(f"files: {', '.join(files[:3])}")

    amounts = mem.get("amounts") or []
    if amounts:
        key_bits.append(f"amounts: {', '.join(amounts[:3])}")

    dates = mem.get("dates") or []
    if dates:
        key_bits.append(f"dates: {', '.join(dates[:3])}")

    context_str = ""
    if key_bits:
        context_str = "Known entities in this thread: " + "; ".join(key_bits) + ". "

    return f"In thread {tid}, {context_str}answer this question: {user_text}"


def retrieve_chunks(rewrite: str, session: dict, search_outside_thread: bool):
    """
    Hybrid retrieval: BM25 + semantic similarity over precomputed embeddings.
    """
    tokens = rewrite.split()
    bm25_scores = np.array(bm25.get_scores(tokens))  # (N,)

    # Semantic query vector
    q_vec = sem_model.encode([rewrite], normalize_embeddings=True)[0]  # (D,)
    sem_scores = embeddings @ q_vec  # cosine similarity

    # Normalize to [0,1]
    bm25_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores
    sem_norm = (sem_scores + 1.0) / 2.0

    thread_id = session["thread_id"]
    N = len(chunks)
    indices = np.arange(N)

    # Thread filter unless overridden
    if not search_outside_thread:
        mask = np.array([chunks[i]["thread_id"] == thread_id for i in range(N)])
        indices = indices[mask]
        bm25_norm = bm25_norm[mask]
        sem_norm = sem_norm[mask]

    combined = 0.6 * bm25_norm + 0.4 * sem_norm
    order = np.argsort(-combined)

    top_k = 8
    top_indices = indices[order[:top_k]]

    retrieved = []
    for local_rank, idx in enumerate(top_indices):
        c = chunks[idx]
        retrieved.append({
            "chunk_id": c["chunk_id"],
            "thread_id": c["thread_id"],
            "message_id": c["message_id"],
            "page_no": c.get("page_no"),
            "source": c.get("source", "email"),
            "score_bm25": float(bm25_norm[order][local_rank]),
            "score_sem": float(sem_norm[order][local_rank]),
            "score_combined": float(combined[order][local_rank]),
            "text": c["text"],
            # carry over from/to so entity extraction can see people
            "from_addr": c.get("from"),
            "to_addr": c.get("to"),
            "date": c.get("date"),
        })
    return retrieved


def build_answer(user_text: str, rewrite: str, retrieved):
    """
    Answer builder with:
    - 'no clear answer' heuristic
    - special handling for simple 'when' questions using email dates
    - snippet list with citations for grounding
    """
    if not retrieved:
        return (
            "I couldn’t find any emails or content in this thread that clearly answer your question.",
            []
        )

    # ---- Heuristic: check scores + keyword overlap ----
    question_tokens = {t.lower() for t in user_text.split() if len(t) > 3}

    def snippet_has_overlap(snippet: str) -> bool:
        words = {w.lower().strip(".,!?;:()[]") for w in snippet.split()}
        return len(question_tokens & words) > 0

    best_score = max(r["score_combined"] for r in retrieved)
    any_overlap = any(snippet_has_overlap(r["text"]) for r in retrieved)

    if best_score < 0.2 or not any_overlap:
        # Fallback: nothing strongly relevant in this thread
        return (
            "Within this thread, I don’t see any email that clearly answers this question. "
            "You may need to search outside this thread or check other conversations.",
            []
        )

    # ---- Optional: direct answer for 'when' questions ----
    direct_answer_line = None
    if "when" in user_text.lower():
        dated = []
        for r in retrieved:
            date_str = r.get("date")
            if not date_str:
                continue
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                dated.append((dt, r))
            except Exception:
                continue

        if dated:
            # pick the latest email as the likely final approval/confirmation
            dt_best, r_best = max(dated, key=lambda x: x[0])
            nice_date = dt_best.strftime("%Y-%m-%d %H:%M")
            direct_answer_line = (
                f"**Answer:** The most relevant approval email in this thread "
                f"was sent on **{nice_date}** "
                f"[msg: {r_best['message_id']}]."
            )

    # ---- Build snippet-based explanation ----
    lines = []
    if direct_answer_line:
        lines.append(direct_answer_line)
        lines.append("")

    lines.append(f"**Question:** {user_text}")
    lines.append("")
    lines.append("**Relevant information:**")

    citations = []
    seen = set()  # avoid exact duplicate snippet+msg combos

    for r in retrieved:
        msg_id = r["message_id"]
        page_no = r.get("page_no")
        snippet = r["text"].replace("\n", " ")
        snippet = (snippet[:300] + "…") if len(snippet) > 300 else snippet

        key = (msg_id, snippet)
        if key in seen:
            continue
        seen.add(key)

        if page_no is not None:
            cite = f"[msg: {msg_id}, page: {page_no}]"
        else:
            cite = f"[msg: {msg_id}]"

        lines.append(f"- {snippet} {cite}")

        citations.append({
            "message_id": msg_id,
            "page_no": page_no,
            "chunk_id": r["chunk_id"],
        })

    answer = "\n".join(lines)
    return answer, citations


def extract_entities_for_turn(user_text: str, retrieved):
    """
    Extract simple entities from this turn:
    - people: email addresses from chunks + question
    - files: filenames like something.pdf
    - amounts: numbers / $ amounts
    - dates: simple date patterns
    """
    texts = [user_text] + [r["text"] for r in retrieved]

    people = set()
    files = set()
    amounts = set()
    dates = set()

    # from/to emails are good 'people' proxies
    for r in retrieved:
        for field in ("from_addr", "to_addr"):
            val = r.get(field)
            if not val:
                continue
            for email_match in EMAIL_PAT.findall(val):
                people.add(email_match)

    # scan all texts
    for t in texts:
        for m in EMAIL_PAT.findall(t):
            people.add(m)
        for m in FILE_PAT.findall(t):
            files.add(m)
        for m in AMOUNT_PAT.findall(t):
            amounts.add(m)
        for m in DATE_PAT.findall(t):
            dates.add(m)

    entities = {
        "people": sorted(people),
        "amounts": sorted(amounts),
        "files": sorted(files),
        "dates": sorted(dates),
    }
    # Strip empty categories
    entities = {k: v for k, v in entities.items() if v}
    return entities


def log_trace(session_id: str, user_text: str, rewrite: str, retrieved, answer, citations):
    trace_path = RUNS_DIR / "trace.jsonl"

    session = get_session(session_id)
    thread_id = session["thread_id"] if session else None

    record = {
        "trace_id": str(uuid.uuid4()),
        "session_id": session_id,
        "thread_id": thread_id,
        "user_text": user_text,
        "rewrite": rewrite,
        "retrieved": [
            {
                "chunk_id": r["chunk_id"],
                "thread_id": r["thread_id"],
                "message_id": r["message_id"],
                "page_no": r["page_no"],
                "score_bm25": r["score_bm25"],
                "score_sem": r["score_sem"],
                "score_combined": r["score_combined"],
            } for r in retrieved
        ],
        "answer": answer,
        "citations": citations,
        "timestamp": time.time(),
    }

    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record["trace_id"]