# api.py
import time
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any

from email_rag.rag_sessions import (
    start_session,
    reset_session,
    get_session,
    update_entity_memory,
)
from email_rag.rag_retrieval import (
    rewrite_query,
    retrieve_chunks,
    build_answer,
    log_trace,
    extract_entities_for_turn,
)

app = FastAPI(title="Email Thread RAG API")


# ---------- Pydantic models ----------

class StartSessionRequest(BaseModel):
    thread_id: str


class StartSessionResponse(BaseModel):
    session_id: str
    thread_id: str


class AskRequest(BaseModel):
    session_id: str
    text: str
    # body flag (optional); also support query flag ?search_outside_thread=true
    search_outside_thread: Optional[bool] = False


class Citation(BaseModel):
    message_id: str
    page_no: Optional[int] = None
    chunk_id: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    thread_id: str
    message_id: str
    page_no: Optional[int] = None
    source: str
    score_bm25: float
    score_sem: float
    score_combined: float


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    rewrite: str
    retrieved: List[RetrievedChunk]
    trace_id: str
    latency_sec: float   # ⬅️ latency included in response


class SwitchThreadRequest(BaseModel):
    thread_id: str


class ResetSessionRequest(BaseModel):
    session_id: str


# ---------- Endpoints ----------

@app.post("/start_session", response_model=StartSessionResponse)
def api_start_session(payload: StartSessionRequest):
    """
    Start a new session bound to a given thread_id.
    """
    session_id = start_session(payload.thread_id)
    return StartSessionResponse(session_id=session_id, thread_id=payload.thread_id)


@app.post("/ask", response_model=AskResponse)
def api_ask(
    payload: AskRequest,
    search_outside_thread: bool = Query(
        False,
        description="Set to true to allow fallback search outside the active thread.",
    ),
):
    """
    Ask a question within an existing session.

    - Uses thread-scoped retrieval by default.
    - Supports global search fallback via ?search_outside_thread=true
      or payload.search_outside_thread = true.
    """
    session = get_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # combine body + query flag (OR)
    search_flag = bool(payload.search_outside_thread or search_outside_thread)

    # ---- measure latency for core RAG pipeline ----
    t0 = time.perf_counter()

    # rewrite using thread + entity memory
    rewrite = rewrite_query(payload.text, session)

    # retrieve chunks
    retrieved = retrieve_chunks(rewrite, session, search_flag)

    # entity memory update
    new_entities = extract_entities_for_turn(payload.text, retrieved)
    if new_entities:
        update_entity_memory(payload.session_id, new_entities)

    # build answer
    answer, citations = build_answer(payload.text, rewrite, retrieved)

    elapsed = time.perf_counter() - t0  # seconds

    # log and get trace_id
    trace_id = log_trace(payload.session_id, payload.text, rewrite, retrieved, answer, citations)

    # format retrieved chunks for response
    retrieved_out = [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            thread_id=r["thread_id"],
            message_id=r["message_id"],
            page_no=r.get("page_no"),
            source=r.get("source", "email"),
            score_bm25=r["score_bm25"],
            score_sem=r["score_sem"],
            score_combined=r["score_combined"],
        )
        for r in retrieved
    ]

    citations_out = [
        Citation(
            message_id=c["message_id"],
            page_no=c.get("page_no"),
            chunk_id=c["chunk_id"],
        )
        for c in citations
    ]

    return AskResponse(
        answer=answer,
        citations=citations_out,
        rewrite=rewrite,
        retrieved=retrieved_out,
        trace_id=trace_id,
        latency_sec=elapsed,
    )


@app.post("/switch_thread", response_model=StartSessionResponse)
def api_switch_thread(payload: SwitchThreadRequest):
    """
    Simplest interpretation: switching thread = start a new session on that thread.

    (Keeps the API contract: { "thread_id": "..." } → session info)
    """
    session_id = start_session(payload.thread_id)
    return StartSessionResponse(session_id=session_id, thread_id=payload.thread_id)


@app.post("/reset_session")
def api_reset_session(payload: ResetSessionRequest):
    """
    Reset an existing session's memory (same behavior as UI reset).
    """
    if get_session(payload.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    reset_session(payload.session_id)
    return {"status": "ok", "session_id": payload.session_id}