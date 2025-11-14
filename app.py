import gradio as gr

from email_rag.rag_data import THREAD_OPTIONS
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
from email_rag.rag_timeline import build_timeline


def init_session_ui(thread_id: str):
    if not thread_id:
        return None, "Please select a thread to start."
    sid = start_session(thread_id)
    return sid, f"Started session for thread: {thread_id}"


def chat_ui(user_text: str, session_id: str, search_outside_thread: bool):
    if not session_id:
        return "Please start a session by selecting a thread.", "", ""

    session = get_session(session_id)
    if session is None:
        return "Session not found. Please start again.", "", ""

    # 1) Rewrite query using thread + entity memory
    rewrite = rewrite_query(user_text, session)

    # 2) Retrieve chunks
    retrieved = retrieve_chunks(rewrite, session, search_outside_thread)

    # 3) Extract entities from this turn + retrieved evidence, update memory
    new_entities = extract_entities_for_turn(user_text, retrieved)
    if new_entities:
        update_entity_memory(session_id, new_entities)

    # 4) Build grounded answer
    answer, citations = build_answer(user_text, rewrite, retrieved)

    # 5) Update simple turn memory
    session["recent_turns"].append({"user": user_text, "answer": answer})
    if len(session["recent_turns"]) > 5:
        session["recent_turns"] = session["recent_turns"][-5:]

    # 6) Log trace for this turn
    log_trace(session_id, user_text, rewrite, retrieved, answer, citations)

    # 7) Debug: show retrieved chunk ids + scores
    debug_retrieved = "\n".join(
        [
            f"{r['chunk_id']} (msg={r['message_id']}, "
            f"bm25={r['score_bm25']:.3f}, sem={r['score_sem']:.3f}, "
            f"combined={r['score_combined']:.3f})"
            for r in retrieved
        ]
    )

    return answer, rewrite, debug_retrieved


def reset_session_ui(session_id: str):
    if session_id:
        reset_session(session_id)
    return "", "Session reset."


def timeline_ui(session_id: str):
    if not session_id:
        return "Please start a session by selecting a thread."
    session = get_session(session_id)
    if session is None:
        return "Session not found. Please start again."
    tid = session["thread_id"]
    return build_timeline(tid)


with gr.Blocks() as demo:
    gr.Markdown("# 📧 Email Thread RAG Assistant\nAsk questions about a selected Enron email thread.")

    with gr.Row():
        thread_dd = gr.Dropdown(
            choices=THREAD_OPTIONS,
            label="Select Thread ID",
            value=THREAD_OPTIONS[0] if THREAD_OPTIONS else None,
            interactive=True,
        )
        start_btn = gr.Button("Start Session")
        session_state = gr.State(value=None)
        status_box = gr.Markdown("")

    start_btn.click(
        fn=init_session_ui,
        inputs=[thread_dd],
        outputs=[session_state, status_box],
    )

    with gr.Row():
        user_box = gr.Textbox(label="Your question", lines=2)

    with gr.Row():
        search_toggle = gr.Checkbox(label="Search outside selected thread", value=False)
        ask_btn = gr.Button("Ask")
        reset_btn = gr.Button("Reset Session")
        timeline_btn = gr.Button("Show Timeline")

    answer_box = gr.Markdown(label="Answer")
    timeline_box = gr.Markdown(label="Thread timeline")

    with gr.Accordion("Debug info", open=False):
        rewrite_box = gr.Textbox(label="Rewritten query", interactive=False)
        retrieved_box = gr.Textbox(label="Retrieved chunks", interactive=False)

    ask_btn.click(
        fn=chat_ui,
        inputs=[user_box, session_state, search_toggle],
        outputs=[answer_box, rewrite_box, retrieved_box],
    )

    reset_btn.click(
        fn=reset_session_ui,
        inputs=[session_state],
        outputs=[session_state, status_box],
    )

    timeline_btn.click(
        fn=timeline_ui,
        inputs=[session_state],
        outputs=[timeline_box],
    )

if __name__ == "__main__":
    demo.launch()