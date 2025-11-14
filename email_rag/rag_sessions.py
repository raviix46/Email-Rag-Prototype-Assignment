# rag_sessions.py
import uuid

SESSIONS = {}  # session_id -> {thread_id, recent_turns, entity_memory}


def _init_entity_memory():
    """Create a fresh entity memory structure."""
    return {
        "people": [],
        "amounts": [],
        "files": [],
        "dates": [],
    }


def start_session(thread_id: str) -> str:
    """Create a new session fixed to a given thread."""
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "thread_id": thread_id,
        "recent_turns": [],
        "entity_memory": _init_entity_memory(),
    }
    return sid


def get_session(session_id: str):
    return SESSIONS.get(session_id)


def reset_session(session_id: str):
    """Reset memory but keep the same thread."""
    if session_id in SESSIONS:
        tid = SESSIONS[session_id]["thread_id"]
        SESSIONS[session_id] = {
            "thread_id": tid,
            "recent_turns": [],
            "entity_memory": _init_entity_memory(),
        }


def update_entity_memory(session_id: str, new_entities: dict):
    """
    Merge newly extracted entities into the session's entity_memory.

    new_entities format:
    {
        "people": [...],
        "amounts": [...],
        "files": [...],
        "dates": [...]
    }
    """
    session = get_session(session_id)
    if session is None:
        return

    mem = session.get("entity_memory")
    if not mem:
        mem = _init_entity_memory()
        session["entity_memory"] = mem

    for key, values in new_entities.items():
        if key not in mem:
            mem[key] = []
        # Append only unique values, preserve insertion order
        for v in values:
            if v not in mem[key]:
                mem[key].append(v)