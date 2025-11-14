from datetime import datetime
from .rag_data import threads, messages


def build_timeline(thread_id: str) -> str:
    """
    Build a simple markdown timeline for a thread:
    - one line per message
    - sorted by date
    - with [msg: <id>] citations
    """
    msg_ids = threads.get(thread_id, [])
    if not msg_ids:
        return f"No messages found for thread {thread_id}."

    entries = []
    for mid in msg_ids:
        m = messages.get(mid)
        if not m:
            continue
        date_str = m.get("date") or ""
        sender = m.get("from") or "(unknown)"
        subject = m.get("subject") or "(no subject)"

        # Try to format date nicely
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            date_fmt = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_fmt = date_str

        line = f"- **{date_fmt}** — **{sender}** — _{subject}_ [msg: {mid}]"
        entries.append((date_str, line))

    # Sort by raw date string; not perfect but fine for this dataset
    entries.sort(key=lambda x: x[0])
    lines = [f"### Timeline for thread {thread_id}", ""]
    lines.extend(line for _, line in entries)

    return "\n".join(lines)