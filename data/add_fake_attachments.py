from pathlib import Path
import json
import sys

# --- ensure project root is on sys.path so `email_rag` is importable ---
ROOT_DIR = Path(__file__).resolve().parents[1]  
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from email_rag.rag_config import DATA_DIR, CHUNKS_PATH, CHUNK_IDS_PATH

FAKE_ATTACHMENTS_PATH = DATA_DIR / "fake_attachments.jsonl"


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    print(f"Loading chunks from {CHUNKS_PATH} ...")
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    existing_ids = {c["chunk_id"] for c in chunks}

    print(f"Loading fake attachments from {FAKE_ATTACHMENTS_PATH} ...")
    attachments = load_jsonl(FAKE_ATTACHMENTS_PATH)

    new_chunks = []
    for att in attachments:
        thread_id = att["thread_id"]
        message_id = att["message_id"]
        page_no = att.get("page_no", 1)
        filename = att.get("filename")

        # unique id for each attachment page
        chunk_id = att.get("chunk_id") or f"att_{message_id}_p{page_no}"
        if chunk_id in existing_ids:
            print(f"Skipping duplicate attachment chunk_id={chunk_id}")
            continue

        rec = {
            "chunk_id": chunk_id,
            "thread_id": thread_id,
            "message_id": message_id,
            "page_no": page_no,
            "source": att.get("source", "attachment"),
            "text": att["text"],
        }
        if filename:
            rec["filename"] = filename

        new_chunks.append(rec)
        existing_ids.add(chunk_id)

    if not new_chunks:
        print("No new attachment chunks to add.")
        return

    print(f"Adding {len(new_chunks)} attachment chunks …")
    chunks.extend(new_chunks)

    save_jsonl(CHUNKS_PATH, chunks)
    print(f"Saved updated chunks to {CHUNKS_PATH}")

    # regenerate chunk_ids.json
    all_ids = [c["chunk_id"] for c in chunks]
    with CHUNK_IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_ids, f)
    print(f"Updated chunk_ids.json at {CHUNK_IDS_PATH}")


if __name__ == "__main__":
    main()