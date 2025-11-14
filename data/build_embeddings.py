# data/build_embeddings.py

import json
from pathlib import Path
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

# --- ensure project root is on sys.path so `email_rag` is importable ---
ROOT_DIR = Path(__file__).resolve().parents[1]  # parent of `data/`
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from email_rag.rag_config import CHUNKS_PATH, CHUNK_IDS_PATH, EMBEDDINGS_PATH


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    print(f"Loading chunks from {CHUNKS_PATH} ...")
    chunks = load_jsonl(CHUNKS_PATH)
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    print(f"Total chunks: {len(chunks)}")

    print("Loading sentence-transformers model ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Encoding chunks ...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype("float32")

    print(f"Saving embeddings to {EMBEDDINGS_PATH} ...")
    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Saving chunk IDs to {CHUNK_IDS_PATH} ...")
    with CHUNK_IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)

    print("Done.")


if __name__ == "__main__":
    main()