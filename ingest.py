# ingest.py
from pathlib import Path
import subprocess

from rag_config import ROOT_DIR

def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    data_dir = ROOT_DIR / "data"

    # 1) Build a small email subset (if not already present)
    subset_path = data_dir / "emails_subset_with_ids.csv"
    if not subset_path.exists():
        run(["python", "data/preprocess_subset.py"])
    else:
        print("emails_subset_with_ids.csv already exists, skipping preprocess_subset.")

    # 2) Build chunks + threads/messages indices
    chunks_path = data_dir / "chunks.jsonl"
    threads_path = data_dir / "threads.json"
    messages_path = data_dir / "messages.json"
    if not (chunks_path.exists() and threads_path.exists() and messages_path.exists()):
        run(["python", "data/build_indices.py"])
    else:
        print("Index files already exist, skipping build_indices.")

    # 3) Build embeddings for all chunks
    emb_path = data_dir / "embeddings.npy"
    if not emb_path.exists():
        run(["python", "data/build_embeddings.py"])
    else:
        print("embeddings.npy already exists, skipping build_embeddings.")

    print("Ingest pipeline completed.")

if __name__ == "__main__":
    main()