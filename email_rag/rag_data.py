import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .rag_config import (
    CHUNKS_PATH,
    THREADS_PATH,
    MESSAGES_PATH,
    EMBEDDINGS_PATH,
    CHUNK_IDS_PATH,
    load_json,
    load_jsonl,
)

# Load base data
chunks = load_jsonl(CHUNKS_PATH)
threads = load_json(THREADS_PATH)
messages = load_json(MESSAGES_PATH)

# Map chunk_id -> chunk
chunk_id_to_chunk = {c["chunk_id"]: c for c in chunks}

# BM25 corpus
corpus_tokens = [c["text"].split() for c in chunks]
bm25 = BM25Okapi(corpus_tokens)

# Semantic embeddings
embeddings = np.load(EMBEDDINGS_PATH)  # (N, D)

with CHUNK_IDS_PATH.open("r", encoding="utf-8") as f:
    chunk_ids = load_json(CHUNK_IDS_PATH)

# Map chunk_id -> index in embeddings
chunk_index = {cid: i for i, cid in enumerate(chunk_ids)}

# SentenceTransformer model (same as used in build_embeddings)
SEM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
sem_model = SentenceTransformer(SEM_MODEL_NAME)

# Thread IDs for dropdown
THREAD_OPTIONS = sorted(list(threads.keys()))