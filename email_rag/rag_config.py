from pathlib import Path
import json

# Project root = parent of the email_rag package
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data and runs live at the project root: /app/data, /app/runs
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = ROOT_DIR / "runs"

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
THREADS_PATH = DATA_DIR / "threads.json"
MESSAGES_PATH = DATA_DIR / "messages.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CHUNK_IDS_PATH = DATA_DIR / "chunk_ids.json"

RUNS_DIR = ROOT_DIR / "runs"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items