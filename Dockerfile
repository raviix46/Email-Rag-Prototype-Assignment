# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps if needed later (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

ENV PYTHONUNBUFFERED=1

# Expose API (FastAPI) and UI (Gradio) ports
EXPOSE 8000
EXPOSE 7860

# Default command can be overridden by docker-compose
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]