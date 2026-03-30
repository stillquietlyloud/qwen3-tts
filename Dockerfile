# ─────────────────────────────────────────────────────────────────────────────
# Qwen3-TTS API Server – Dockerfile
#
# Build:  docker build -t qwen3-tts-server .
# Run:    docker run --gpus all -p 8000:8000 qwen3-tts-server
# ─────────────────────────────────────────────────────────────────────────────

# CUDA 12.1 + cuDNN 8 on Ubuntu 22.04 (best compatibility with PyTorch 2.3)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ── System packages ───────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        git \
        ffmpeg \
        libsndfile1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# 1. PyTorch (CUDA 12.1 wheel)
RUN pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. FlashAttention 2 (compile from source against the CUDA headers above)
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation

# 3. Application requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY server/ /app/server/

# ── Runtime environment ───────────────────────────────────────────────────────
# Model will be downloaded on first startup (or mount a local cache dir)
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=8000
ENV TTS_DEVICE=cuda:0
ENV TTS_DTYPE=bfloat16
ENV TTS_FLASH_ATTN=1
ENV TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
# Mount /model_cache to persist HuggingFace downloads across container restarts
ENV HF_HOME=/model_cache
ENV TRANSFORMERS_CACHE=/model_cache

EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "server.main"]
