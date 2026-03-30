#!/usr/bin/env bash
# =============================================================================
# setup.sh – Qwen3-TTS Voice-Design Service setup for Ubuntu 24.04
#
# Target hardware: NVIDIA RTX 4070 (12 GB VRAM), 10-core CPU, 32 GB RAM
#
# Run as root or with sudo:  sudo bash setup.sh
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 1. System update ─────────────────────────────────────────────────────────
info "Updating system packages …"
apt-get update -q
apt-get upgrade -y -q
apt-get install -y -q \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    ca-certificates \
    gnupg \
    software-properties-common

# ── 2. NVIDIA drivers (skip if already installed) ────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    info "Installing NVIDIA driver (open-source kernel module) …"
    # Ubuntu 24.04 has nvidia-driver-555+ in its repos
    apt-get install -y -q nvidia-driver-555 nvidia-utils-555
    info "NVIDIA driver installed.  A reboot is required before continuing."
    info "Re-run this script after reboot."
    exit 0
else
    info "NVIDIA driver already present: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi

# ── 3. CUDA Toolkit 12.1 (if not present) ────────────────────────────────────
if ! nvcc --version &>/dev/null 2>&1; then
    info "Installing CUDA Toolkit 12.1 …"
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    apt-get update -q
    apt-get install -y -q cuda-toolkit-12-1
fi

# ── 4. Python virtual environment ────────────────────────────────────────────
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    info "Creating Python 3.12 virtual environment at ${VENV_DIR} …"
    python3.12 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

# ── 5. PyTorch with CUDA 12.1 ────────────────────────────────────────────────
info "Installing PyTorch 2.3 (CUDA 12.1) …"
pip install --quiet --upgrade pip
pip install --quiet \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 6. FlashAttention 2 ──────────────────────────────────────────────────────
info "Installing FlashAttention 2 (this may take several minutes) …"
MAX_JOBS=4 pip install --quiet flash-attn --no-build-isolation

# ── 7. Application requirements ──────────────────────────────────────────────
info "Installing application requirements …"
pip install --quiet -r "${REPO_DIR}/requirements.txt"

# ── 8. (Optional) pre-download model weights ─────────────────────────────────
MODEL_CACHE="${HOME}/.cache/huggingface/hub"
info "Model cache directory: ${MODEL_CACHE}"
info "The model will be downloaded automatically on first server start."
info "To pre-download now, run:"
info "  source ${VENV_DIR}/bin/activate"
info "  huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# ── 9. Environment file for systemd ──────────────────────────────────────────
ENV_FILE="/etc/qwen3-tts.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    info "Creating environment file at ${ENV_FILE} …"
    cat > "${ENV_FILE}" <<EOF
# Qwen3-TTS Voice-Design Service Configuration
# Edit this file and restart the service: systemctl restart qwen3-tts
TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
TTS_HOST=0.0.0.0
TTS_PORT=8000
TTS_DEVICE=cuda:0
TTS_DTYPE=bfloat16
TTS_FLASH_ATTN=1
TTS_DEFAULT_LANGUAGE=English
TTS_OUTPUT_FORMAT=wav
TTS_MAX_NEW_TOKENS=4096
TTS_DEFAULT_VOICE_INSTRUCT=A warm, clear male narrator voice, mid-30s, natural midrange, measured pace, suitable for audiobook narration.
# Uncomment and set a value to require API key authentication:
# TTS_API_KEY=change-me-before-deploying
EOF
    chmod 600 "${ENV_FILE}"
    info "Environment file created.  Edit ${ENV_FILE} to customise settings."
else
    info "Environment file ${ENV_FILE} already exists."
fi

# ── 10. systemd service ──────────────────────────────────────────────────────
SERVICE_FILE="/etc/systemd/system/qwen3-tts.service"
if [[ ! -f "${SERVICE_FILE}" ]]; then
    info "Installing systemd service …"
    cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Qwen3-TTS Voice-Design API Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=${SUDO_USER:-$(whoami)}
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${ENV_FILE}
Environment="PATH=${VENV_DIR}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=${VENV_DIR}/bin/python -m server.main
Restart=on-failure
RestartSec=10

# Performance: give the service full access to GPU resources
Nice=-5
LimitMEMLOCK=infinity
LimitNOFILE=65536

# Security hardening
ProtectSystem=full
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable qwen3-tts
    info "Service installed.  Start with:  systemctl start qwen3-tts"
else
    info "systemd service already exists."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "Setup complete!"
echo ""
echo "  Activate venv :  source ${VENV_DIR}/bin/activate"
echo "  Start server  :  python -m server.main"
echo "  Or via systemd:  systemctl start qwen3-tts"
echo "  View logs     :  journalctl -u qwen3-tts -f"
echo "  Edit config   :  sudo nano ${ENV_FILE}"
echo "  API docs      :  http://<your-ip>:8000/docs"
echo ""
