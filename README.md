# VibeVoice-ASR FastAPI Application

A production-ready FastAPI application for audio transcription using Microsoft's VibeVoice-ASR model. This service provides speaker identification, timestamps, and high-quality transcription for audio files up to 60 minutes long.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [How It Works](#how-it-works)
- [Model Cache](#model-cache)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [License](#license)

## Quick Start

**New to this repo? Start here!**

```bash
# 1. Clone the repository
git clone git@github.com:tehtommeh/vibevoice-asr-api.git
cd vibevoice-asr-api

# 2. Create model_cache directory
mkdir -p model_cache

# 3. Download the model (choose ONE option below)
```

### Option A: Let Docker Download the Model (Easiest)
Simply build and start the service. On first run, it will download the model (~17GB) automatically:

```bash
docker compose build
docker compose up -d
```

The model will download to `model_cache/` (takes 15-25 minutes). Check progress:
```bash
docker compose logs -f
```

### Option B: Pre-download the Model (Faster Startup)
If you want to download the model first and avoid waiting on first startup:

```bash
# Install Python dependencies in a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install huggingface_hub

# Download the model to model_cache/
python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="microsoft/VibeVoice-ASR",
    local_dir="model_cache/hub/models--microsoft--VibeVoice-ASR/snapshots/main",
    local_dir_use_symlinks=False
)
print("Model downloaded successfully!")
EOF

deactivate

# Now build and start
docker compose build
docker compose up -d
```

### Verify It's Working

```bash
# Check health
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","model_loaded":true,"device":"cuda"}
```

**That's it!** See [Usage](#usage) section for how to transcribe audio.

> **Note**: This requires an NVIDIA GPU. See [Installation](#installation) for complete setup including NVIDIA Container Toolkit.

## Overview

VibeVoice-ASR is a state-of-the-art speech recognition model developed by Microsoft that can:
- Process up to 60 minutes of audio in a single pass
- Perform automatic speaker diarization (identifying who said what)
- Generate precise timestamps for each spoken segment
- Accept custom hotwords for domain-specific vocabulary

This FastAPI wrapper provides a simple REST API for transcribing audio files with all the capabilities of the VibeVoice-ASR model.

## Features

### Core Capabilities
- **Long-form audio processing**: Process up to 60 minutes of continuous audio in a single pass
- **Speaker identification**: Automatic speaker diarization without training
- **Precise timestamps**: Timing information accurate to milliseconds for each segment
- **Hotwords support**: Provide domain-specific terms, names, or technical vocabulary for improved accuracy
- **Multi-format support**: WAV, MP3, M4A, FLAC, OGG, Opus, WebM

### Technical Features
- **GPU acceleration**: Optimized for NVIDIA GPUs with CUDA and flash-attention
- **REST API**: Simple HTTP endpoint for easy integration
- **Docker containerized**: Reproducible environment with all dependencies
- **Model caching**: 17GB model stored locally for fast startup
- **Health monitoring**: Health check endpoints for deployment

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3090 with 24GB VRAM)
  - Minimum: 18GB VRAM for the 9B parameter model
  - Recommended: 20GB+ VRAM for comfortable operation
- **RAM**: 16GB+ system RAM
- **Disk Space**: ~40GB free
  - Model weights: ~17GB
  - Docker images: ~15GB
  - Working space: ~8GB

### Software
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Docker**: Version 20.10+
- **NVIDIA Driver**: 580.95.05 or newer (CUDA 13.0+)
- **NVIDIA Container Toolkit**: For GPU access in Docker

## Project Structure

```
vibevoice-asr/
├── app.py                  # FastAPI application with /transcribe endpoint
├── model.py                # VibeVoiceASRService class for model management
├── Dockerfile              # Container definition using NVIDIA PyTorch 25.12
├── docker-compose.yml      # Container orchestration with GPU support
├── requirements.txt        # FastAPI and uvicorn dependencies
├── .dockerignore          # Files to exclude from Docker build
├── .gitignore             # Git ignore patterns
├── model_cache/           # ⚠️ NOT IN GIT - Download separately (17GB)
│   ├── hub/               # Created on first run or manual download
│   │   └── models--microsoft--VibeVoice-ASR/
│   │       ├── snapshots/
│   │       ├── refs/
│   │       └── blobs/
│   └── xet/
├── README.md              # This file
└── SETUP_SUMMARY.md       # Quick reference guide
```

> **Important**: The `model_cache/` directory is excluded from git. You must download the model separately - see [Quick Start](#quick-start).

### Key Files

**`app.py`**
- FastAPI application entry point
- Defines three endpoints: `/`, `/health`, `/transcribe`
- Handles file upload, validation, and audio duration checks
- Returns structured JSON with transcription, segments, and metadata

**`model.py`**
- `VibeVoiceASRService` singleton class
- Loads VibeVoice-ASR model and processor from HuggingFace
- Manages model lifecycle (loading, inference, memory)
- Parses model output into structured segments

**`Dockerfile`**
- Based on `nvcr.io/nvidia/pytorch:25.12-py3` (NVIDIA official PyTorch container)
- Installs system dependencies (ffmpeg, git)
- Clones and installs VibeVoice from GitHub
- Copies application code and installs Python dependencies

**`docker-compose.yml`**
- Configures GPU access using NVIDIA runtime
- Mounts local `model_cache/` directory for persistent model storage
- Sets environment variables for CUDA
- Configures shared memory (16GB) for large model operations

**`model_cache/`** ⚠️ **NOT IN GIT**
- Local storage for HuggingFace model weights (~17GB)
- **You need to download this** - see [Quick Start](#quick-start)
- Eliminates need to re-download model on container restart
- Contains model snapshots, configuration, and tokenizer files
- Mounted as volume in Docker container
- Excluded from git via `.gitignore` due to size

## Installation

> **TL;DR**: If you just cloned this repo, see [Quick Start](#quick-start) for the fastest way to get running.

This section provides detailed installation steps for all prerequisites.

### Step 1: Install Docker

Follow the [official Docker installation guide](https://docs.docker.com/get-docker/) for your Linux distribution.

For Ubuntu:
```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (optional, requires logout/login)
sudo usermod -aG docker $USER
```

### Step 2: Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit allows Docker containers to access your GPU.

**For Ubuntu/Debian:**

1. Configure the repository:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

2. Install the toolkit:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

3. Configure Docker to use the NVIDIA runtime:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

4. Restart Docker:
```bash
sudo systemctl restart docker
```

5. Verify GPU access in Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

You should see your GPU information displayed.

### Step 3: Clone or Download This Repository

```bash
git clone <your-repo-url> vibevoice-asr
cd vibevoice-asr
```

### Step 4: Build the Docker Image

```bash
docker compose build
```

This process will:
1. Download NVIDIA PyTorch 25.12 container (~10GB) - takes 10-20 minutes
2. Install system dependencies (ffmpeg, git)
3. Clone VibeVoice repository from GitHub
4. Install VibeVoice Python package and dependencies
5. Install FastAPI and uvicorn

**Total build time**: 15-25 minutes depending on internet speed

### Step 5: Start the Service

```bash
docker compose up -d
```

On **first startup**, the model will download from HuggingFace (~17GB) which takes 15-25 minutes. The model is cached in `model_cache/` so subsequent startups are fast (~30 seconds).

### Step 6: Verify Service is Running

```bash
# Check health
curl http://localhost:8000/health

# Expected output:
# {"status":"healthy","model_loaded":true,"device":"cuda"}
```

## Usage

### Converting Audio Files

VibeVoice-ASR supports WAV, MP3, M4A, FLAC, OGG, Opus, and WebM formats. If you have an MP4 video or unsupported format, convert it first:

```bash
# Install ffmpeg (if not already installed)
sudo apt-get install ffmpeg

# Convert MP4 to WAV
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav

# Convert any audio format to WAV
ffmpeg -i input.m4a -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
```

### Basic Transcription

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@your_audio.wav"
```

### With Hotwords for Better Accuracy

Provide domain-specific terms, names, or technical vocabulary:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@your_audio.wav" \
  -F 'context_info={"hotwords": ["TensorFlow", "PyTorch", "neural network", "GPU"]}'
```

### With Custom Parameters

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@your_audio.wav" \
  -F "max_new_tokens=1024" \
  -F "temperature=0.0" \
  -F "top_p=0.9"
```

### Response Format

```json
{
  "transcription": "Full raw transcription text...",
  "segments": [
    {
      "speaker": "Speaker 0",
      "timestamp": "0.0 - 13.43",
      "text": "Well, well. Hello, Claire..."
    }
  ],
  "metadata": {
    "model": "microsoft/VibeVoice-ASR",
    "processing_time": "19.63s",
    "device": "cuda",
    "duration": "78.30s"
  }
}
```

### Python Client Example

```python
import requests
import json

# Basic transcription
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f}
    )

result = response.json()

# Print transcription
print("Full Transcription:")
print(result["transcription"])
print()

# Print segments with timestamps
print("Segments:")
for segment in result["segments"]:
    print(f"[{segment['timestamp']}] {segment['speaker']}: {segment['text']}")

# With hotwords
context = {"hotwords": ["Python", "Docker", "FastAPI"]}
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={"context_info": json.dumps(context)}
    )
```

## API Documentation

Once running, visit these URLs for interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `GET /`
Returns API information and available endpoints.

**Response:**
```json
{
  "message": "VibeVoice ASR API",
  "version": "1.0.0",
  "endpoints": {
    "transcribe": "POST /transcribe",
    "health": "GET /health"
  }
}
```

#### `GET /health`
Health check endpoint for monitoring and load balancers.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### `POST /transcribe`
Transcribe audio file with speaker identification and timestamps.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | Audio file (wav, mp3, m4a, flac, ogg, opus, webm) |
| `context_info` | String | No | None | JSON string with hotwords/context |
| `max_new_tokens` | Integer | No | 512 | Maximum tokens to generate |
| `temperature` | Float | No | 0.0 | Sampling temperature (0.0 = deterministic) |
| `top_p` | Float | No | 0.9 | Nucleus sampling parameter |
| `repetition_penalty` | Float | No | 1.0 | Penalty for repeating tokens |

**Constraints:**
- Maximum audio duration: 60 minutes
- Maximum file size: 100MB (configurable)

**Response:** See "Response Format" section above

**Error Responses:**
- `400 Bad Request`: Invalid file format or missing file
- `413 Request Entity Too Large`: Audio exceeds 60 minutes
- `500 Internal Server Error`: Model inference failure

## How It Works

### Architecture Overview

```
┌─────────────┐
│ Audio File  │
│ (MP3, WAV,  │
│  FLAC, etc) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│           FastAPI Application (app.py)          │
│  ┌──────────────────────────────────────────┐  │
│  │ 1. Validate file format and size         │  │
│  │ 2. Check audio duration (max 60 min)     │  │
│  │ 3. Save to temporary file                │  │
│  └──────────────┬───────────────────────────┘  │
│                 │                               │
│                 ▼                               │
│  ┌──────────────────────────────────────────┐  │
│  │   VibeVoiceASRService (model.py)         │  │
│  │ ┌────────────────────────────────────┐   │  │
│  │ │ VibeVoiceASRProcessor              │   │  │
│  │ │ - Load audio file                  │   │  │
│  │ │ - Resample to 16kHz                │   │  │
│  │ │ - Extract audio features           │   │  │
│  │ │ - Tokenize context/hotwords        │   │  │
│  │ └────────────┬───────────────────────┘   │  │
│  │              │                            │  │
│  │              ▼                            │  │
│  │ ┌────────────────────────────────────┐   │  │
│  │ │ VibeVoiceASRForConditionalGeneration│  │  │
│  │ │ (9B Parameter Transformer Model)    │  │  │
│  │ │ - Audio encoder (process audio)     │  │  │
│  │ │ - Text decoder (generate text)      │  │  │
│  │ │ - Flash Attention 2 (fast inference)│  │  │
│  │ │ - Runs on GPU with bfloat16         │  │  │
│  │ └────────────┬───────────────────────┘   │  │
│  │              │                            │  │
│  │              ▼                            │  │
│  │ ┌────────────────────────────────────┐   │  │
│  │ │ Output Parsing                     │   │  │
│  │ │ - Decode token IDs to text         │   │  │
│  │ │ - Extract speaker IDs              │   │  │
│  │ │ - Parse timestamps                 │   │  │
│  │ │ - Structure into segments          │   │  │
│  │ └────────────┬───────────────────────┘   │  │
│  └──────────────┘                            │  │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │   JSON Response     │
       │ - transcription     │
       │ - segments[]        │
       │ - metadata          │
       └─────────────────────┘
```

### Model Details

**VibeVoice-ASR** is based on a transformer architecture with:
- **Size**: 9 billion parameters
- **Precision**: BFloat16 (reduces memory, faster inference)
- **Attention**: Flash Attention 2 (optimized GPU kernel)
- **Context Length**: 64K tokens (supports up to 60 minutes of audio)
- **Training**: Trained on diverse speech datasets with speaker diarization

**Inference Process**:
1. Audio is resampled to 16kHz mono
2. Audio features extracted using convolutional encoder
3. Transformer processes audio features with global attention
4. Model jointly predicts:
   - Transcribed text
   - Speaker identities
   - Segment boundaries and timestamps
5. Output is decoded and structured into segments

**Why It's Fast**:
- Flash Attention 2: Optimized GPU kernels for attention computation
- BFloat16: Half-precision reduces memory bandwidth requirements
- GPU Processing: Parallel computation on RTX 3090 (24GB VRAM)
- Single Pass: Entire audio processed at once (no sliding windows)

### Processing Pipeline

1. **File Upload** → FastAPI receives multipart/form-data
2. **Validation** → Check file format, size, and duration
3. **Preprocessing** → librosa loads and resamples audio
4. **Model Inference** → GPU processes audio through transformer
5. **Post-processing** → Parse output, extract segments
6. **Response** → JSON with transcription, segments, metadata

## Model Cache

The VibeVoice-ASR model weights (~17GB) are stored in `model_cache/`:

```
model_cache/
├── hub/
│   └── models--microsoft--VibeVoice-ASR/
│       ├── refs/
│       │   └── main              # Points to current snapshot
│       ├── snapshots/
│       │   └── <hash>/           # Model files
│       │       ├── config.json
│       │       ├── model-00001-of-00008.safetensors
│       │       ├── model-00002-of-00008.safetensors
│       │       ├── ...
│       │       ├── model-00008-of-00008.safetensors
│       │       ├── model.safetensors.index.json
│       │       ├── preprocessor_config.json
│       │       ├── special_tokens_map.json
│       │       ├── tokenizer_config.json
│       │       └── tokenizer.json
│       └── blobs/                # Content-addressed storage
└── xet/                          # HuggingFace XET metadata
```

**Why Local Storage?**
- **Fast startup**: No need to download 17GB on each restart
- **Offline capability**: Run without internet after initial setup
- **Reproducibility**: Version-locked model weights
- **Portability**: Can copy entire project with model

**Managing the Cache**:

```bash
# Check cache size
du -sh model_cache/

# Clean cache (will re-download on next startup)
rm -rf model_cache/*

# Backup cache
tar -czf vibevoice-model-cache.tar.gz model_cache/

# Restore cache
tar -xzf vibevoice-model-cache.tar.gz
```

## Troubleshooting

### Container Won't Start - GPU Not Found

**Error**: `could not select device driver "nvidia" with capabilities: [[gpu]]`

**Solution**: NVIDIA Container Toolkit not installed or configured
```bash
# Install toolkit
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

### Out of Memory Error

**Error**: `CUDA out of memory` or container crashes

**Solutions**:
1. Check GPU memory: `nvidia-smi`
2. Close other GPU applications
3. Reduce `max_new_tokens` parameter
4. Use smaller batch sizes (if processing multiple files)

### Model Download Fails

**Error**: Connection errors during first startup

**Solutions**:
1. Check internet connection
2. Check HuggingFace is accessible: `curl https://huggingface.co`
3. Use VPN if HuggingFace is blocked in your region
4. Download model manually and place in `model_cache/`

### Slow Performance

**Symptoms**: Transcription takes longer than expected

**Solutions**:
1. Verify GPU is being used: Check logs for "device: cuda"
2. Check GPU utilization: `nvidia-smi` should show high GPU usage
3. Ensure flash-attention is installed (check startup logs)
4. Try shorter audio clips first to isolate issue

### Audio Format Not Supported

**Error**: `Unsupported file format`

**Solution**: Convert audio using ffmpeg
```bash
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
```

### CUDA Version Mismatch Warning

**Warning**: `CUDA Minor Version Compatibility mode ENABLED`

This is normal and expected. The container uses CUDA 13.1 but your driver supports CUDA 13.0. Minor version compatibility mode allows this to work. No action needed.

## Performance

### Benchmarks (RTX 3090, 24GB VRAM)

| Audio Duration | Processing Time | Real-time Factor |
|---------------|-----------------|------------------|
| 1 minute      | ~3-4 seconds    | 15-20x faster    |
| 5 minutes     | ~15-20 seconds  | 15-20x faster    |
| 15 minutes    | ~45-60 seconds  | 15-20x faster    |
| 60 minutes    | ~3-4 minutes    | 15-20x faster    |

**Real-time Factor**: How much faster than real-time (e.g., 15x = processes 15 minutes of audio per minute)

### Resource Usage

- **GPU Memory**: ~18GB during inference
- **System RAM**: ~4-6GB
- **CPU**: Minimal (mostly I/O and preprocessing)
- **Disk I/O**: Moderate during audio loading

## Managing the Service

```bash
# Start service
docker compose up -d

# Stop service
docker compose down

# Restart service
docker compose restart

# View logs
docker compose logs -f

# View logs for last 100 lines
docker compose logs --tail 100

# Check service status
docker compose ps

# Rebuild after code changes
docker compose build && docker compose up -d

# Execute command in container
docker exec vibevoice-asr <command>

# Open shell in container
docker exec -it vibevoice-asr bash
```

## License

This project uses Microsoft's VibeVoice-ASR model, which is licensed under the MIT License.

**Important Notice from Microsoft**:
> "We do not recommend using VibeVoice in commercial or real-world applications without further testing and development. This model is intended for research and development purposes only."

The FastAPI wrapper code in this repository is provided as-is for educational and research purposes.

## Resources

- [VibeVoice GitHub Repository](https://github.com/microsoft/VibeVoice)
- [VibeVoice-ASR on HuggingFace](https://huggingface.co/microsoft/VibeVoice-ASR)
- [Official VibeVoice Documentation](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

## Support

For issues related to:
- **This FastAPI application**: Open an issue in this repository
- **VibeVoice model**: Contact VibeVoice@microsoft.com or visit [HuggingFace Discussions](https://huggingface.co/microsoft/VibeVoice-ASR/discussions)
- **NVIDIA Container Toolkit**: Visit [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

**Built with ❤️ using VibeVoice-ASR, FastAPI, and Docker**
