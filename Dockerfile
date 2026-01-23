# Use NVIDIA PyTorch container as recommended by Microsoft
FROM nvcr.io/nvidia/pytorch:25.12-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone VibeVoice repository
RUN git clone https://github.com/microsoft/VibeVoice.git /tmp/VibeVoice

# Install VibeVoice with ASR dependencies
WORKDIR /tmp/VibeVoice
RUN pip install --no-cache-dir -e .[asr]

# Copy application files
WORKDIR /app
COPY requirements.txt .
COPY app.py .
COPY model.py .

# Install FastAPI dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5m --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
