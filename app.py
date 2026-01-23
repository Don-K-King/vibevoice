from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import tempfile
import os
import json
import librosa

from model import asr_service


app = FastAPI(
    title="VibeVoice ASR API",
    description="Audio transcription API using Microsoft's VibeVoice-ASR model",
    version="1.0.0"
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class TranscriptionSegment(BaseModel):
    speaker: str
    timestamp: str
    text: str


class TranscriptionResponse(BaseModel):
    transcription: str
    segments: List[TranscriptionSegment]
    metadata: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("Starting VibeVoice ASR API...")
    try:
        asr_service.load_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model will be loaded on first request")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "VibeVoice ASR API",
        "version": "1.0.0",
        "endpoints": {
            "transcribe": "POST /transcribe",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if asr_service.is_loaded() else "model_not_loaded",
        model_loaded=asr_service.is_loaded(),
        device=asr_service.device
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    context_info: Optional[str] = Form(None, description="JSON string with hotwords/context"),
    max_new_tokens: int = Form(512, description="Maximum tokens to generate"),
    temperature: float = Form(0.0, description="Sampling temperature (0.0 = deterministic)"),
    top_p: float = Form(0.9, description="Nucleus sampling parameter"),
    repetition_penalty: float = Form(1.0, description="Penalty for repeating tokens")
):
    """
    Transcribe audio file with speaker identification and timestamps.

    Args:
        file: Audio file upload (wav, mp3, m4a, flac, etc.)
        context_info: Optional JSON string with hotwords, speaker names, or domain-specific terms
        max_new_tokens: Maximum tokens to generate (default: 512)
        temperature: Sampling temperature (default: 0.0 for deterministic output)
        top_p: Nucleus sampling parameter (default: 0.9)
        repetition_penalty: Penalty for repeating tokens (default: 1.0)

    Returns:
        Transcription with segments containing speaker IDs, timestamps, and text
    """
    # Validate file
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )

    # Check file extension
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(allowed_extensions)}"
        )

    # Load model if not already loaded
    if not asr_service.is_loaded():
        try:
            asr_service.load_model()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )

    # Save uploaded file to temporary location
    temp_file = None
    try:
        # Create temporary file with proper extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            temp_file = temp.name
            contents = await file.read()
            temp.write(contents)

        # Check audio duration (max 60 minutes as per model spec)
        try:
            y, sr = librosa.load(temp_file, sr=None, duration=None)
            duration_seconds = librosa.get_duration(y=y, sr=sr)
            duration_minutes = duration_seconds / 60.0

            if duration_minutes > 60:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Audio duration ({duration_minutes:.1f} min) exceeds maximum of 60 minutes"
                )
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            # If duration check fails, continue anyway (might be format issue)
            print(f"Warning: Could not check audio duration: {e}")

        # Parse context_info if provided
        parsed_context = None
        if context_info:
            try:
                parsed_context = json.loads(context_info)
                if isinstance(parsed_context, dict):
                    # Convert to string format expected by model
                    parsed_context = json.dumps(parsed_context)
            except json.JSONDecodeError:
                # If not valid JSON, use as-is
                parsed_context = context_info

        # Perform transcription
        try:
            result = asr_service.transcribe(
                audio_path=temp_file,
                context_info=parsed_context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {str(e)}"
            )

        # Add duration to metadata
        if 'duration_seconds' in locals():
            result['metadata']['duration'] = f"{duration_seconds:.2f}s"

        return TranscriptionResponse(**result)

    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
