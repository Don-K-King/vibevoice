import torch
import time
import tempfile
import os
from typing import Dict, Any, Optional
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
import re


class VibeVoiceASRService:
    """Singleton service for VibeVoice ASR model inference."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the ASR model and processor."""
        if not self._initialized:
            self.model = None
            self.processor = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"VibeVoiceASRService initializing on device: {self.device}")
            self._initialized = True

    def load_model(self, model_path: str = "microsoft/VibeVoice-ASR"):
        """Load the model and processor."""
        if self.model is not None:
            print("Model already loaded")
            return

        print(f"Loading VibeVoice ASR model from {model_path}...")
        start_time = time.time()

        # Load processor
        self.processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        print("Processor loaded")

        # Load model
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=self.device if self.device == "cuda" else None,
            attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()

        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds")

    def transcribe(
        self,
        audio_path: str,
        context_info: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0
    ) -> Dict[str, Any]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            context_info: Optional context/hotwords for improved accuracy
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Dictionary with transcription, segments, and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Prepare inputs
        inputs = self.processor(
            audio=audio_path,
            sampling_rate=None,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            context_info=context_info
        )

        # Move to device
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate transcription
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        # Decode results
        generated_text = self.processor.decode(output_ids[0], skip_special_tokens=True)

        processing_time = time.time() - start_time

        # Parse segments from output
        segments = self._parse_segments(generated_text)

        return {
            "transcription": generated_text,
            "segments": segments,
            "metadata": {
                "model": "microsoft/VibeVoice-ASR",
                "processing_time": f"{processing_time:.2f}s",
                "device": self.device
            }
        }

    def _parse_segments(self, text: str) -> list:
        """
        Parse structured segments from transcription text.

        Expected format includes speaker IDs and timestamps like:
        [Speaker 1] [00:00:00 - 00:00:05] text here
        """
        segments = []

        # Pattern to match speaker and timestamp format
        # This is a heuristic parser - adjust based on actual model output format
        pattern = r'\[([^\]]+)\]\s*\[([^\]]+)\]\s*([^\[]+)'

        matches = re.finditer(pattern, text)
        for match in matches:
            speaker = match.group(1).strip()
            timestamp = match.group(2).strip()
            segment_text = match.group(3).strip()

            segments.append({
                "speaker": speaker,
                "timestamp": timestamp,
                "text": segment_text
            })

        # If no segments found, return the whole text as a single segment
        if not segments:
            segments.append({
                "speaker": "unknown",
                "timestamp": "unknown",
                "text": text
            })

        return segments

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Global instance
asr_service = VibeVoiceASRService()
