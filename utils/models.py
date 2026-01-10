# utils/models.py
import torch
import librosa
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)

# Select device (CPU on Streamlit Cloud)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global cached models
_processor = None
_asr_model = None
_summarizer = None


def load_models():
    """
    Load and cache AI models to avoid reloading on every request.
    Models used:
    - Whisper (speech-to-text)
    - DistilBART (text summarization)
    """
    global _processor, _asr_model, _summarizer

    if _processor is None or _asr_model is None or _summarizer is None:
        _processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        _asr_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small"
        ).to(device)

        _summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6"
        )

    return _processor, _asr_model, _summarizer


def transcribe_audio(audio_path: str) -> str:
    """
    Convert lecture audio file into text using Whisper.
    """
    processor, asr_model, _ = load_models()

    speech, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        predicted_ids = asr_model.generate(**inputs)

    transcript = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcript


def summarize_text(text: str) -> str:
    """
    Generate a concise summary from lecture transcript.
    """
    _, _, summarizer = load_models()

    return summarizer(
        text,
        max_length=300,
        min_length=50,
        do_sample=False
    )[0]["summary_text"]
