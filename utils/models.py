# utils/models.py
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)
from huggingface_hub import login
import os

# Optionally log in if using private models
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache models globally
_processor, _asr_model, _summarizer = None, None, None


def load_models():
    global _processor, _asr_model, _summarizer
    if _processor is None or _asr_model is None or _summarizer is None:
        _processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        _asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _processor, _asr_model, _summarizer


def transcribe_audio(audio_path: str):
    import librosa
    processor, asr_model, _ = load_models()
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted_ids = asr_model.generate(**inputs)
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcript


def summarize_text(text: str):
    _, _, summarizer = load_models()
    return summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
