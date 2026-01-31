from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_voice
import base64
import tempfile
import os

app = FastAPI(title="AI Generated Voice Detection API")


class AudioBase64Request(BaseModel):
    audio_base64: str


@app.post("/detect-voice")
async def detect_voice(request: AudioBase64Request):
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        return {
            "error": "Invalid Base64 audio input"
        }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    try:
        result = predict_voice(temp_audio_path)
    finally:
        os.remove(temp_audio_path)

    return {
        "result": result["classification"],
        "confidence": result["confidence"],
        "explanation": result["explanation"]
    }
