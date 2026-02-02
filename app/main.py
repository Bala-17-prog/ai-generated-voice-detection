from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import predict_voice
import base64
import tempfile
import os

app = FastAPI(
    title="AI Generated Voice Detection API",
    description="Detects whether a voice sample is AI-generated or Human",
    version="1.0.0"
)

# ✅ REQUEST MODEL (MATCHES GUVI TESTER EXACTLY)
class DetectVoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# ✅ HEALTH CHECK (REQUIRED)
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AI Generated Voice Detection API is running"
    }


# ✅ MAIN ENDPOINT
@app.post("/detect-voice")
async def detect_voice(request: DetectVoiceRequest):
    try:
        # Remove data URI prefix if present
        audio_base64 = request.audioBase64
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")[1]

        # Decode Base64
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 audio input"
        )

    # Save temp audio file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{request.audioFormat}"
    ) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    try:
        # Run model
        result = predict_voice(temp_audio_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    # ✅ HACKATHON-COMPLIANT RESPONSE
    return {
        "result": result["classification"],  # AI_GENERATED or HUMAN
        "confidence": float(result["confidence"]),  # 0.0–1.0
        "explanation": result["explanation"]
    }
