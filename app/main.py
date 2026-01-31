from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.model import predict_voice
import base64
import tempfile
import os
import uuid

app = FastAPI(
    title="AI Generated Voice Detection API",
    description="Detects whether a voice sample is AI-generated or Human using audio signal analysis.",
    version="1.0.0"
)


class AudioBase64Request(BaseModel):
    audio_base64: str = Field(
        ...,
        description="Base64-encoded MP3 audio string (without or with data:audio/*;base64 prefix)"
    )


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AI Generated Voice Detection API is running"
    }


@app.post("/detect-voice")
async def detect_voice(request: AudioBase64Request):
    # 1️⃣ Clean Base64 (remove data URI if present)
    try:
        audio_base64 = request.audio_base64
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")[1]

        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 audio input"
        )

    # 2️⃣ Save temp MP3 file
    temp_file_path = f"{uuid.uuid4()}.mp3"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(audio_bytes)

        # 3️⃣ Run prediction
        result = predict_voice(temp_file_path)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}"
        )

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # 4️⃣ Return hackathon-compliant response
    return {
        "result": result["classification"],   # AI_GENERATED or HUMAN
        "confidence": float(result["confidence"]),  # 0.0 – 1.0
        "explanation": result["explanation"]
    }
