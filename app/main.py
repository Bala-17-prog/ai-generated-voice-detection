from fastapi import FastAPI, UploadFile, File
from app.model import predict_voice
import shutil
import os

app = FastAPI(title="AI Generated Voice Detection API")


@app.post("/detect-voice")
async def detect_voice(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_voice(temp_file)

    os.remove(temp_file)

    return {
        "result": result["classification"],
        "confidence": result["confidence"],
        "explanation": result["explanation"]
    }
