import base64
import uuid

def save_base64_audio(base64_audio):
    audio_bytes = base64.b64decode(base64_audio)
    filename = f"temp_{uuid.uuid4().hex}.mp3"

    with open(filename, "wb") as f:
        f.write(audio_bytes)

    return filename
