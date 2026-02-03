import librosa
import numpy as np


# 🔹 Fast audio loader (CRITICAL for timeout safety)
def load_audio_fast(audio_path, sr=16000, duration=6.0):
    """
    Loads only the first `duration` seconds of audio.
    Prevents timeout for long audio files.
    """
    y, sr = librosa.load(
        audio_path,
        sr=sr,
        mono=True,
        duration=duration
    )
    return y, sr


def predict_voice(audio_path: str):
    # 🚀 FAST LOAD (only first 6 seconds)
    y, sr = load_audio_fast(audio_path)

    # Safety check
    if y is None or len(y) == 0:
        return {
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "explanation": "Audio could not be processed or contains no signal."
        }

    # ---- Feature Extraction ----
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

    # ---- Scores ----
    ai_score = 0.0
    human_score = 0.0

    # ---- AI Indicators (WEAKENED) ----
    if spectral_flatness < 0.18:
        ai_score += 0.2
    if pitch_std < 8:
        ai_score += 0.2
    if spectral_bandwidth < 1400:
        ai_score += 0.2

    # ---- HUMAN Indicators (STRENGTHENED) ----
    if pitch_std > 15:
        human_score += 0.4
    if spectral_bandwidth > 1800:
        human_score += 0.3
    if zcr > 0.04:
        human_score += 0.2
    if rms > 0.02:
        human_score += 0.1

    # ---- Decision ----
    if human_score >= ai_score:
        classification = "HUMAN"
        confidence = min(0.6 + human_score, 0.95)
        explanation = (
            "Natural pitch variation, broader spectral bandwidth, and temporal irregularities "
            "indicate human speech."
        )
    else:
        classification = "AI_GENERATED"
        confidence = min(0.6 + ai_score, 0.95)
        explanation = (
            "Stable pitch, reduced spectral variability, and uniform signal patterns "
            "suggest AI-generated voice."
        )

    # ---- STRONG AI FALLBACK (HIGH-CONFIDENCE OVERRIDE) ----
    if (
        classification == "HUMAN"
        and pitch_std < 6
        and spectral_flatness < 0.15
        and spectral_bandwidth < 1200
    ):
        classification = "AI_GENERATED"
        confidence = round(confidence - 0.1, 2)
        explanation = (
            "Despite human-like qualities, extremely stable pitch and spectral consistency "
            "indicate high-confidence AI-generated voice."
        )

    return {
        "classification": classification,
        "confidence": round(float(confidence), 2),
        "explanation": explanation
    }
