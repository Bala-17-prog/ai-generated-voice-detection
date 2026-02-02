import librosa
import numpy as np


def predict_voice(audio_path: str):
    """
    Predict whether the given audio is AI-generated or Human.
    Returns classification, confidence, and explanation.
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # ---- Feature Extraction ----
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0.0

    # ---- Base Scoring ----
    ai_score = 0.0
    human_score = 0.0

    # AI voice traits
    if spectral_flatness < 0.25:
        ai_score += 0.25
    if pitch_std < 15:
        ai_score += 0.25
    if zcr < 0.04:
        ai_score += 0.2
    if spectral_bandwidth < 1800:
        ai_score += 0.2

    # Human voice traits
    if pitch_std > 20:
        human_score += 0.3
    if spectral_bandwidth > 2000:
        human_score += 0.3
    if zcr > 0.05:
        human_score += 0.2
    if rms > 0.03:
        human_score += 0.2

    # ---- Initial Decision ----
    if ai_score > human_score:
        classification = "AI_GENERATED"
        confidence = min(0.6 + ai_score, 0.95)
        explanation = (
            "Consistent pitch patterns, low spectral flatness, and reduced temporal variability "
            "suggest synthetic voice generation."
        )
    else:
        classification = "HUMAN"
        confidence = min(0.6 + human_score, 0.95)
        explanation = (
            "Natural pitch variations, wider spectral bandwidth, and irregular temporal dynamics "
            "indicate human speech."
        )

    # ---- AI-Bias Heuristic (IMPORTANT FIX) ----
    # Catch high-quality AI voices misclassified as human
    if (
        classification == "HUMAN"
        and confidence > 0.8
        and (spectral_flatness < 0.2 or pitch_std < 10)
    ):
        classification = "AI_GENERATED"
        confidence = round(confidence - 0.15, 2)
        explanation = (
            "Although resembling human speech, unnaturally stable pitch and spectral consistency "
            "indicate high-quality AI-generated voice."
        )

    return {
        "classification": classification,
        "confidence": round(float(confidence), 2),
        "explanation": explanation
    }
