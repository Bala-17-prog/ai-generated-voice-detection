import librosa
import numpy as np

def predict_voice(audio_path: str):
    y, sr = librosa.load(audio_path, sr=None)

    # --- Feature extraction ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc)
    mfcc_var = np.var(mfcc)

    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_var = np.var(zcr)

    rms = librosa.feature.rms(y=y)
    rms_var = np.var(rms)

    # --- Heuristic scoring ---
    ai_score = 0

    if spectral_flatness < 0.15:
        ai_score += 1
    if mfcc_var < 200:
        ai_score += 1
    if zcr_var < 0.001:
        ai_score += 1
    if rms_var < 0.01:
        ai_score += 1

    # --- Final decision ---
    if ai_score >= 2:
        classification = "AI_GENERATED"
        confidence = min(0.6 + ai_score * 0.1, 0.95)
        explanation = "Spectral smoothness and low variance indicate synthetic speech"
    else:
        classification = "HUMAN"
        confidence = min(0.6 + (4 - ai_score) * 0.1, 0.95)
        explanation = "Natural variation in pitch and energy suggests human speech"

    return {
        "classification": classification,
        "confidence": round(confidence, 2),
        "explanation": explanation
    }
