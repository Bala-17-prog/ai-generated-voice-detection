import librosa
import numpy as np

def predict_voice(audio_path: str):
    y, sr = librosa.load(audio_path, sr=None)

    # Features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = np.var(mfcc)

    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_var = np.var(spectral_bandwidth)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_var = np.var(zcr)

    rms = librosa.feature.rms(y=y)
    rms_var = np.var(rms)

    # Scoring
    ai_score = 0

    if spectral_flatness < 0.2:
        ai_score += 1
    if mfcc_var < 300:
        ai_score += 1
    if bandwidth_var < 2000:   # 🔥 KEY FEATURE
        ai_score += 1
    if zcr_var < 0.002:
        ai_score += 1
    if rms_var < 0.02:
        ai_score += 1

    # Decision
    if ai_score >= 3:
        classification = "AI_GENERATED"
        confidence = round(min(0.65 + ai_score * 0.07, 0.95), 2)
        explanation = "Low spectral and temporal variance patterns indicate synthetic speech"
    else:
        classification = "HUMAN"
        confidence = round(min(0.65 + (5 - ai_score) * 0.07, 0.95), 2)
        explanation = "High spectral bandwidth and natural temporal variation suggest human speech"

    return {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation
    }
