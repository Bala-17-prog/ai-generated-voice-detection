import numpy as np
import librosa

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return mfcc_mean, zcr, spectral_centroid


def predict_voice(audio_path):
    mfcc, zcr, centroid = extract_features(audio_path)

    # Simple but effective heuristic
    score = 0

    if mfcc < -200:
        score += 1
    if zcr > 0.08:
        score += 1
    if centroid > 3000:
        score += 1

    if score >= 2:
        return {
            "classification": "AI-generated",
            "confidence": 0.75,
            "explanation": "Audio shows synthetic characteristics (high spectral centroid / unnatural MFCC patterns)"
        }
    else:
        return {
            "classification": "Human",
            "confidence": 0.80,
            "explanation": "Audio patterns match natural human speech"
        }
