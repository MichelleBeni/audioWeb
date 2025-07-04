
import numpy as np
import librosa
import os

def extract_extra_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Use pyin to extract F0 (pitch) values
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

        f0 = f0[~np.isnan(f0)]
        pitch_variability = np.std(f0) if len(f0) > 0 else 0
        pitch_change_rate = np.mean(np.abs(np.diff(f0))) if len(f0) > 1 else 0

        return {
            'pitch_variability': float(pitch_variability),
            'pitch_change_rate': float(pitch_change_rate),
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            'pitch_variability': 0,
            'pitch_change_rate': 0
        }
