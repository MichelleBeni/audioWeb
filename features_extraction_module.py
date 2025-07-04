import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def extract_extra_features(audio_file_path):
    import librosa
    y, sr = librosa.load(audio_file_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_variability = np.std(pitch_values)
    pitch_change_rate = np.mean(np.abs(np.diff(pitch_values)))

    return {
        'pitch_variability': float(pitch_variability),
        'pitch_change_rate': float(pitch_change_rate),
    }

def generate_expressiveness_plot(pitch_val, reference_df):
    fig, ax = plt.subplots()
    ax.scatter(reference_df['pitch_variability'], reference_df['Expressiveness'], color='orange')
    ax.axvline(x=pitch_val, color='red', linestyle='--')
    ax.set_xlabel("pitch_variability")
    ax.set_ylabel("Expressiveness")
    ax.set_title("Expressiveness vs pitch_variability")
    return fig

def generate_clarity_plot(pitch_change_val, reference_df):
    fig, ax = plt.subplots()
    ax.scatter(reference_df['pitch_change_rate'], reference_df['Clarity'], color='orange')
    ax.axvline(x=pitch_change_val, color='red', linestyle='--')
    ax.set_xlabel("pitch_change_rate")
    ax.set_ylabel("Clarity")
    ax.set_title("Clarity vs pitch_change_rate")
    return fig