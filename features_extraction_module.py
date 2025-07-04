import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import os
import uuid

def extract_extra_features(file_path):
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_var = np.std(pitch_values)
    pitch_change_rate = np.mean(np.abs(np.diff(pitch_values)))
    return pitch_var, pitch_change_rate

def generate_expressiveness_plot(pitch_var, reference_df):
    return create_scatter(reference_df, pitch_var, "pitch_variability", "Expressiveness", "expressiveness")

def generate_clarity_plot(pitch_rate, reference_df):
    return create_scatter(reference_df, pitch_rate, "pitch_change_rate", "Clarity", "clarity")

def create_scatter(reference_df, val, feature_col, score_col, prefix):
    fig, ax = plt.subplots()
    ax.scatter(reference_df[feature_col], reference_df[score_col], color="orange")
    ax.axvline(val, color="red", linestyle="dashed")
    ax.set_xlabel(feature_col)
    ax.set_ylabel(score_col)
    ax.set_title(f"{score_col} vs {feature_col}")

    filename = f"{prefix}_{uuid.uuid4().hex[:6]}.png"
    filepath = os.path.join("plots", filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    return filename
