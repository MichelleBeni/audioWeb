
from flask import Flask, request, render_template, jsonify
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'static/plots'
REFERENCE_PATH = 'reference_dataset.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_variability = np.std(pitch_values)
    pitch_diff = np.diff(pitch_values)
    pitch_change_rate = np.mean(np.abs(pitch_diff))
    return pitch_variability, pitch_change_rate

def create_scatter(reference_df, x_col, y_col, label, new_x, output_path):
    fig, ax = plt.subplots()
    ax.scatter(reference_df[x_col], reference_df[y_col], color='orange')
    ax.axvline(new_x, color='red', linestyle='--')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{label} vs {x_col}")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_id = str(uuid.uuid4())[:6]
    file_path = os.path.join(UPLOAD_FOLDER, file_id + "_" + file.filename)
    file.save(file_path)

    pitch_var, pitch_change = extract_features(file_path)
    reference_df = pd.read_csv(REFERENCE_PATH)

    expr_path = os.path.join(PLOTS_FOLDER, f"expressiveness_{file_id}.png")
    clar_path = os.path.join(PLOTS_FOLDER, f"clarity_{file_id}.png")
    create_scatter(reference_df, 'pitch_variability', 'Expressiveness', 'Expressiveness', pitch_var, expr_path)
    create_scatter(reference_df, 'pitch_change_rate', 'Clarity', 'Clarity', pitch_change, clar_path)

    return render_template('index.html', pitch_variability=round(pitch_var, 3),
                           pitch_change_rate=round(pitch_change, 3),
                           plots={'expressiveness': '/' + expr_path,
                                  'clarity': '/' + clar_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
