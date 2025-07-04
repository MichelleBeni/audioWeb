
import os
import uuid
from flask import Flask, request, render_template, jsonify
from features_extraction_module import extract_extra_features, generate_expressiveness_plot, generate_clarity_plot
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'plots'
REFERENCE_DATASET = 'reference_dataset.csv'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

reference_df = pd.read_csv(REFERENCE_DATASET)

@app.route('/')
def index():
    return render_template('upload_interface.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['audio']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    features = extract_extra_features(filename)
    pitch_var = features['pitch_variability']
    pitch_rate = features['pitch_change_rate']

    expressiveness_plot = generate_expressiveness_plot(pitch_var, reference_df)
    clarity_plot = generate_clarity_plot(pitch_rate, reference_df)

    expressiveness_filename = f"expressiveness_{uuid.uuid4().hex[:6]}.png"
    clarity_filename = f"clarity_{uuid.uuid4().hex[:6]}.png"

    expressiveness_plot.savefig(os.path.join(PLOTS_FOLDER, expressiveness_filename), bbox_inches='tight')
    clarity_plot.savefig(os.path.join(PLOTS_FOLDER, clarity_filename), bbox_inches='tight')

    return jsonify({
        'pitch_variability': pitch_var,
        'pitch_change_rate': pitch_rate,
        'plots': {
            'expressiveness': f'/plots/{expressiveness_filename}',
            'clarity': f'/plots/{clarity_filename}'
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
