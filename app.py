import os
from flask import Flask, request, render_template, send_from_directory
from features_extraction_module import extract_extra_features, generate_expressiveness_plot, generate_clarity_plot
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PLOTS_FOLDER = "plots"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PLOTS_FOLDER"] = PLOTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

reference_df = pd.read_csv("reference_dataset.csv")

@app.route("/")
def index():
    return render_template("upload_interface.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return "Error: No file uploaded", 400
    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    pitch_var, pitch_rate = extract_extra_features(filepath)

    expressiveness_filename = generate_expressiveness_plot(pitch_var, reference_df)
    clarity_filename = generate_clarity_plot(pitch_rate, reference_df)

    return render_template("upload_interface.html",
                           pitch_var=round(pitch_var, 3),
                           pitch_rate=round(pitch_rate, 3),
                           expressiveness_plot=expressiveness_filename,
                           clarity_plot=clarity_filename)

@app.route("/plots/<filename>")
def plot_file(filename):
    return send_from_directory(app.config["PLOTS_FOLDER"], filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
