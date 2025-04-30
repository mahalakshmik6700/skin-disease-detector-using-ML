import os
import joblib
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, request
app = Flask(__name__,template_folder='../templates',static_folder='../static')
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import numpy as np
import pickle
from scripts.feature_extraction import extract_hog_features
base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir,'models','hist_gradient_boosting_model.pkl')
print("Model path:", model_path)
print("Files in models/:", os.listdir(os.path.join(base_dir, 'models')))
model=joblib.load(model_path)
encoder_path = os.path.join(base_dir,'models', 'label_encoder.pkl')
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png','jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model and label encoder

with open(encoder_path,'rb')as f:
    label_encoder=joblib.load(encoder_path)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract HOG features from the uploaded image
        features = extract_hog_features(file_path)
        prediction = model.predict([features])[0]
        label = label_encoder.inverse_transform([prediction])[0]

        return render_template('result.html', label=label, filename=filename)

    return "Invalid file", 400
if __name__ == '__main__':
    app.run(debug=True)
