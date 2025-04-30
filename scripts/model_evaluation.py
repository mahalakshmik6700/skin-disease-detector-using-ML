import os
import joblib
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
MODEL_DIR = "models"
VAL_FEATURES_PATH = "val_hog_features.pkl"
VAL_LABELS_PATH = "val_labels.npy"

# Load validation data
print("Loading validation data...")
val_features = pickle.load(open(VAL_FEATURES_PATH, "rb"))
val_labels = np.load(VAL_LABELS_PATH)

# Load feature selector and label encoder
feature_selector = joblib.load(os.path.join(MODEL_DIR, "feature_selector.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# Preprocess validation features
X_val_selected = feature_selector.transform(val_features)
y_val_encoded = label_encoder.transform(val_labels)

# Load models
model_names = ["adaboost", "bagging", "extra_trees", "hist_gradient_boosting"]
models = {}

for name in model_names:
    print(f"Loading {name} model...")
    models[name] = joblib.load(os.path.join(MODEL_DIR, f"{name}_model.pkl"))

# Evaluate models
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_val_selected)

    accuracy = accuracy_score(y_val_encoded, y_pred)
    report = classification_report(y_val_encoded, y_pred, target_names=label_encoder.classes_)
    matrix = confusion_matrix(y_val_encoded, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)
