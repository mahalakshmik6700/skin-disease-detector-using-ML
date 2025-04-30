import os
import numpy as np
import joblib
import pickle
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Paths (relative to project root)
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load training features and labels
train_features = pickle.load(open("train_hog_features.pkl", "rb"))
train_labels = np.load("train_labels.npy")

# Label encode target variable
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(train_features, train_labels_encoded)

# Define models
models = {
    'adaboost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'bagging': BaggingClassifier(n_estimators=100, random_state=42),
    'extra_trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'hist_gradient_boosting': HistGradientBoostingClassifier(random_state=42)
}

# Train and save models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_selected, train_labels_encoded)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_model.pkl"))
    print(f"{name} saved to {MODEL_DIR}")

# Save feature selector and label encoder
joblib.dump(selector, os.path.join(MODEL_DIR, "feature_selector.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("All models and preprocessing objects saved.")
