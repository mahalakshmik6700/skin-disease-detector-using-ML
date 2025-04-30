import os
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import pickle

def load_data(features_path, labels_path):
    features = joblib.load(features_path)
    labels = np.load(labels_path)
    return np.array(features), np.array(labels)

def split_and_save_data(features, labels, output_dir, test_size=0.2):
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=test_size, random_state=42)

    joblib.dump(X_train, os.path.join(output_dir, 'train_hog_features.pkl'))
    joblib.dump(X_val, os.path.join(output_dir, 'val_hog_features.pkl'))
    np.save(os.path.join(output_dir, 'train_labels.npy'), y_train)
    np.save(os.path.join(output_dir, 'val_labels.npy'), y_val)

if __name__ == "__main__":
    FEATURES_PATH = "data/hog_features.pkl"
    LABELS_PATH = "data/labels.npy"
    OUTPUT_DIR = "data"

    features, labels = load_data(FEATURES_PATH, LABELS_PATH)
    split_and_save_data(features, labels, OUTPUT_DIR)
