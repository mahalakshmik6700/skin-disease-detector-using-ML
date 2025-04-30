import cv2
import numpy as np
from skimage.feature import hog
from PIL import Image
import os

def extract_hog_features(image_path, resize_shape=(128, 128), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Load an image, convert to grayscale, resize, and extract HOG features.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(resize_shape)
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        features = hog(gray,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm='L2-Hys')
        return features
    except Exception as e:
        raise RuntimeError(f"Failed to extract HOG features from {image_path}: {e}")

if __name__ == "__main__":
    # Example usage (for testing)
    test_path = "data/sample_image.jpg"
    features = extract_hog_features(test_path)
    print(f"Extracted {len(features)} HOG features.")
