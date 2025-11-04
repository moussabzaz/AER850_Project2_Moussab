# Step 5 
# Moussab Arfat Zaz - 501082410
# AER850 Project 2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array, load_img

# Test image locations
sample_images = {
    "crack": "C:/Users/mouss/OneDrive/Desktop/Data/Test/crack/test_crack.jpg",
    "missing_head": "C:/Users/mouss/OneDrive/Desktop/Data/Test/missing-head/test_missinghead.jpg",
    "paint_off": "C:/Users/mouss/OneDrive/Desktop/Data/Test/paint-off/test_paintoff.jpg"
}

# Load pretrained model
model_path = "model.keras"
model = tf.keras.models.load_model(model_path)

# Label names (must match model training order)
labels = ["crack", "missing-head", "paint-off"]

def predict_image(file_path):
    """Load an image, preprocess it, and generate a prediction."""
    # Load and preprocess
    image = load_img(file_path, target_size=(500, 500))
    image_arr = img_to_array(image).astype("float32") / 255.0
    image_arr = np.expand_dims(image_arr, 0)

    # Model inference
    result = model.predict(image_arr)
    class_idx = np.argmax(result)
    class_label = labels[class_idx]
    confidence = float(result[0, class_idx])

    return class_label, confidence, image

# Evaluate and visualize each image
for true_label, path in sample_images.items():
    pred_label, conf_score, original_img = predict_image(path)

    plt.figure(figsize=(5, 5))
    plt.imshow(original_img)
    plt.title(f"Actual: {true_label.replace('_', ' ').title()} | "
              f"Predicted: {pred_label.title()} ({conf_score*100:.2f}%)")
    plt.axis("off")
    plt.show()
