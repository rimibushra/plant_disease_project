from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Project paths
# -----------------------------
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "baseline_best.keras"
class_file = project_root / "outputs" / "class_names.txt"
image_path = project_root / "sample_images" / "test_leaf.jpg"

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = (160, 160)

# -----------------------------
# Check files exist
# -----------------------------
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not class_file.exists():
    raise FileNotFoundError(f"Class names file not found: {class_file}")

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# -----------------------------
# Load class names
# -----------------------------
with open(class_file, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

print("Loaded", len(class_names), "class names.")

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(model_path)

# -----------------------------
# Load and prepare image
# -----------------------------
img = Image.open(image_path).convert("RGB")
img_resized = img.resize(IMG_SIZE)

img_array = np.array(img_resized, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 160, 160, 3)

# -----------------------------
# Predict
# -----------------------------
predictions = model.predict(img_array)
predicted_index = int(np.argmax(predictions[0]))
predicted_class = class_names[predicted_index]
confidence = float(np.max(predictions[0]))

# -----------------------------
# Show result in terminal
# -----------------------------
print("\nPredicted class:", predicted_class)
print("Confidence:", round(confidence * 100, 2), "%")

# -----------------------------
# Show image with prediction
# -----------------------------
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence * 100:.2f}%")
plt.axis("off")
plt.tight_layout()
plt.show()