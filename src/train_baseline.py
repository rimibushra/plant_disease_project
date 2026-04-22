from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Project paths
# -----------------------------
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "dataset" / "split_data"
models_dir = project_root / "models"
outputs_dir = project_root / "outputs"

train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"

models_dir.mkdir(parents=True, exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 10

# -----------------------------
# Load datasets
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Save class names BEFORE prefetch
class_names = train_ds.class_names
num_classes = len(class_names)

print("\nDataset loaded successfully.")
print("Number of classes:", num_classes)

print("\nClass names (IMPORTANT):")
for i, name in enumerate(class_names):
    print(i, ":", name)

# Save class names to a text file so predict.py can read them
class_file = outputs_dir / "class_names.txt"
with open(class_file, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"\nClass names saved to: {class_file}")

# -----------------------------
# Optimize dataset pipeline
# -----------------------------
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# -----------------------------
# Build baseline CNN model
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    data_augmentation,
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation="softmax")
])

# -----------------------------
# Compile model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Show model summary
# -----------------------------
model.summary()

# -----------------------------
# Callbacks
# -----------------------------
checkpoint_path = models_dir / "baseline_best.keras"

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True
)

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint]
)

# -----------------------------
# Evaluate on test set
# -----------------------------
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# -----------------------------
# Save final model
# -----------------------------
final_model_path = models_dir / "baseline_final.keras"
model.save(final_model_path)

print(f"\nBest model saved to: {checkpoint_path}")
print(f"Final model saved to: {final_model_path}")

# -----------------------------
# Plot training history
# -----------------------------
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(1, len(accuracy) + 1)

# Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, accuracy, label="Training Accuracy")
plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")
plt.title("Baseline CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(outputs_dir / "baseline_accuracy.png")
plt.show()

# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.title("Baseline CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(outputs_dir / "baseline_loss.png")
plt.show()