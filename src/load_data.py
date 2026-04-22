from pathlib import Path
import tensorflow as tf

# -----------------------------
# Project paths
# -----------------------------
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "dataset" / "split_data"

train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

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

# -----------------------------
# Save class names BEFORE prefetch
# -----------------------------
class_names = train_ds.class_names

# -----------------------------
# Optimize dataset pipeline
# -----------------------------
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Show class names
# -----------------------------
print("\nDataset loaded successfully.")
print("Number of classes:", len(class_names))

print("\nClass names:")
for name in class_names:
    print("-", name)

# -----------------------------
# Show one batch shape
# -----------------------------
for images, labels in train_ds.take(1):
    print("\nOne batch of images shape:", images.shape)
    print("One batch of labels shape:", labels.shape)