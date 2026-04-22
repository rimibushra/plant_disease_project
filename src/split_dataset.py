from pathlib import Path
import shutil
import random

random.seed(42)

# Find the main project folder
project_root = Path(__file__).resolve().parent.parent

# Original dataset folder
source_dir = project_root / "dataset" / "PlantVillage"

# New split folders
output_dir = project_root / "dataset" / "split_data"
train_dir = output_dir / "train"
val_dir = output_dir / "val"
test_dir = output_dir / "test"

# Create the split folders
for folder in [train_dir, val_dir, test_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# Go through each class folder
for class_folder in source_dir.iterdir():
    if class_folder.is_dir():
        images = [img for img in class_folder.iterdir() if img.is_file()]
        random.shuffle(images)

        total = len(images)
        train_end = int(total * 0.64)
        val_end = int(total * 0.80)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        print(f"\n{class_folder.name}")
        print(f"  Total: {total}")
        print(f"  Train: {len(train_images)}")
        print(f"  Val:   {len(val_images)}")
        print(f"  Test:  {len(test_images)}")

        for split_name, split_images in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images)
        ]:
            class_output = output_dir / split_name / class_folder.name
            class_output.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                destination = class_output / img_path.name
                shutil.copy2(img_path, destination)

print("\nDataset split completed successfully.")