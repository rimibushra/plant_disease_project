from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "dataset" / "PlantVillage"

if not dataset_path.exists():
    print("Dataset folder not found.")
else:
    class_folders = [folder for folder in dataset_path.iterdir() if folder.is_dir()]
    print("Dataset found.")
    print("Number of classes:", len(class_folders))

    print("\nFirst 10 class folders:")
    for folder in class_folders[:10]:
        print("-", folder.name)