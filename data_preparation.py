
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

dataset_path = "data/archive/OSAIL_KL_Dataset/Labeled"
output_dir = "data"

image_data = []
labels = ["0", "1", "2", "3", "4"]

for label in labels:
    label_path = os.path.join(dataset_path, label)

    if not os.path.isdir(label_path):
        raise FileNotFoundError(f"Label folder not found: {label_path}")

    for image_file in os.listdir(label_path):
        img_path = os.path.join(label_path, image_file)

        if os.path.isfile(img_path):
            image_data.append({
                "Path": img_path,
                "Name": image_file,
                "KL": int(label)
            })

df = pd.DataFrame(image_data)

if df.empty:
    raise ValueError("No images found. Check the dataset path.")

print("Total images:", len(df))
print("\nImages per class:")
print(df["KL"].value_counts().sort_index())

os.makedirs(output_dir, exist_ok=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(
    skf.split(df["Path"], df["KL"])
):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_file = os.path.join(output_dir, f"fold_{fold}_train.csv")
    val_file = os.path.join(output_dir, f"fold_{fold}_val.csv")

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    print(f"\nFold {fold}")
    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))
    print("Saved:", train_file)
    print("Saved:", val_file)

print("\nAll five folds have been prepared.")
