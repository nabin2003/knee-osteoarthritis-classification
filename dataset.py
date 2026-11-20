
import cv2
import torch
from torch.utils.data import Dataset


def read_img(path_file):
    xray = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)

    if xray is None:
        raise FileNotFoundError(f"Could not read image: {path_file}")

    xray = cv2.resize(xray, (224, 224))
    return xray


class knee_xray_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset.reset_index(drop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset["Path"].iloc[idx]
        name = self.dataset["Name"].iloc[idx]
        target = self.dataset["KL"].iloc[idx]

        img = read_img(img_path)

        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0).repeat(3, 1, 1) / 255.0

        return {
            "xray": img,
            "name": name,
            "target": int(target)
        }
