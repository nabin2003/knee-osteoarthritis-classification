
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from args import get_args
from dataset import knee_xray_dataset
from model import MyModel
from trainer import train_model

args = get_args()

for fold in range(5):
    print(f"\nStarting fold {fold}")

    train_path = os.path.join(
        args.csv_dir, f'fold_{fold}_train.csv'
    )
    val_path = os.path.join(
        args.csv_dir, f'fold_{fold}_val.csv'
    )

    train_set = pd.read_csv(train_path)
    val_set = pd.read_csv(val_path)

    train_dataset = knee_xray_dataset(train_set)
    val_dataset = knee_xray_dataset(val_set)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False
    )

    model = MyModel(backbone=args.backbone)

    train_model(
        model,
        train_loader,
        val_loader,
        fold=fold
    )

    print(f"Completed fold {fold}")
