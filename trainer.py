
from args import *
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

args = get_args()


def calculate_metrics(targets, outputs):
    targets = np.array(targets)
    outputs = np.array(outputs)
    predictions = np.argmax(outputs, axis=1)

    balanced_acc = balanced_accuracy_score(targets, predictions)

    try:
        if outputs.shape[1] == 2:
            roc_auc = roc_auc_score(targets, outputs[:, 1])
        else:
            roc_auc = roc_auc_score(
                targets,
                outputs,
                multi_class='ovr',
                average='macro',
                labels=np.arange(outputs.shape[1])
            )
    except ValueError:
        roc_auc = float('nan')

    try:
        if outputs.shape[1] == 2:
            avg_precision = average_precision_score(
                targets, outputs[:, 1]
            )
        else:
            binary_targets = label_binarize(
                targets, classes=np.arange(outputs.shape[1])
            )
            avg_precision = average_precision_score(
                binary_targets, outputs, average='macro'
            )
    except ValueError:
        avg_precision = float('nan')

    return balanced_acc, roc_auc, avg_precision


def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['xray'].to(device)
            labels = batch['target'].to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            all_targets.extend(labels.cpu().numpy())
            all_outputs.extend(probabilities.cpu().numpy())

    val_loss = running_loss / len(val_loader)

    balanced_acc, roc_auc, avg_precision = calculate_metrics(
        all_targets, all_outputs
    )

    return val_loss, balanced_acc, roc_auc, avg_precision


def save_plot(epochs, train_values, val_values, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, label='Training')
    plt.plot(epochs, val_values, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, filename))
    plt.close()


def train_model(model, train_loader, val_loader, fold=0):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    best_balanced_acc = -1.0

    train_losses, val_losses = [], []
    train_bal_accs, val_bal_accs = [], []
    train_roc_aucs, val_roc_aucs = [], []
    train_avg_precs, val_avg_precs = [], []

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        all_train_targets = []
        all_train_outputs = []

        for batch in train_loader:
            inputs = batch['xray'].to(device)
            labels = batch['target'].to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            all_train_targets.extend(labels.detach().cpu().numpy())
            all_train_outputs.extend(
                probabilities.detach().cpu().numpy()
            )

        train_loss = running_loss / len(train_loader)
        train_bal_acc, train_roc_auc, train_avg_prec = calculate_metrics(
            all_train_targets, all_train_outputs
        )

        val_loss, val_bal_acc, val_roc_auc, val_avg_prec = validate_model(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_bal_accs.append(train_bal_acc)
        val_bal_accs.append(val_bal_acc)
        train_roc_aucs.append(train_roc_auc)
        val_roc_aucs.append(val_roc_auc)
        train_avg_precs.append(train_avg_prec)
        val_avg_precs.append(val_avg_prec)

        print(f'Epoch {epoch + 1}/{args.epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Training Balanced Accuracy: {train_bal_acc:.4f}')
        print(f'Validation Balanced Accuracy: {val_bal_acc:.4f}')
        print(f'Training ROC-AUC: {train_roc_auc:.4f}')
        print(f'Validation ROC-AUC: {val_roc_auc:.4f}')
        print(f'Training Average Precision: {train_avg_prec:.4f}')
        print(f'Validation Average Precision: {val_avg_prec:.4f}')
        print('-' * 40)

        if np.isfinite(val_bal_acc) and val_bal_acc > best_balanced_acc:
            best_balanced_acc = val_bal_acc
            save_path = os.path.join(
                args.out_dir, f'best_model_fold_{fold}.pth'
            )
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved: {save_path}')

    epochs = range(1, args.epochs + 1)

    save_plot(
        epochs, train_losses, val_losses,
        'Training and Validation Loss', 'Loss',
        f'loss_plot_fold_{fold}.png'
    )

    save_plot(
        epochs, train_bal_accs, val_bal_accs,
        'Training and Validation Balanced Accuracy',
        'Balanced Accuracy',
        f'balanced_accuracy_plot_fold_{fold}.png'
    )

    save_plot(
        epochs, train_roc_aucs, val_roc_aucs,
        'Training and Validation ROC-AUC', 'ROC-AUC',
        f'roc_auc_plot_fold_{fold}.png'
    )

    save_plot(
        epochs, train_avg_precs, val_avg_precs,
        'Training and Validation Average Precision',
        'Average Precision',
        f'avg_precision_plot_fold_{fold}.png'
    )

    print(f'Fold {fold} training completed.')
    print(f'Best validation Balanced Accuracy: {best_balanced_acc:.4f}')
