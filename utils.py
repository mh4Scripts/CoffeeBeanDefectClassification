import os
import time
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

def check_set_gpu(override=None):
    if override == None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using MPS: {torch.backends.mps.is_available()}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU: {torch.device('cpu')}")
    else:
        device = torch.device(override)
    return device

def save_checkpoint(model, optimizer, epoch, val_acc, filename):
    """Save model checkpoint."""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }
    torch.save(state, filename)

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation history."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def log_metrics_to_wandb(fold, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, val_f1, lr, epoch, use_wandb):
    """Log metrics to wandb if enabled."""
    if use_wandb:
        wandb.log({
            "fold": fold + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "learning_rate": lr,
            "epoch": epoch + 1
        })

def log_metrics_to_tensorboard(writer, fold, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, val_f1, lr, epoch):
    """Log metrics to TensorBoard."""
    writer.add_scalar(f'Fold {fold + 1}/Loss/train', train_loss, epoch)
    writer.add_scalar(f'Fold {fold + 1}/Loss/val', val_loss, epoch)
    writer.add_scalar(f'Fold {fold + 1}/Accuracy/train', train_acc, epoch)
    writer.add_scalar(f'Fold {fold + 1}/Accuracy/val', val_acc, epoch)
    writer.add_scalar(f'Fold {fold + 1}/Precision/val', val_precision, epoch)
    writer.add_scalar(f'Fold {fold + 1}/Recall/val', val_recall, epoch)
    writer.add_scalar(f'Fold {fold + 1}/F1 Score/val', val_f1, epoch)
    writer.add_scalar(f'Fold {fold + 1}/Learning Rate', lr, epoch)
