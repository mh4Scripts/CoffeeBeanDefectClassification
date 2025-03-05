import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import wandb

from data_reader import get_kfold_data_loaders, CoffeeDataset
from utils import check_set_gpu
from get_model import get_model

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i+1) % 10 == 0:  # Print every 10 mini-batches
            print(f'Epoch: {epoch+1}, Batch: {i+1}/{len(train_loader)}, '
                  f'Loss: {running_loss/(i+1):.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def save_checkpoint(model, optimizer, epoch, val_acc, filename):
    """Save model checkpoint"""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }
    torch.save(state, filename)

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation history"""
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

def train(model_name, batch_size=32, lr=0.00001, epochs=200, patience=5, device_override=None, use_wandb=True):
    """Main training function dengan modifikasi untuk K-Fold"""
    # Generate timestamp for run name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{model_name}_{timestamp}"
    
    # Initialize wandb if enabled
    writer = None
    if use_wandb:
        wandb.init(
            project="coffee-classification",
            name=run_name,
            config={
                "model": model_name,
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": epochs,
                "patience": patience
            }
        )
    else:
        log_dir = os.path.join("runs", run_name)
        writer = SummaryWriter(log_dir=log_dir)
    
    # Set device using the utility function
    device = check_set_gpu(device_override)
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Get full dataset
    full_dataset = CoffeeDataset()
    
    # Get K-Fold data loaders
    fold_loaders = get_kfold_data_loaders(full_dataset, batch_size=batch_size)
    num_classes = len(full_dataset.classes)
    print(f"Classes: {full_dataset.classes}")
    print(f"Number of classes: {num_classes}")
    
    # Store results for each fold
    fold_results = []
    
    # Iterate through each fold
    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\nStarting Fold {fold + 1}/{len(fold_loaders)}")

        # Get model
        model = get_model(model_name, num_classes)
        model = model.to(device)

        # Log model architecture ke TensorBoard
        if use_wandb:
            wandb.watch(model, log="all")
        else:
            example_input = next(iter(train_loader))[0][0].unsqueeze(0).to(device)
            writer.add_graph(model, example_input)

        # Loss function dan optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        # Inisialisasi variabel
        best_val_acc = 0.0
        epochs_no_improve = 0
        early_stop = False

        # History untuk plotting
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        start_time = time.time()
        for epoch in range(epochs):
            if early_stop:
                print("Early stopping triggered!")
                break

            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 20)

            # Train and validate - pass device to the functions
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # Update learning rate
            scheduler.step(val_loss)

            # Save history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Log metrics to wandb if enabled
            if use_wandb:
                wandb.log({
                    "fold": fold + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1
                })
            else:
                writer.add_scalar(f'Fold {fold+1}/Loss/train', train_loss, epoch)
                writer.add_scalar(f'Fold {fold+1}/Loss/val', val_loss, epoch)
                writer.add_scalar(f'Fold {fold+1}/Accuracy/train', train_acc, epoch)
                writer.add_scalar(f'Fold {fold+1}/Accuracy/val', val_acc, epoch)
                writer.add_scalar(f'Fold {fold+1}/Learning Rate', optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save model jika validasi accuracy membaik
            if val_acc > best_val_acc:
                print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
                best_val_acc = val_acc
                checkpoint_path = f"models/{run_name}_fold{fold+1}_best.pth"
                save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)

                # Save best model to wandb if enabled
                if use_wandb:
                    wandb.save(checkpoint_path)

                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs")
    
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                early_stop = True

        # Print fold summary
        training_time = time.time() - start_time
        print(f"Fold {fold + 1} complete in {training_time:.2f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")

        # Save fold results
        fold_results.append({
            "fold": fold + 1,
            "best_val_acc": best_val_acc,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs
        })

        # Plot training history for this fold
        fig = plot_training_history(train_losses, val_losses, train_accs, val_accs)
        if use_wandb:
            wandb.log({f"Fold {fold+1}/training_history": wandb.Image("training_history.png")})
        else:
            writer.add_figure(f"Fold {fold+1}/Training History", fig)

        # Save final model for this fold
        final_model_path = f"models/{run_name}_fold{fold+1}_final.pth"
        save_checkpoint(model, optimizer, epoch, val_acc, final_model_path)
        if use_wandb:
            wandb.save(final_model_path)

    # Calculate average validation accuracy across all folds
    avg_val_acc = np.mean([result["best_val_acc"] for result in fold_results])
    print(f"\nAverage validation accuracy across all folds: {avg_val_acc:.2f}%")
    
    if use_wandb:
        wandb.log({"average_val_acc": avg_val_acc})
        wandb.finish()
    else:
        writer.close()
    
    return fold_results, avg_val_acc

def main():
    """Main function dengan modifikasi argument"""
    parser = argparse.ArgumentParser(description="Train coffee classification models")
    parser.add_argument("--model", type=str, choices=["efficientnet", "resnet50", "mobilenetv3", "densenet121", "vit", "convnext", "regnet"], 
                        default="efficientnet", help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], 
                        default=None, help="Device to use (overrides automatic detection)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging and use tensorboard")
    
    args = parser.parse_args()
    
    print(f"Training with {args.model} model")
    train(
        args.model,
        args.batch_size,
        args.lr,
        args.epochs,
        args.patience,
        args.device,
        not args.no_wandb
    )

if __name__ == "__main__":
    main()
