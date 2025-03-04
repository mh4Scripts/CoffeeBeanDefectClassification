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

from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
    resnet152, ResNet152_Weights,
    vit_b_16, ViT_B_16_Weights
)

from datareader import get_data_loaders
from utils import check_set_gpu

def get_model(model_name, num_classes):
    """
    Create a model with pretrained weights and modified classifier layer
    
    Args:
        model_name (str): Name of the model to use
        num_classes (int): Number of output classes
        
    Returns:
        model: PyTorch model
    """
    if model_name == "efficientnet":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == "shufflenet":
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        model = shufflenet_v2_x0_5(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V1
        model = resnet152(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "vit":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

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
    """Main training function dengan modifikasi TensorBoard"""
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
    
    # Get data loaders
    train_loader, val_loader, classes = get_data_loaders(batch_size=batch_size)
    num_classes = len(classes)
    print(f"Classes: {classes}")
    print(f"Number of classes: {num_classes}")
    
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
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1
            })
        else:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save model jika validasi accuracy membaik
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
            best_val_acc = val_acc
            checkpoint_path = f"models/{run_name}_best.pth"
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
    
    # Print final summary
    training_time = time.time() - start_time
    print(f"Training complete in {training_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    fig = plot_training_history(train_losses, val_losses, train_accs, val_accs)
    if use_wandb:
        wandb.log({"training_history": wandb.Image("training_history.png")})
    else:
        writer.add_figure("Training History", fig)
    
    # Save final model
    final_model_path = f"models/{run_name}_final.pth"
    save_checkpoint(model, optimizer, epoch, val_acc, final_model_path)
    if use_wandb:
        wandb.save(final_model_path)
    
    if use_wandb:
        wandb.finish()
    else:
        writer.close()
    
    return model, best_val_acc

def main():
    """Main function dengan modifikasi argument"""
    parser = argparse.ArgumentParser(description="Train coffee classification models")
    parser.add_argument("--model", type=str, choices=["efficientnet", "shufflenet", "resnet152", "vit"], 
                        default="efficientnet", help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
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
