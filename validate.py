import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Lists to store predictions and labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions and labels for metrics calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    val_precision = precision_score(all_labels, all_preds, average='weighted')  # Weighted for multi-class
    val_recall = recall_score(all_labels, all_preds, average='weighted')        # Weighted for multi-class
    val_f1 = f1_score(all_labels, all_preds, average='weighted')                # Weighted for multi-class
    
    return val_loss, val_acc, val_precision, val_recall, val_f1
