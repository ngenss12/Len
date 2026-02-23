import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import time
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    balanced_accuracy_score,
)

class MacroF1Loss(nn.Module):
    """
    Differentiable Macro F1 Loss for multi-class classification
    Directly optimizes the F1-score metric
    """
    def __init__(self, num_classes, epsilon=1e-7):
        super(MacroF1Loss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Model logits [batch_size, num_classes]
            y_true: Ground truth labels [batch_size] (class indices)
        """
        # Convert logits to probabilities
        y_pred = F.softmax(y_pred, dim=1)
        
        # Convert labels to one-hot encoding
        y_true_one_hot = F.one_hot(y_true, num_classes=self.num_classes).float()
        
        # Calculate TP, FP, FN for each class
        tp = torch.sum(y_true_one_hot * y_pred, dim=0)
        fp = torch.sum((1 - y_true_one_hot) * y_pred, dim=0)
        fn = torch.sum(y_true_one_hot * (1 - y_pred), dim=0)
        
        # Calculate precision and recall for each class
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        # Calculate F1 for each class
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        
        # Handle NaN values (when both precision and recall are 0)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        
        # Macro F1: average across all classes
        macro_f1 = torch.mean(f1)
        
        # Return 1 - F1 (we minimize loss, so we want to minimize 1-F1)
        return 1 - macro_f1

def _extract_labels_from_loader(loader):
    ds = loader.dataset
    # FinalDataset has a dataframe with `label` column.
    if hasattr(ds, "data") and "label" in getattr(ds, "data", {}).columns:
        return ds.data["label"].astype(int).to_numpy()
    # Fallback: read labels from batches.
    all_labels = []
    for batch in loader:
        all_labels.extend(batch["label"].cpu().numpy().tolist())
    return np.array(all_labels, dtype=np.int64)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=70,
    initial_lr=0.001,
    early_stop_patience=0,
    f1_loss_weight=0.1,
    previous_best_val_f1=0.0,
):
    """
    Training setup as per Section 3.4
    """
    
    device = torch.device(device)
    use_cuda = device.type == "cuda"
    model = model.to(device)

    # Conservative setup: CE-dominant objective to avoid unstable class bias.
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.02)
    criterion_f1 = MacroF1Loss(num_classes=4)
    
    # Adam optimizer
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # Learning rate scheduler (reduce when plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4,
        min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_balanced_acc': [],
        'val_precision': [],
        'val_recall': [],
        'epoch_time_sec': []
    }
    
    best_val_f1 = float(previous_best_val_f1)  
    best_model_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device, non_blocking=use_cuda)
            rt = batch['rt'].to(device, non_blocking=use_cuda)
            labels = batch['label'].to(device, non_blocking=use_cuda)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, rt)
            loss_ce = criterion_ce(outputs, labels)
            if f1_loss_weight > 0:
                loss_f1 = criterion_f1(outputs, labels)
                loss = (1.0 - f1_loss_weight) * loss_ce + f1_loss_weight * loss_f1
            else:
                loss = loss_ce
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)

            # Store predictions and labels for metrics
            train_preds.append(predicted)
            train_labels.append(labels)
            
            if (batch_idx + 1) % 20 == 0:
                temp_preds = torch.cat(train_preds).cpu().numpy()
                temp_labels = torch.cat(train_labels).cpu().numpy()
                batch_acc = 100. * (temp_preds == temp_labels).sum() / len(temp_labels)
                print(f'Batch {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {batch_acc:.2f}%')
        
        train_preds = torch.cat(train_preds).cpu().numpy()
        train_labels = torch.cat(train_labels).cpu().numpy()
        train_loss /= len(train_loader)
        train_acc = 100. * (train_preds == train_labels).sum() / len(train_labels)
        train_f1 = f1_score(train_labels, train_preds, average='macro') * 100
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=use_cuda)
                rt = batch['rt'].to(device, non_blocking=use_cuda)
                labels = batch['label'].to(device, non_blocking=use_cuda)
                
                outputs = model(images, rt)
                loss_ce = criterion_ce(outputs, labels)
                if f1_loss_weight > 0:
                    loss_f1 = criterion_f1(outputs, labels)
                    loss = (1.0 - f1_loss_weight) * loss_ce + f1_loss_weight * loss_f1
                else:
                    loss = loss_ce
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = 100. * (np.array(val_preds) == np.array(val_labels)).sum() / len(val_labels)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0) * 100
        val_balanced_acc = balanced_accuracy_score(val_labels, val_preds) * 100
        val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0) * 100
        val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0) * 100
        
        # Update scheduler based on F1-score
        scheduler.step(val_f1)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_balanced_acc'].append(val_balanced_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        epoch_time = time.time() - epoch_start
        history['epoch_time_sec'].append(epoch_time)
        
        print(f'\nTraining   - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.2f}%')
        print(f'Validation - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.2f}%')
        print(f'           - Balanced Acc: {val_balanced_acc:.2f}% | Precision: {val_precision:.2f}% | Recall: {val_recall:.2f}%')
        print(f'           - Epoch time: {epoch_time:.2f}s')
        
        # Save best model based on F1-score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            print(f'New best model! Val F1: {val_f1:.2f}% (prev best: {previous_best_val_f1:.2f}%)')
        else:
            no_improve += 1
            if early_stop_patience and early_stop_patience > 0:
                print(f'No improvement count: {no_improve}/{early_stop_patience}')

        if early_stop_patience and early_stop_patience > 0 and no_improve >= early_stop_patience:
            print(f"Early stopping triggered (no val_f1 improvement for {early_stop_patience} epochs).")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f'Final selected best Val F1 thresholded by previous best: {best_val_f1:.2f}%')
    
    return model, history
