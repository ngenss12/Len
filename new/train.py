import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

def train_model(model, train_loader, val_loader, device, epochs=70, initial_lr=0.001):
    """
    Training setup as per Section 3.4
    """
    
    # Loss function
    criterion = MacroF1Loss(num_classes=4)
    
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Learning rate scheduler (reduce when plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.01, patience=5, 
        min_lr=0.00001
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
        'val_recall': []
    }
    
    best_val_f1 = 0.0  # Track F1 instead of accuracy
    best_model_state = None
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            rt = batch['rt'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, rt)
            loss = criterion(outputs, labels)
            
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
                images = batch['image'].to(device)
                rt = batch['rt'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, rt)
                loss = criterion(outputs, labels)
                
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
        
        print(f'\nTraining   - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.2f}%')
        print(f'Validation - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.2f}%')
        print(f'           - Balanced Acc: {val_balanced_acc:.2f}% | Precision: {val_precision:.2f}% | Recall: {val_recall:.2f}%')
        
        # Save best model based on F1-score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f'New best model! Val F1: {val_f1:.2f}%')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, history