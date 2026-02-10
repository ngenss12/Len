from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             cohen_kappa_score, roc_auc_score, confusion_matrix,
                             roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F

def evaluate_model(model, test_loader, device):
    """
    Compute all metrics from Section 3.5
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            rt = batch['rt'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, rt)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 1. Accuracy (Eq. 11)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 2. Precision, Recall, F-score (Eq. 12, 13, 14)
    precision, recall, fscore, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Weighted averages
    precision_weighted = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )[0]
    recall_weighted = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )[1]
    fscore_weighted = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )[2]
    
    # 3. Kappa score (Eq. 15)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # 4. ROC-AUC
    auc_scores = []
    for i in range(4):
        # One-hot encode labels for class i
        labels_binary = (all_labels == i).astype(int)
        auc = roc_auc_score(labels_binary, all_probs[:, i])
        auc_scores.append(auc)
    
    # Micro-average AUC
    labels_onehot = np.eye(4)[all_labels]
    auc_micro = roc_auc_score(labels_onehot, all_probs, average='micro')
    
    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy * 100,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'fscore_per_class': fscore,
        'support': support,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'fscore_weighted': fscore_weighted,
        'kappa': kappa,
        'auc_per_class': auc_scores,
        'auc_micro': auc_micro,
        'confusion_matrix': cm
    }
    
    return results

def print_results(results):
    """Print evaluation results in paper format"""
    class_names = ['Bulk Carrier', 'Container Ship', 'Fishing', 'Tanker']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Precision: {results['precision_weighted']:.4f}")
    print(f"  Recall:    {results['recall_weighted']:.4f}")
    print(f"  F-score:   {results['fscore_weighted']:.4f}")
    print(f"  Kappa:     {results['kappa']:.4f}")
    print(f"  AUC (micro): {results['auc_micro']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F-score':<12} {'AUC':<12}")
    print("-" * 68)
    for i, name in enumerate(class_names):
        print(f"{name:<20} {results['precision_per_class'][i]:<12.4f} "
              f"{results['recall_per_class'][i]:<12.4f} "
              f"{results['fscore_per_class'][i]:<12.4f} "
              f"{results['auc_per_class'][i]:<12.4f}")
    
    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])

def plot_confusion_matrix(cm, output_dir="", class_names=['Bulk Carrier', 'Container', 'Fishing', 'Tanker']):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(all_labels, all_probs, class_names=['Bulk Carrier', 'Container', 'Fishing', 'Tanker']):
    """Plot ROC curves for all classes"""
    plt.figure(figsize=(12, 8))
    
    for i in range(4):
        labels_binary = (all_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(labels_binary, all_probs[:, i])
        auc = roc_auc_score(labels_binary, all_probs[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
