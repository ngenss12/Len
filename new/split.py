import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from collections import Counter

def stratified_train_val_split(dataset, labels, train_size=0.8, random_state=42):
    """
    Perform stratified train-validation split
    
    Args:
        dataset: OpenSARShipDataset instance
        train_size: Proportion of training data (0.8 = 80%)
        random_state: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset (Subset objects)
    """
    # Get all indices and labels
    indices = np.arange(len(dataset))
    
    # Perform stratified split
    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_size,
        stratify=labels,  # This ensures proportional class distribution
        random_state=random_state
    )
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Print split statistics
    print("\n" + "="*60)
    print("STRATIFIED SPLIT STATISTICS")
    print("="*60)
    
    train_labels = [dataset[i]['label'] for i in train_indices]
    val_labels = [dataset[i]['label'] for i in val_indices]
    
    class_names = ['Bulk Carrier', 'Container Ship', 'Fishing', 'Tanker']
    
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"Val samples: {len(val_dataset)} ({len(val_dataset)/len(dataset)*100:.1f}%)")
    
    print("\nTrain set distribution:")
    train_counter = Counter(train_labels)
    for i in range(4):
        count = train_counter[i]
        pct = count / len(train_labels) * 100
        print(f"  {class_names[i]}: {count} ({pct:.1f}%)")
    
    print("\nValidation set distribution:")
    val_counter = Counter(val_labels)
    for i in range(4):
        count = val_counter[i]
        pct = count / len(val_labels) * 100
        print(f"  {class_names[i]}: {count} ({pct:.1f}%)")
    
    # Verify proportions are similar
    print("\nClass proportion comparison:")
    print(f"{'Class':<20} {'Train %':<12} {'Val %':<12} {'Difference':<12}")
    print("-" * 56)
    for i in range(4):
        train_pct = train_counter[i] / len(train_labels) * 100
        val_pct = val_counter[i] / len(val_labels) * 100
        diff = abs(train_pct - val_pct)
        print(f"{class_names[i]:<20} {train_pct:<12.2f} {val_pct:<12.2f} {diff:<12.2f}")
    
    return train_dataset, val_dataset