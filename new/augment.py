import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import hflip, vflip, rotate
from torchvision.transforms.v2 import GaussianNoise

from itertools import combinations

class ShipAugmentation:
    """
    Data augmentation with all possible combinations
    - Horizontal flip
    - Vertical flip
    - 90° rotation
    - 270° rotation
    - Gaussian noise
    """
    
    def __init__(self, noise_std=1e-3):
        self.noise_std = noise_std
        # Define individual augmentation functions
        self.augmentations = {
            'h_flip': lambda img: np.flip(img, axis=1).copy(),  # Changed axis
            'v_flip': lambda img: np.flip(img, axis=0).copy(),  # Changed axis
            'rot_90': lambda img: np.rot90(img, k=1, axes=(0, 1)).copy(),  # Changed axes
            'rot_270': lambda img: np.rot90(img, k=3, axes=(0, 1)).copy(),  # Changed axes
            'noise': lambda img: img + np.random.normal(0, self.noise_std, img.shape)
        }
    def apply_augmentations(self, image, aug_list):
        """Apply a list of augmentations in sequence"""
        for aug_name in aug_list:
            image = self.augmentations[aug_name](image)
        return image
    
    def get_all_combinations(self):
        """Generate all possible combinations of augmentations"""
        aug_names = list(self.augmentations.keys())
        all_combos = []
        
        # Generate all combinations of different lengths
        for r in range(1, len(aug_names) + 1):
            for combo in combinations(aug_names, r):
                all_combos.append(list(combo))
        
        return all_combos
    
    def __call__(self, image, aug_combo=None):
        """
        Apply augmentation combination
        
        Args:
            image: Input image
            aug_combo: List of augmentation names, or None for original
        """
        if aug_combo is None or len(aug_combo) == 0:
            return image.copy()
        return self.apply_augmentations(image, aug_combo)
    
    def augment_all(self, image):
        """Generate all possible augmented versions of an image"""
        all_combos = self.get_all_combinations()
        augmented_images = []
        
        for combo in all_combos:
            aug_img = self(image, combo)
            augmented_images.append((combo, aug_img))
        
        return augmented_images


class AugmentedDataset(Dataset):
    """Wrapper dataset that includes augmented samples"""
    
    def __init__(self, original_dataset, augmented_samples):
        self.original_dataset = original_dataset
        self.augmented_samples = augmented_samples
        self.total_len = len(original_dataset) + len(augmented_samples)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            aug_idx = idx - len(self.original_dataset)
            return self.augmented_samples[aug_idx]