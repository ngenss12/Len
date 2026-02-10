import numpy as np

class FeatureNormalizer:
    """
    Normalize RT features to [0, 1] range as per Eq. (5)
    Should be fit on training data and applied to val/test
    """
    
    def __init__(self):
        self.min_vals = None
        self.max_vals = None
    
    def fit(self, rt_features):
        """Compute min and max from training data"""
        self.min_vals = np.min(rt_features, axis=0)
        self.max_vals = np.max(rt_features, axis=0)
    
    def transform(self, rt_features):
        """Normalize RT features to [0, 1]"""
        # Avoid division by zero
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (rt_features - self.min_vals) / range_vals
        return normalized
    
    def fit_transform(self, rt_features):
        """Fit and transform"""
        self.fit(rt_features)
        return self.transform(rt_features)

def normalize_dataset_features(train_dataset, val_dataset, test_dataset):
    """Normalize RT features across all datasets"""
    
    # Extract all features from training set
    train_features = []
    for i in range(len(train_dataset)):
        train_features.append(train_dataset[i]['rt'].numpy())
    train_features = np.array(train_features)
    
    # Fit normalizer on training data
    normalizer = FeatureNormalizer()
    normalizer.fit(train_features)
    
    # Apply to all datasets
    for dataset in [train_dataset, val_dataset, test_dataset]:
        for i in range(len(dataset)):
            original_features = dataset.dataset[i]['rt'].numpy()
            normalized = normalizer.transform(original_features.reshape(1, -1))
            dataset.dataset.data.at[i, 'rt_normalized'] = normalized[0]
    
    return normalizer
