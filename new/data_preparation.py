
# Configuration
import random
import pandas as pd
import numpy as np
import torch
import tifffile as tiff
import joblib
import os
from augment import AugmentedDataset, ShipAugmentation
from dataload import OpenSARShipDataset
from split import stratified_train_val_split
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize

random.seed(42)

def _to_numpy_hwc(image):
    """Convert image tensor/array to numpy HWC for numpy augment ops + tifffile write."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 2, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    return image.astype(np.float32, copy=False)

def _to_float_list(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).tolist()

def create_balanced_dataset(dataset, target=-1, is_augmented=False):
    """
    Balance dataset by augmentation
    Target: 1600 training, 400 validation per class (except fishing: 1520/380)
    
    Args:
        dataset: Original dataset
        target_counts: Target number of samples per class
        augmentation: ShipAugmentation instance (if None, will create one)
    """
    augmentation = ShipAugmentation()
    aug_names = list(augmentation.augmentations.keys())
    
    # Get class distribution
    labels = []
    for i in range(len(dataset)):
        lb = dataset[i]['label']
        # Subset can return torch scalar tensor; normalize to plain int
        if hasattr(lb, "item"):
            lb = lb.item()
        labels.append(int(lb))
    class_indices = {i: [] for i in range(4)}
    
    for idx, label in enumerate(labels):
        class_indices[int(label)].append(idx)
    
    # Store augmented samples
    final = []
    for class_id, indices in class_indices.items():
        current_count = len(indices)
        
        if current_count < target:
            if not is_augmented:
                # If not augmented dataset, just copy existing samples
                for idx in indices:
                    row = dataset[idx]
                    final.append({
                        'label': int(row['label'].item() if hasattr(row['label'], "item") else row['label']),
                        'rt': _to_float_list(row['rt']),
                        'img_path': row['img_path'],
                    })
                continue

            # Keep originals first
            for i in indices:
                row = dataset[i]
                final.append({
                    'img_path': row['img_path'],
                    'label': int(row['label'].item() if hasattr(row['label'], "item") else row['label']),
                    'rt': _to_float_list(row['rt']),
                })

            # Conservative augmentation: one random transform per synthetic sample.
            n_augment = target - current_count
            ptr = 0
            while n_augment > 0:
                i = indices[ptr % current_count]
                original_sample = dataset[i]
                original_img = _to_numpy_hwc(original_sample['image'])

                aug_name = random.choice(aug_names)
                augmented_img = augmentation.apply_augmentations(original_img, [aug_name])
                stem = Path(original_sample["img_path"]).stem
                suffix = Path(original_sample["img_path"]).suffix
                filename = f"augment/{aug_name}_{ptr}_{stem}{suffix}"
                try:
                    tiff.imwrite(filename, np.asarray(augmented_img, dtype=np.float32))
                    final.append({
                        'label': int(original_sample['label'].item() if hasattr(original_sample['label'], "item") else original_sample['label']),
                        'rt': _to_float_list(original_sample['rt']),
                        'img_path': filename,
                    })
                    n_augment -= 1
                except Exception as e:
                    print(f"Error saving image {filename}: {e}")
                ptr += 1
        else:
            if target == -1:
                target = current_count
            else:
                target = min(target, current_count)
            
            undersampled_indices = random.sample(indices, target)
            for idx in undersampled_indices:
                row = dataset[idx]
                final.append({
                    'label': int(row['label'].item() if hasattr(row['label'], "item") else row['label']),
                    'rt': _to_float_list(row['rt']),
                    'img_path': row['img_path'],
                })

    return final

def calculate_global_stats(dataset):
    pixel_sum = 0
    pixel_sq_sum = 0
    pixel_count = 0
    for elmt in dataset:
        img = tiff.imread(elmt['img_path']).astype(np.float32)
        pixel_sum += img.sum()
        pixel_sq_sum += (img ** 2).sum()
        pixel_count += img.size

    global_mean = pixel_sum / pixel_count
    variance = (pixel_sq_sum / pixel_count) - (global_mean ** 2)
    global_std = np.sqrt(variance)

    print(f"Training set statistics: mean={global_mean:.6f}, std={global_std:.6f}")
    return global_mean, global_std


# Preprocess dataset
os.makedirs("resized_new", exist_ok=True)
os.makedirs("augment", exist_ok=True)
os.makedirs("final/train", exist_ok=True)
os.makedirs("final/val", exist_ok=True)
os.makedirs("final/test", exist_ok=True)
base_dir = Path(__file__).resolve().parent
patch_cal_dir = base_dir / "PATCH_CAL"
if not patch_cal_dir.exists():
    alt_patch_cal_dir = base_dir / "data gambar" / "PATCH_CAL"
    if alt_patch_cal_dir.exists():
        patch_cal_dir = alt_patch_cal_dir
    else:
        raise FileNotFoundError(
            f"PATCH_CAL folder not found. Checked: {patch_cal_dir} and {alt_patch_cal_dir}"
        )

for file_path in patch_cal_dir.iterdir():
    img = tiff.imread(file_path).astype(np.float32)
    img_resized = resize(img, (64, 64), order=3, mode='reflect', preserve_range=True)
    if (img_resized.shape != (64, 64, 2)):
        print(img.shape)
        print(f"Error resizing {file_path.name}: got shape {img_resized.shape}")
    tiff.imwrite("resized_new/" + file_path.name, img_resized.astype(np.float32))


print("\nLoading datasets...")
dataset = OpenSARShipDataset(root_dir=str(base_dir))

# 2. Stratified train-val split
temp_subset, val_subset = stratified_train_val_split(dataset, labels=dataset.get_labels(), train_size=0.8)
train_subset, test_subset = stratified_train_val_split(temp_subset, labels=[dataset[i]['label'] for i in temp_subset.indices], train_size=0.8)

# # 3. Balance datasets
# print("\nBuilding train and validation datasets...")
train_data = create_balanced_dataset(train_subset, 1600, is_augmented=True)
val_data = create_balanced_dataset(val_subset, is_augmented=False)
test_data = create_balanced_dataset(test_subset, is_augmented=False)

global_mean, global_std = calculate_global_stats(train_data)
np.savez('standardization_params.npz', mean=global_mean, std=global_std)
print(f"Global mean: {global_mean}, Global std: {global_std}")

train_features = [row["rt"] for row in train_data]
val_features = [row["rt"] for row in val_data]
test_features = [row["rt"] for row in test_data]
scaler = MinMaxScaler()
train_scaled_features = scaler.fit_transform(train_features)
val_scaled_features = scaler.transform(val_features)
test_scaled_features = scaler.transform(test_features)
joblib.dump(scaler, 'feature_scaler.save')

final_train = []
for i in range(len(train_data)):
    row = train_data[i]
    img = tiff.imread(row["img_path"]).astype(np.float32)
    img_std = (img - global_mean) / global_std
    new_path = "final/train/" + str(Path(row['img_path']).name)
    tiff.imwrite(new_path, img_std.astype(np.float32))
    row['img_path'] = new_path
    row['rt'] = train_scaled_features[i].tolist()
    final_train.append(row)

final_val = []
for i in range(len(val_data)):
    row = val_data[i]
    img = tiff.imread(row["img_path"]).astype(np.float32)
    img_std = (img - global_mean) / global_std
    new_path = "final/val/" + str(Path(row['img_path']).name)
    tiff.imwrite(new_path, img_std.astype(np.float32))
    row['img_path'] = new_path
    row['rt'] = val_scaled_features[i].tolist()
    final_val.append(row)

final_test = []
for i in range(len(test_data)):
    row = test_data[i]
    img = tiff.imread(row["img_path"]).astype(np.float32)
    img_std = (img - global_mean) / global_std
    new_path = "final/test/" + str(Path(row['img_path']).name)
    tiff.imwrite(new_path, img_std.astype(np.float32))
    row['img_path'] = new_path
    row['rt'] = test_scaled_features[i].tolist()
    final_test.append(row)

print(len(final_train), len(final_val), len(final_test))
print("Number per class in train set:", pd.Series([row['label'] for row in final_train]).value_counts().to_dict())
print("Number per class in validation set:", pd.Series([row['label'] for row in final_val]).value_counts().to_dict())
print("Number per class in test set:", pd.Series([row['label'] for row in final_test]).value_counts().to_dict())

train_df = pd.DataFrame(final_train)
train_df.to_csv('final/train.csv', index=False)

val_df = pd.DataFrame(final_val)
val_df.to_csv('final/val.csv', index=False)

test_df = pd.DataFrame(final_test)
test_df.to_csv('final/test.csv', index=False)
