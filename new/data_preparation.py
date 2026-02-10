
# Configuration
import random
import pandas as pd
import numpy as np
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
    aug_combinations = augmentation.get_all_combinations()
    
    # Get class distribution
    labels = [dataset[i]['label'] for i in range(len(dataset))]
    class_indices = {i: [] for i in range(4)}
    
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
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
                        'label': row['label'],
                        'rt': row['rt'],
                        'img_path': row['img_path'],
                    })
                continue

            # Amount of augmentation needed
            n_augment = target - current_count

            # Count how many augmentations to reach required amount for each sample
            n_per_sample = n_augment // current_count + 1

            for i in indices:
                # Pick a random sample from this class
                original_sample = dataset[i]
                final.append({
                    'img_path': original_sample['img_path'],
                    'label': original_sample['label'],
                    'rt': original_sample['rt'],
                })
                original_img = original_sample['image']

                for augs in aug_combinations[:n_per_sample]:
                    if n_augment <= 0:
                        break
                    
                    augmented_img = augmentation.apply_augmentations(original_img, augs)
                    filename = f'augment/{"_".join(augs)}_{Path(original_sample["img_path"]).name}'
                    try:
                        tiff.imwrite(filename, augmented_img)
                        final.append({
                            'label': original_sample['label'],
                            'rt': original_sample['rt'],
                            'img_path': filename,
                        })
                        n_augment -= 1
                    except Exception as e:
                        print(f"Error saving image {filename}: {e}")
        else:
            if target == -1:
                target = current_count
            else:
                target = min(target, current_count)
            
            undersampled_indices = random.sample(indices, target)
            for idx in undersampled_indices:
                row = dataset[idx]
                final.append({
                    'label': row['label'],
                    'rt': row['rt'],
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
os.makedirs("final/train", exist_ok=True)
os.makedirs("final/val", exist_ok=True)
os.makedirs("final/test", exist_ok=True)
for file_path in Path("new/PATCH_CAL").iterdir():
    img = tiff.imread(file_path).astype(np.float32)
    img_resized = resize(img, (64, 64), order=3, mode='reflect', preserve_range=True)
    if (img_resized.shape != (64, 64, 2)):
        print(img.shape)
        print(f"Error resizing {file_path.name}: got shape {img_resized.shape}")
    tiff.imwrite("resized_new/" + file_path.name, img_resized.astype(np.float32))


print("\nLoading datasets...")
dataset = OpenSARShipDataset(
    root_dir="new",
)

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
