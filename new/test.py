from pathlib import Path
import tifffile as tiff
import numpy as np

for file_path in Path("new/PATCH_CAL").iterdir():
    img = tiff.imread(file_path).astype(np.float32)
    # Check if image is not 3 dimensional
    if len(img.shape) != 3:
        print(f"Image {file_path.name} has unexpected shape: {img.shape}")