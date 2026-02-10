

import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import tifffile as tiff


class OpenSARShipDataset(Dataset):
    """
    PyTorch Dataset for OpenSARShip (RT-based)

    Expected:
    - root_dir/metadata.csv
    - root_dir/resized_new/<patch_cal_filename>.tif (or .tiff)
    """

    def __init__(self, root_dir, image_dir="resized_new", use_cache=False):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_dir
        self.use_cache = use_cache

        # Load metadata
        self.ais_data = pd.read_csv(self.root_dir / "metadata.csv")

        # Build filtered list
        self.filter_data()

        # Optional cache (careful with RAM and multi-worker DataLoader)
        self._cache = {} if use_cache else None

    def filter_data(self):
        """Filter dataset for 4 target ship classes: Bulk Carrier, Container Ship, Fishing, Tanker"""
        ship_type_col = "category"

        filtered_data = []
        label_count = {0: 0, 1: 0, 2: 0, 3: 0}

        print(self.ais_data[ship_type_col].value_counts())

        for _, row in self.ais_data.iterrows():
            ship_type = row.get(ship_type_col, "")

            # --- Map to 4 classes ---
            label = None
            if "Cargo" in str(ship_type):
                elaborated_type = row.get("Elaborated_type", "")
                if str(elaborated_type) == "Bulk Carrier":
                    label = 0
                elif str(elaborated_type) == "Container Ship":
                    label = 1
                else:
                    continue
            elif str(ship_type) == "Fishing":
                label = 2
            elif "Tanker" in str(ship_type):
                label = 3
            else:
                continue

            patch_name = row.get("patch_cal", None)
            if patch_name is None:
                continue

            img_path = (self.image_dir / str(patch_name)).resolve()
            if not img_path.exists():
                continue

            # Collect all metadata needed for RT (use 0.0 if column doesn't exist)
            # NOTE: These keys must match what extract_rt_tensor reads.
            sample = {
                "label": int(label),
                "img_path": str(img_path),
                "img_id": str(patch_name),

                # --- RT-related metadata (add/rename columns here to match your metadata.csv) ---
                "Incidence": float(row.get("Incidence", 0.0)),
                "AzimuthAngle": float(row.get("AzimuthAngle", 0.0)),
                "RelativeHeading": float(row.get("RelativeHeading", 0.0)),
                "SlantRange": float(row.get("SlantRange", 0.0)),
                "dx": float(row.get("dx", 0.0)),
                "dy": float(row.get("dy", 0.0)),
                "Speed": float(row.get("Speed", 0.0)),
                "LookDirection": float(row.get("LookDirection", 0.0)),
            }

            filtered_data.append(sample)
            label_count[int(label)] += 1

        self.data = filtered_data
        print(f"Loaded {len(self.data)} samples")
        print(f"Class distribution:\n{label_count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # ---- Optional cache ----
        if self.use_cache and idx in self._cache:
            return self._cache[idx]

        # ---- Load image ----
        image = tiff.imread(row["img_path"]).astype(np.float32)

        # Convert to torch tensor in CHW format
        image = torch.from_numpy(image).float()
        if image.ndim == 2:
            # (H, W) -> (1, H, W)
            image = image.unsqueeze(0)
        elif image.ndim == 3:
            # If (H, W, C) -> (C, H, W)
            # Common case in your code: C=2 at last dim
            if image.shape[-1] in (1, 2, 3, 4):
                image = image.permute(2, 0, 1)

        # ---- Build RT tensor ----
        rt = self.extract_rt_tensor(row)
        rt = torch.from_numpy(rt).float()

        out = {
            "image": image,                 # torch.FloatTensor [C,H,W]
            "rt": rt,                       # torch.FloatTensor [8]
            "label": torch.tensor(row["label"], dtype=torch.long),
            "img_id": row["img_id"],
            "img_path": row["img_path"],
        }

        if self.use_cache:
            self._cache[idx] = out

        return out

    def get_labels(self):
        """Return all labels for stratified splitting"""
        return [self.data[i]["label"] for i in range(len(self.data))]

    def extract_rt_tensor(self, row):
        """
        Extract Rotation–Translation (RT) tensor from metadata only.
        Output fixed-size: (8,)
        """
        rt = np.zeros(8, dtype=np.float32)

        # --- ROTATION-like components (angles/orientation context) ---
        rt[0] = float(row.get("Incidence", 0.0))         # incidence angle
        rt[1] = float(row.get("AzimuthAngle", 0.0))      # radar azimuth angle
        rt[2] = float(row.get("RelativeHeading", 0.0))   # relative ship heading vs sensor

        # --- TRANSLATION-like components (relative position context) ---
        rt[3] = float(row.get("SlantRange", 0.0))        # sensor-target distance
        rt[4] = float(row.get("dx", 0.0))                # cross-track offset
        rt[5] = float(row.get("dy", 0.0))                # along-track offset

        # --- MOTION / platform context ---
        rt[6] = float(row.get("Speed", 0.0))             # AIS speed (or relative speed)
        rt[7] = float(row.get("LookDirection", 0.0))     # left/right looking encoded

        return rt


class FinalDataset(Dataset):
    """
    Load training/val/test from CSV.
    Supports both 'features' column (old) and 'rt' column (new).
    """

    def __init__(self, csv_path, feature_col_preference=("rt", "features")):
        self.data = pd.read_csv(csv_path)
        self.feature_col_preference = feature_col_preference

    def __len__(self):
        return len(self.data)

    def _read_feature_vector(self, row):
        # Prefer 'rt' if exists; fallback to 'features'
        col = None
        for c in self.feature_col_preference:
            if c in row and pd.notna(row[c]):
                col = c
                break
        if col is None:
            raise KeyError(f"CSV must contain one of these columns: {self.feature_col_preference}")

        # Safer than eval()
        vec = ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col]
        return np.array(vec, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        image = tiff.imread(str(row["img_path"])).astype(np.float32)
        image = torch.from_numpy(image).float()

        # Ensure CHW
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3:
            if image.shape[-1] in (1, 2, 3, 4):
                image = image.permute(2, 0, 1)

        # Load RT/features vector
        features = self._read_feature_vector(row)

        return {
            "image": image,  # torch.FloatTensor [C,H,W]
            "rt": torch.from_numpy(features).float(),  # torch.FloatTensor [N]
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }
