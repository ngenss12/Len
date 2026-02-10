import pandas as pd
import tifffile as tiff
import numpy as np
from pathlib import Path

df = pd.read_csv("augmented.csv")
params = np.load('standardization_params.npz')
mean = params['mean']
std = params['std']

new_data = []
for idx, row in df.iterrows():
    img = tiff.imread(row["img_path"]).astype(np.float32)
    img_std = (img - mean) / std
    tiff.imwrite("standardised/" + str(Path(row['img_path']).name), img_std.astype(np.float32))
    new_data.append({
        'label': row['label'],
        'rt': row['rt'],
        'img_path': "standardised/" + str(Path(row['img_path']).name),
    })

new_df = pd.DataFrame(new_data)
new_df.to_csv('standardised_dataset.csv', index=False)
