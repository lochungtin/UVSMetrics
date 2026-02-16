import numpy as np
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


def crop_masks_with_shared_margin(masks, margin=2):
    shape = masks[0].shape
    combined = np.zeros(shape, dtype=bool)
    for m in masks:
        combined |= m > 0

    axes = tuple(range(3))
    x_any = np.any(combined, axis=(1, 2))
    y_any = np.any(combined, axis=(0, 2))
    z_any = np.any(combined, axis=(0, 1))

    x_indices = np.where(x_any)[0]
    y_indices = np.where(y_any)[0]
    z_indices = np.where(z_any)[0]

    xmin, xmax = x_indices[0], x_indices[-1]
    ymin, ymax = y_indices[0], y_indices[-1]
    zmin, zmax = z_indices[0], z_indices[-1]

    xmin = max(xmin - margin, 0)
    ymin = max(ymin - margin, 0)
    zmin = max(zmin - margin, 0)

    xmax = min(xmax + margin, shape[0] - 1)
    ymax = min(ymax + margin, shape[1] - 1)
    zmax = min(zmax + margin, shape[2] - 1)

    cropped_masks = [m[xmin : xmax + 1, ymin : ymax + 1, zmin : zmax + 1] for m in masks]

    return cropped_masks


positions = ["prostate_supine", "prostate_upright"]
objects = [
    "bladder",
    "femoral_head_l",
    "femoral_head_r",
    "penile_bulb",
    "prostate",
    "rectum",
    "seminal_vesicles",
]
files = []
for pos in positions:
    for obj in objects:
        files.append(f"{pos}_{obj}.npy")

masks = [np.load(join("data", file)) for file in files]
cropped = crop_masks_with_shared_margin(masks)
print(cropped[0].shape)

for file, mask in zip(files, cropped):
    np.save(join("cropped", file), mask)
