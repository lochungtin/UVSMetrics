import numpy as np
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from metrics import getFacingSurfaces

positions = ["prostate_supine", "prostate_upright"]
objects = [
    "bladder",
    "femoral_head_l",
    "femoral_head_r",
    "prostate",
    "rectum",
]
files = []
for obj in objects:
    files.append(f"prostate_upright_{obj}.npy")

masks = [np.load(join("data", file)).transpose(2, 1, 0) for file in files]
combined = np.zeros_like(masks[0], dtype=bool)
# for m in masks:
#     combined |= m > 0

sA, sB = getFacingSurfaces(masks[3], masks[4])

combined = sA | sB

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

verts, faces, normals, values = measure.marching_cubes(combined.astype(np.uint8), level=0)

plt.close("all")
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2])
ax.axis("equal")
plt.show()
