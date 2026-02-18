import numpy as np
import torch
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    center_of_mass,
    distance_transform_edt,
    shift,
)

if torch.cuda.is_available():
    DEV = "cuda"
else:
    DEV = "cpu"
print(f"Metric Functions Using: {DEV}")


class Mask:
    def __init__(self, mask):
        self.mask = mask.astype(bool)
        self.surface = self.mask & ~binary_erosion(self.mask)
        self.volume = np.sum(self.mask).astype(np.int32)
        self.surface_area = np.sum(self.surface).astype(np.int32)
        self.center_of_mass = np.asarray(center_of_mass(self.mask))
        self.SHAPE = self.mask.shape
        self.dmap = None

    # === Geometrical Analysis ===

    def getVolDiff(self, mask):
        return mask.volume - self.volume

    def getSADiff(self, mask):
        return mask.surface_area - self.surface_area

    # === Get Distance Map ===

    def getDMap(self):
        if self.dmap is None:
            self.dmap = distance_transform_edt(~self.mask)
        return self.dmap

    def maskDMap(self, mask):
        m = mask
        if isinstance(mask, Mask):
            m = mask.mask
        return self.getDMap()[np.where(m)]

    # === Get Coordinates ===

    def getMaskCoordinates(self):
        return np.where(self.mask)

    def getSurfaceCoordinates(self):
        return np.argwhere(self.surface)

    # === Rigid Alignment ===

    def getTranslationVector(self, mask):
        return mask.center_of_mass - self.center_of_mass

    def translate(self, vector):
        return Mask(shift(self.mask.astype(float), vector, order=0) > 0.5)

    def rigidAlign(self, target):
        return self.translate(self.getTranslationVector(target))

    # === Bi-directional Surface Discrepancy ===

    def bidirectionalSurfaceDiscrepancy(self, mask, bi_only=True):
        selfAligned = self.rigidAlign(mask)
        stD = selfAligned.maskDMap(mask.surface)
        tsD = mask.maskDMap(selfAligned.surface)
        if bi_only:
            return np.concatenate([stD, tsD])
        return np.concatenate([stD, tsD]), stD, tsD

    # === Mask Binary Dilation ===

    def dilateMask(self, thickness):
        return Mask(self.getDMap() <= thickness)

    # === Mutual Ray-cast Visible Surfaces ===
    def getRayCastVisibleSurface(self, raySource, cutAwayDist=8, N=32):
        _raySource = Mask(
            (self.getDMap() * raySource.mask > cutAwayDist) & raySource.mask
        ).surface
        _raySource |= raySource.surface

        sourceSurfaceCoords = np.argwhere(_raySource)
        selfSurfaceCoords = self.getSurfaceCoordinates()

        visible = np.zeros(len(selfSurfaceCoords), dtype=bool)

        sourceMaskTensor = torch.tensor(raySource.mask, dtype=torch.bool, device=DEV)
        selfMaskTensor = torch.tensor(self.mask, dtype=torch.bool, device=DEV)

        T = torch.linspace(2.0 / (N + 1), (N - 1.0) / (N + 1), N, device=DEV)

        for src in sourceSurfaceCoords:
            todo = np.where(~visible)[0]
            if len(todo) == 0:
                break

            _sA = torch.tensor(src, dtype=torch.float32, device=DEV)
            _sB = torch.tensor(selfSurfaceCoords[todo], dtype=torch.float32, device=DEV)

            pts = _sA[None, None, :] + T[None, :, None] * (_sB - _sA)[:, None, :]

            idx = pts.round().long()

            idx[..., 0].clamp_(0, self.SHAPE[0] - 1)
            idx[..., 1].clamp_(0, self.SHAPE[1] - 1)
            idx[..., 2].clamp_(0, self.SHAPE[2] - 1)

            x, y, z = idx[..., 0], idx[..., 1], idx[..., 2]
            blocked = (selfMaskTensor[x, y, z] | sourceMaskTensor[x, y, z]).any(dim=1)
            clear = (~blocked).cpu().numpy()

            visible[todo[clear]] = True

        face = np.zeros(self.SHAPE, dtype=bool)
        face[tuple(selfSurfaceCoords[visible].T)] = True
        return Mask(face)


# === helper functions ===


def seriesAnalysis(arr, labels=False):
    out = [
        np.min(arr),
        np.percentile(arr, 5),
        np.percentile(arr, 10),
        np.mean(arr),
        np.median(arr),
        np.percentile(arr, 90),
        np.percentile(arr, 95),
        np.max(arr),
    ]
    if labels:
        return out, ["MIN", "P05", "P10", "AVG", "MDN", "P90", "P95", "MAX"]
    return out


def prettyPrintTable(rowNames, rows, labels, rounding=3, minWidth=0):
    def pad(data, gaps):
        return [str(d).ljust(g + 1) for d, g in zip(data, gaps)]

    for r in range(len(rows)):
        rows[r] = [round(i, rounding) if isinstance(i, float) else i for i in rows[r]]

    gaps = [max(map(len, rowNames))]
    for i in range(len(labels)):
        colMax = len(labels[i])
        for row in rows:
            colMax = max(len(str(row[i])), colMax, minWidth)
        gaps.append(colMax)

    rows = [list(map(str, row)) for row in rows]

    rowSep = "+" + "+".join("-" * (i + 2) for i in gaps) + "+"
    print(rowSep)
    print("| " + "| ".join(pad([""] + labels, gaps)) + "|")
    print(rowSep)
    for name, row in zip(rowNames, rows):
        print("| " + "| ".join(pad([name] + row, gaps)) + "|")
        print(rowSep)
