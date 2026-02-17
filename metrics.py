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


def getTranslationVector(source, target):
    return getCenterOfMass(target) - getCenterOfMass(source)


def getCenterOfMass(mask):
    return np.asarray(center_of_mass(mask))


def getSurface(mask, idx=False):
    if idx:
        return np.argwhere(mask & ~binary_erosion(mask.astype(bool)))
    return mask & ~binary_erosion(mask.astype(bool))


def getDMap(mask):
    return distance_transform_edt(~(mask.astype(bool)))


# === Rigid Alignment ===


def applyTranslationVector(mask, vector):
    return shift(mask.astype(float), vector, order=0) > 0.5


def rigidAlign(source, target, vector=None):
    vector = getTranslationVector(source, target) if vector is None else vector
    return applyTranslationVector(source, vector), target


# === Volume Metric ===


def getVolume(mask):
    return np.sum(mask).astype(np.int32)


def getSurfaceArea(mask):
    return getVolume(getSurface(mask))


def getVolumeDiff(A, B):
    return getVolume(A) - getVolume(B)


def getSurfaceAreaDiff(A, B):
    return getSurfaceArea(A) - getSurfaceArea(B)


# === Mask (binary) Dilation Metric ===


def getDilatedMask(thickness, mask=None, dmap=None):
    dmap = getDMap(mask) if dmap is None else dmap
    return dmap <= thickness


# === Surface Facing Metric ===


def getFacingSurface(A, B, cutAwayDist=8, N=32):
    cutA = (getDMap(B) * A > cutAwayDist) & A

    sA, sB = getSurface(cutA, idx=True), getSurface(B, idx=True)
    visible = np.zeros(len(sB), dtype=bool)

    _A = torch.tensor(cutA, dtype=torch.bool, device=DEV)
    _B = torch.tensor(B, dtype=torch.bool, device=DEV)
    SHAPE = _A.shape

    T = torch.linspace(2.0 / (N + 1), (N - 1.0) / (N + 1), N, device=DEV)

    for src in sA:
        todo = np.where(~visible)[0]
        if len(todo) == 0:
            break

        _sA = torch.tensor(src, dtype=torch.float32, device=DEV)
        _sB = torch.tensor(sB[todo], dtype=torch.float32, device=DEV)

        pts = _sA[None, None, :] + T[None, :, None] * (_sB - _sA)[:, None, :]

        idx = pts.round().long()

        idx[..., 0].clamp_(0, SHAPE[0] - 1)
        idx[..., 1].clamp_(0, SHAPE[1] - 1)
        idx[..., 2].clamp_(0, SHAPE[2] - 1)

        x, y, z = idx[..., 0], idx[..., 1], idx[..., 2]
        blocked = (_A[x, y, z] | _B[x, y, z]).any(dim=1)
        clear = (~blocked).cpu().numpy()

        visible[todo[clear]] = True

    face = np.zeros_like(A, dtype=bool)
    face[tuple(sB[visible].T)] = True
    return face


def getFacingSurfaces(A, B, cutAwayDist=8, N=32):
    _sA = getSurface(A)
    _sB = getSurface(B)

    _SE3 = np.zeros((5, 5, 5))

    overlapA = _sA & B
    overlapB = _sB & A

    neighbourA = _sA & binary_dilation(B, structure=_SE3)
    neighbourB = _sB & binary_dilation(A, structure=_SE3)

    interfaceA = A & binary_dilation(B & ~A, structure=_SE3)
    interfaceB = B & binary_dilation(A & ~B, structure=_SE3)

    faceA = (
        getFacingSurface(B, A, cutAwayDist=cutAwayDist, N=N)
        | overlapA
        | neighbourA
        | interfaceA
    )
    faceB = (
        getFacingSurface(A, B, cutAwayDist=cutAwayDist, N=N)
        | overlapB
        | neighbourB
        | interfaceB
    )

    return faceA, faceB


def getFacingSurfacePercentage(A, facingA):
    return np.sum(facingA) / getSurfaceArea(A)


def getFacingSurfaceDistance(dmapA, facingA):
    return dmapA[np.where(facingA)]
