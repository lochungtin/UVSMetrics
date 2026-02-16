import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, center_of_mass, shift


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


def getFacingSurface(A, B, sA, sB, N=32):
    visible = np.zeros(len(sB), dtype=bool)
    T = np.linspace(2.0 / (N + 1), (N - 1.0) / (N + 1), N, dtype=np.float32)
    SHAPE = A.shape

    for src in sA.astype(np.float32):
        todo = np.where(~visible)[0]
        if len(todo) == 0:
            break

        pts = src[None, None, :] + T[None, :, None] * (sB[todo] - src)[:, None, :]

        idx = np.round(pts).astype(np.int32)

        idx[..., 0] = np.clip(idx[..., 0], 0, SHAPE[0] - 1)
        idx[..., 1] = np.clip(idx[..., 1], 0, SHAPE[1] - 1)
        idx[..., 2] = np.clip(idx[..., 2], 0, SHAPE[2] - 1)

        x, y, z = idx[..., 0], idx[..., 1], idx[..., 2]
        clear = ~(A[x, y, z] | B[x, y, z]).any(axis=1)

        visible[todo[clear]] = True

    face = np.zeros_like(A, dtype=bool)
    face[tuple(sB[visible].T)] = True
    return face


def getFacingSurfaces(A, B):
    sA = getSurface(A, idx=True)
    sB = getSurface(B, idx=True)
    return getFacingSurface(B, A, sB, sA), getFacingSurface(A, B, sA, sB)


def getFacingSurfacePercentage(source, target, facingS=None, facingT=None):
    if facingS is None or facingT is None:
        facingS, facingT = getFacingSurfaces(source, target)

    return np.sum(facingS) / getSurfaceArea(source), np.sum(facingT) / getSurfaceArea(target)


def getFacingSurfaceDistance(source, target, dmapS, dmapT, facingS=None, facingT=None):
    if facingS is None or facingT is None:
        facingS, facingT = getFacingSurfaces(source, target)

    return dmapT[np.where(facingS)], dmapS[np.where(facingT)]
