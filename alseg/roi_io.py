"""Convert ImageJ RoiSets drawn on individual z-slices into 3D label volumes.

Each ROI in the RoiSet is a polygon on one z-slice. ROIs that share a name
(e.g. "DM2") are merged into one 3D object. If ROI names are not glomerulus
names (ImageJ's default names look like "0012-0123-0456"), set
``link_unnamed=True`` and per-slice ROIs are linked into 3D objects by
overlap between neighboring slices instead.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import roifile
from skimage.draw import polygon as draw_polygon

# ImageJ's auto-generated names: "ssss-yyyy-xxxx" or "cccc-ssss-yyyy-xxxx" ...
_DEFAULT_NAME = re.compile(r"^\d{4}(-\d{4,5}){1,3}(-\d+)?$")


def roi_z_index(roi: roifile.ImagejRoi, n_slices: int) -> int:
    """0-based z index of an ROI.

    ImageJ stores a 1-based position. Hyperstack ROIs use ``z_position``;
    ROIs drawn on a plain stack use ``position``. A value of 0 means "all
    slices", which we treat as an error for per-slice annotation.
    """
    z = roi.z_position if roi.z_position > 0 else roi.position
    if z <= 0:
        raise ValueError(f"ROI {roi.name!r} has no slice position (drawn on all slices?)")
    z -= 1
    if z >= n_slices:
        raise ValueError(f"ROI {roi.name!r} is on slice {z + 1} but the stack has {n_slices}")
    return z


def roi_to_mask2d(roi: roifile.ImagejRoi, shape_yx: tuple[int, int]) -> np.ndarray:
    """Rasterize one ROI (polygon, freehand, rectangle, oval...) to a boolean mask."""
    mask = np.zeros(shape_yx, bool)
    for xy in _subpaths(roi):
        rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=shape_yx)
        mask[rr, cc] ^= True  # XOR so holes in composite ROIs stay holes
    return mask


def _subpaths(roi):
    coords = roi.coordinates(multi=True)
    if isinstance(coords, np.ndarray):
        coords = [coords]
    return [np.asarray(c, float) for c in coords if len(c) >= 3]


def load_roiset(
    roi_path: str | Path,
    shape_zyx: tuple[int, int, int],
    name_to_label: dict[str, int] | None = None,
    link_unnamed: bool | None = None,
    min_link_iou: float = 0.2,
) -> tuple[np.ndarray, dict[int, str], np.ndarray]:
    """Read a RoiSet.zip (or single .roi) into a label volume.

    Returns
    -------
    labels : (Z, Y, X) uint16, 0 = background
    lut : {label: glomerulus name}
    overlap : (Z, Y, X) bool, voxels claimed by more than one ROI. These are
        set to 0 in ``labels``; pass them as "ignore" when training.
    """
    rois = roifile.ImagejRoi.fromfile(str(roi_path))
    if not isinstance(rois, list):
        rois = [rois]
    Z, Y, X = shape_zyx
    if link_unnamed is None:
        link_unnamed = all(_DEFAULT_NAME.match(r.name or "") for r in rois)

    # (name, z, mask) per ROI
    items = []
    for r in rois:
        z = roi_z_index(r, Z)
        items.append((r.name or "", z, roi_to_mask2d(r, (Y, X))))

    if link_unnamed:
        items = _link_by_overlap(items, min_link_iou)

    by_name: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for name, z, m in items:
        by_name[_clean_name(name)].append((z, m))

    if name_to_label is None:
        name_to_label = {n: i + 1 for i, n in enumerate(sorted(by_name))}

    labels = np.zeros(shape_zyx, np.uint16)
    count = np.zeros(shape_zyx, np.uint8)
    for name, slices in by_name.items():
        if name not in name_to_label:
            raise KeyError(f"ROI name {name!r} not in name_to_label")
        lab = name_to_label[name]
        for z, m in slices:
            newly = m & (labels[z] != lab)
            count[z][newly] += 1
            labels[z][m] = lab
    overlap = count > 1
    labels[overlap] = 0
    lut = {v: k for k, v in name_to_label.items() if k in by_name}
    return labels, lut, overlap


def _clean_name(name: str) -> str:
    # "DM2_z12", "DM2 (3)", "dm2" -> "DM2". Adjust to your lab's naming scheme.
    name = re.split(r"[_\s(]", name.strip())[0]
    return name.upper()


def _link_by_overlap(items, min_iou):
    """Give per-slice ROIs a shared name when they overlap across adjacent slices."""
    items = sorted(items, key=lambda t: t[1])
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (_, zi, mi) in enumerate(items):
        for j in range(i + 1, len(items)):
            _, zj, mj = items[j]
            if zj > zi + 1:
                break
            if zj != zi + 1:
                continue
            inter = np.logical_and(mi, mj).sum()
            union = np.logical_or(mi, mj).sum()
            if union and inter / union >= min_iou:
                parent[find(j)] = find(i)
    return [(f"obj{find(i):04d}", z, m) for i, (_, z, m) in enumerate(items)]


def build_name_table(roi_paths, shapes) -> dict[str, int]:
    """One fixed glomerulus-name -> label table shared by all flies."""
    names = set()
    for p, s in zip(roi_paths, shapes):
        _, lut, _ = load_roiset(p, s)
        names |= set(lut.values())
    return {n: i + 1 for i, n in enumerate(sorted(names))}
