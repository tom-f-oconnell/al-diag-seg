"""ImageJ RoiSets drawn on individual z-slices to 3D label volumes, and back.

Each ROI in the RoiSet is a polygon on one z-slice. ROIs that share a name
(e.g. "DM2") are merged into one 3D object. If ROI names are not glomerulus
names (ImageJ's default names look like "0012-0123-0456"), set
``link_unnamed=True`` and per-slice ROIs are linked into 3D objects by
overlap between neighboring slices instead.

Names are kept as written, but for a suffix after the glomerulus name (``DA1_t0``,
``DM2_z12``, ``DM2 (3)`` are DA1 and DM2): the database's names are case-sensitive
(``DA4l``, ``VA1d``), so they are not changed.

An ROI covers the pixels whose centers are inside its polygon, with pixel (y, x)'s center
at ImageJ coordinates (x, y), as hong2p (al_analysis) reads them, so masks here are the
ones the database's responses were computed in. `write_roiset` writes outlines that are
read back to the same masks.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import roifile
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as ndi_label
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
    keep=None,
) -> tuple[np.ndarray, dict[int, str], np.ndarray]:
    """Read a RoiSet.zip (or single .roi) into a label volume. ``keep``: if given, only the
    ROIs it says to keep, by ``keep(name, names)``: an ROI's name as in the RoiSet, and the
    names of all of them (see `alseg.olfdb.roi_keeper`).

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

    if keep is not None:
        names = {r.name or "" for r in rois}
        rois = [r for r in rois if keep(r.name or "", names)]
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
    # "DM2_z12", "DM2 (3)", "DA1_t0" -> "DM2", "DA1" (case as written: the database's names are case-sensitive)
    return re.split(r"[_\s(]", name.strip())[0]


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


def build_name_table(roi_paths, shapes, keep=None) -> dict[str, int]:
    """One fixed glomerulus-name -> label table shared by all flies (``keep``: as in `load_roiset`)."""
    names = set()
    for p, s in zip(roi_paths, shapes):
        _, lut, _ = load_roiset(p, s, keep=keep)
        names |= set(lut.values())
    return {n: i + 1 for i, n in enumerate(sorted(names))}


def _outlines(mask: np.ndarray) -> list[np.ndarray]:
    """(N, 2) x, y outlines of each connected piece of a 2D mask (holes filled), which
    `roi_to_mask2d` turns back into those pieces."""
    from skimage.measure import find_contours

    out = []
    pieces, n = ndi_label(mask)
    for k in range(1, n + 1):
        piece = binary_fill_holes(pieces == k)
        # marching squares at 0.5 between pixel centers, so each pixel of the piece has its
        # center inside, and no other pixel does
        c = max(find_contours(np.pad(piece, 1).astype(float), 0.5), key=len) - 1
        out.append(np.ascontiguousarray(c[:, ::-1], np.float32))
    return out


def write_roiset(labels: np.ndarray, names: dict[int, str | None], path: str | Path) -> int:
    """Writes a (Z, Y, X) label volume as an ImageJ RoiSet.zip: a polygon ROI per label and
    plane (one per connected piece, if a label is in several on a plane), named
    ``names[label]``, or the label's number for one without a name (as numbered ROIs, they
    are not glomeruli to the database), with its plane as a hyperstack Z, as al_analysis'
    RoiSets have them. Returns how many ROIs were written.
    """
    path = Path(path)
    if path.exists():
        path.unlink()
    n = 0
    for z in range(labels.shape[0]):
        for lab in np.unique(labels[z]):
            if lab == 0:
                continue
            for xy in _outlines(labels[z] == lab):
                roi = roifile.ImagejRoi.frompoints(xy, name=names.get(int(lab)) or str(int(lab)), z=z)  # (0-based: written 1-based)
                roi.roitype = roifile.ROI_TYPE.POLYGON
                roi.tofile(path, name=f"{n:04d}-{z + 1:04d}-{int(lab):04d}.roi")
                n += 1
    return n
