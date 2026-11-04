"""Fine-tune Cellpose on per-slice glomerulus labels, and predict 3D labels.

The ROIs were drawn per z-slice, so we train 2D: every z-slice of every
training fly is one training image. At prediction time, slices are
segmented independently and stitched into 3D objects by overlap.

Flies can have different in-plane pixel sizes, so with ``xy_um`` every slice is
resampled in-plane to that pixel size for the model, and predictions are resampled
back. Planes are never resampled: interpolating between planes several um apart would
invent data.

Written against Cellpose 4 (Cellpose-SAM, 3 input channels). Cellpose 3
works the same way but takes a ``channels`` argument and a ``model_type``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def erode_labels(labels: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Shrink each object in a 2D label image by ``iterations`` pixels.

    Training on cores teaches the model to stay off borders, where signal
    from the neighboring glomerulus leaks in. Objects that would vanish are
    kept at their original size.
    """
    if iterations <= 0:
        return labels
    out = np.zeros_like(labels)
    for lab, sl in _objects(labels):
        m = labels[sl] == lab
        e = ndi.binary_erosion(m, iterations=iterations, border_value=0)
        if e.sum() < 0.25 * m.sum():
            e = m
        out[sl][e] = lab
    return out


def _objects(labels):
    for i, sl in enumerate(ndi.find_objects(labels)):
        if sl is not None:
            yield i + 1, sl


def xy_shape(shape_yx, spacing, xy_um: float | None) -> tuple[int, int]:
    """In-plane shape of a (Y, X) grid with voxel size ``spacing`` (z, y, x) after
    resampling to ``xy_um`` pixels (unchanged if ``xy_um`` is None)."""
    if xy_um is None:
        return tuple(shape_yx)
    return tuple(max(1, int(round(n * d / xy_um))) for n, d in zip(shape_yx, spacing[1:]))


def resize_xy(a: np.ndarray, shape_yx, order: int) -> np.ndarray:
    """Resample the last two axes of ``a`` to ``shape_yx`` (order 0 for labels)."""
    if tuple(a.shape[-2:]) == tuple(shape_yx):
        return a
    from skimage.transform import resize

    out = resize(a, a.shape[:-2] + tuple(shape_yx), order=order, mode="edge",
                 preserve_range=True, anti_aliasing=False)
    return out.astype(a.dtype)


def make_training_slices(
    flies: list[dict],
    erode: int = 1,
    min_objects: int = 1,
    xy_um: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """flies: [{"channels": (C, Z, Y, X), "labels": (Z, Y, X), "spacing": (z, y, x)}, ...]

    Returns lists of (Y, X, C) images and (Y, X) relabeled instance masks, resampled
    in-plane to ``xy_um`` pixels if given (``erode`` is in those pixels). Slices with
    fewer than ``min_objects`` labeled glomeruli are skipped.
    """
    imgs, masks = [], []
    for f in flies:
        shape = xy_shape(f["labels"].shape[1:], f.get("spacing", (1, 1, 1)), xy_um)
        ch, lab = resize_xy(f["channels"], shape, 1), resize_xy(f["labels"], shape, 0)
        for z in range(lab.shape[0]):
            m = lab[z]
            if len(np.unique(m)) - 1 < min_objects:
                continue
            m2 = _relabel_2d(m)
            imgs.append(np.moveaxis(ch[:, z], 0, -1).astype(np.float32))
            masks.append(erode_labels(m2, erode))
    return imgs, masks


def _relabel_2d(m: np.ndarray) -> np.ndarray:
    """Split a glomerulus that appears as two disconnected pieces on a slice."""
    out = np.zeros(m.shape, np.int32)
    n = 0
    for lab in np.unique(m):
        if lab == 0:
            continue
        cc, k = ndi.label(m == lab)
        out[cc > 0] = cc[cc > 0] + n
        n += k
    return out


def augment(imgs, masks, seed=0):
    """Flips and 90-degree rotations. Cellpose adds scale/rotation jitter itself."""
    rng = np.random.default_rng(seed)
    A, B = [], []
    for x, m in zip(imgs, masks):
        k = rng.integers(4)
        x, m = np.rot90(x, k), np.rot90(m, k)
        if rng.random() < 0.5:
            x, m = x[:, ::-1], m[:, ::-1]
        A.append(np.ascontiguousarray(x))
        B.append(np.ascontiguousarray(m))
    return imgs + A, masks + B


def train(
    train_flies,
    val_flies,
    save_dir: str | Path,
    model_name: str = "al_glomeruli",
    n_epochs: int = 200,
    learning_rate: float = 1e-5,
    erode: int = 1,
    gpu: bool = True,
    pretrained_model: str = "cpsam_v2",
    xy_um: float | None = None,
):
    """Fine-tune Cellpose, on slices resampled to ``xy_um`` pixels if given.
    Returns the path of the saved model, and the training and validation losses."""
    from cellpose import models, train as cp_train

    Xtr, Ytr = augment(*make_training_slices(train_flies, erode=erode, xy_um=xy_um))
    Xva, Yva = make_training_slices(val_flies, erode=erode, xy_um=xy_um)
    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    path, train_losses, val_losses = cp_train.train_seg(
        model.net,
        train_data=Xtr,
        train_labels=Ytr,
        test_data=Xva,
        test_labels=Yva,
        channel_axis=-1,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=0.1,
        batch_size=8,
        min_train_masks=1,
        save_path=str(save_dir),
        model_name=model_name,
    )
    return path, train_losses, val_losses


def predict(
    model_path: str | Path,
    channels: np.ndarray,
    diameter: float | None = None,
    stitch_threshold: float = 0.25,
    gpu: bool = True,
    dilate: int = 0,
    spacing=(1.0, 1.0, 1.0),
    xy_um: float | None = None,
) -> np.ndarray:
    """Segment a fly. channels: (C, Z, Y, X). Returns (Z, Y, X) labels.

    Pass the ``xy_um`` the model was trained at, and the fly's voxel size
    (``spacing``), to segment at that pixel size; labels come back on the fly's grid.
    If the model was trained on eroded targets, predictions are cores; pass
    ``dilate`` to grow them back, or better, keep them as cores and let
    :mod:`alseg.refine` decide which border voxels are clean.
    """
    from cellpose import models

    model = models.CellposeModel(gpu=gpu, pretrained_model=str(model_path))
    native = channels.shape[2:]
    channels = resize_xy(channels, xy_shape(native, spacing, xy_um), 1)
    stack = np.moveaxis(channels, 0, -1)  # (Z, Y, X, C)
    masks, _, _ = model.eval(
        stack,
        channel_axis=-1,
        z_axis=0,
        do_3D=False,
        stitch_threshold=stitch_threshold,
        diameter=diameter,
        normalize=False,
    )
    masks = resize_xy(np.asarray(masks), native, 0)
    if dilate:
        from skimage.segmentation import expand_labels

        masks = np.stack([expand_labels(m, dilate) for m in masks])
    return masks
