"""A model's atlas on a fly's grid, moved to where the fly's ROIs say its glomeruli are.

The atlas reaches a fly's grid by registering images (`atlas.atlas_to_fly`), which leaves its glomeruli some um
from the fly's own, most of all in z, where a fly has few planes far apart. An ROI named as a glomerulus says where
that glomerulus of the atlas is in this fly. So a transformation of the atlas (um; atlas -> fly) is fit to such pairs
(`fit`), the atlas' maps are resampled by it (`move`), and the ROIs are named with the atlas moved
(`alseg.model.Model.align`, which `Model.predict` calls, as `Options.align` asks). Which ROIs: for an affine map,
those that odor tuning alone names, wherever they are (`TUNING_COST`); else, and for a shift, those named with the
atlas as registered, in steps (`STEPS`).

The transformation is one of `KINDS`, for the whole fly: a shift, or an affine map. It is fit robustly (reweighted
least squares, Tukey's biweight), as some of the names it is fit to are wrong: an ROI named wrong lies far from where
the others put its glomerulus, and gets no weight. The linear part of an affine map is held to the identity as by
``ridge`` more ROIs lying where the atlas expects them, so that a fly with few names is moved by little more than a
shift. What kind of map the atlas is off by, which ROIs to fit it to, and what moving it gains, were measured on the
database's flies: the README has the numbers (an affine map; a smooth warp on top of it gains nothing).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .identity import atlas_centroids

# the transformations a fly's atlas can be moved by: which entries of the matrix A of x -> A x + b are fit
# (rows and columns z, y, x; the others are the identity's)
KINDS = {
    "shift": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    "affine": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
}
# the steps an atlas is moved in by a fit to the ROIs named with it, by what is asked for (`alseg.model.Options.align`):
# each a fit of that kind, of the atlas as registered, to the ROIs as they are named after the step before. An affine
# map comes after a shift: the ROIs named with the atlas as registered are those near where it expects them, so a map
# fit to them alone falls short of the move needed
STEPS = {"shift": ("shift",), "affine": ("shift", "affine")}
# an affine map is rather fit to the ROIs that odor tuning alone names, wherever they are (`alseg.model.Model.align`):
# those tuned like a glomerulus at this cost at most (1 - the correlation of their tunings: as alike as masks of
# adjacent planes are to be joined, by default), in a fly with this many of the model's odors at least. With fewer,
# tuning alone names too many ROIs wrong: on the database's flies named by 16 of their 22 odors or fewer, the fit to
# the ROIs named with the atlas named more glomeruli right; by 18 or more, this one (the README has the numbers)
TUNING_COST = 0.3
TUNING_ODORS = 18
MIN_ROIS = 5  # named ROIs a fit is made from at least
RIDGE = 5.0
_TUKEY = 4.685  # (Tukey's biweight at 95% efficiency for Gaussian residuals)
# um: the least spread of the ROIs along an axis that A is held to the identity by (ROIs all on one plane have none in z)
_SPREAD = 1.0
_STRETCH = 2.0  # the most a fit may stretch or shrink the atlas along any direction: one that does more is not believed


def pairs(labels, names: dict, prior, prior_names, spacing_zyx):
    """What a fit is made from: the named ROIs whose glomerulus the atlas has on the fly's grid.

    Args:
        labels: the fly's ROIs ((Z, Y, X) labels).
        names: {label: name or None}, as they were named with ``prior``.
        prior, prior_names: the atlas on the fly's grid (`atlas.atlas_to_fly`) and its glomeruli.

    Returns:
        (the ROIs' labels, where the atlas expects each one's glomerulus, where each ROI is): the last two (n, 3), um
        (z, y, x): centres of mass, as the position term of naming has them.
    """
    sp = np.asarray(spacing_zyx, float)
    expected = atlas_centroids(prior, prior_names, sp)
    labs = sorted(l for l, n in names.items() if n and n in expected)
    if not labs:
        return [], np.zeros((0, 3)), np.zeros((0, 3))
    found = np.array(ndi.center_of_mass(labels > 0, labels, labs)) * sp
    return labs, np.array([expected[names[l]] for l in labs]), found


def _weights(dist: np.ndarray) -> np.ndarray:
    """Tukey's biweights of the pairs' distances (um), on the scale of their median (1 um at least)."""
    u = dist / (_TUKEY * max(1.4826 * float(np.median(dist)), 1.0))
    return np.where(u < 1, (1 - u ** 2) ** 2, 0.0)


def fit(expected, found, kind: str = "affine", ridge: float = RIDGE, min_rois: int = MIN_ROIS, n_iter: int = 10) -> dict | None:
    """The transformation of ``kind`` (of `KINDS`) taking ``expected`` to ``found`` ((n, 3) each, um): where the atlas
    expects glomeruli, and where the ROIs named as them are (`pairs`).

    Returns:
        None for fewer than ``min_rois`` pairs; else a dict with ``kind`` (the kind fit: "shift" where an affine map
        was asked for but could not be believed, as it would stretch the atlas over twofold), ``matrix`` (A, 3 x 3)
        and ``offset_um`` (b): a place x of the atlas on the fly's grid (um; z, y, x) belongs at A x + b; ``shift_um``
        (how far that moves the middle of the pairs), ``rois`` (how many pairs), ``weights`` (each one's in the fit, 0
        to 1: 0 for one taken to be named wrong) and ``before_um`` and ``after_um`` (the median distance of an ROI from
        where the atlas expects its glomerulus, before the move and after).
    """
    if kind not in KINDS:
        raise ValueError(f"align: one of {', '.join(KINDS)}, not {kind!r}")
    e, r = np.asarray(expected, float).reshape(-1, 3), np.asarray(found, float).reshape(-1, 3)
    if len(e) < max(min_rois, 1):
        return None
    free = np.array(KINDS[kind], bool)
    shift = np.median(r - e, 0)  # (the median offset, which wrong names move little)
    w0 = _weights(np.linalg.norm(r - e - shift, axis=1))
    A, b, w = np.eye(3), shift, w0
    for _ in range(n_iter if free.any() else 0):
        if w.sum() < 3:
            break
        c = np.average(e, axis=0, weights=w)
        X, Y = e - c, r - c
        var = np.maximum(np.average(X ** 2, axis=0, weights=w), _SPREAD ** 2)
        A, t = np.eye(3), np.zeros(3)
        for d in range(3):  # (each row of A, and of the shift, by weighted least squares)
            cols = np.flatnonzero(free[d])
            D = np.c_[X[:, cols], np.ones(len(X))]
            target = Y[:, d] - (0 if free[d, d] else X[:, d])
            # (held to the identity: by ``ridge``, and a little whatever it is, for what the ROIs say nothing of)
            P = np.diag([ridge * var[j] + 1e-9 for j in cols] + [1e-9])
            held = np.array([float(j == d) for j in cols] + [0.0])
            coef = np.linalg.solve(D.T @ (w[:, None] * D) + P, D.T @ (w * target) + P @ held)
            A[d, cols], t[d] = coef[:-1], coef[-1]
        b = c + t - A @ c
        new = _weights(np.linalg.norm(r - (e @ A.T + b), axis=1))
        done = np.allclose(new, w, atol=1e-4)
        w = new
        if done:
            break
    sv = np.linalg.svd(A, compute_uv=False)
    if not (np.isfinite(A).all() and np.linalg.det(A) > 0 and sv.max() <= _STRETCH and sv.min() >= 1 / _STRETCH):
        kind, A, b, w = "shift", np.eye(3), shift, w0
    mid = np.average(e, axis=0, weights=w) if w.sum() > 0 else e.mean(0)
    return {"kind": kind, "matrix": A.tolist(), "offset_um": b.tolist(), "shift_um": (A @ mid + b - mid).tolist(),
            "rois": len(e), "weights": w.tolist(),
            "before_um": float(np.median(np.linalg.norm(r - e, axis=1))),
            "after_um": float(np.median(np.linalg.norm(r - (e @ A.T + b), axis=1)))}


def apply(fitted: dict, points) -> np.ndarray:
    """Places of the atlas on the fly's grid ((n, 3), um) where ``fitted`` (`fit`'s) moves them."""
    return np.asarray(points, float) @ np.asarray(fitted["matrix"], float).T + np.asarray(fitted["offset_um"], float)


def move(prior, fitted: dict, spacing_zyx) -> np.ndarray:
    """The atlas on a fly's grid (``prior``: (glomeruli, Z, Y, X)) moved by ``fitted`` (`fit`'s): each map resampled
    (linearly; beyond the grid, a map is taken to go on as at its edge, so that a move along z does not empty the
    first or last plane)."""
    sp = np.asarray(spacing_zyx, float)
    inv = np.linalg.inv(np.asarray(fitted["matrix"], float))
    # (a voxel v of the moved atlas is the atlas at A^-1 (v * sp - b), in voxels)
    matrix = inv * sp[None, :] / sp[:, None]
    offset = -(inv @ np.asarray(fitted["offset_um"], float)) / sp
    return np.stack([ndi.affine_transform(np.asarray(m, np.float32), matrix, offset, order=1, mode="nearest") for m in prior])


def align(labels, names: dict, prior, prior_names, spacing_zyx, kind: str = "affine", **kw):
    """Fits a fly's atlas to the ROIs named with it and moves it (`pairs`, `fit`, `move`; ``kw``: more of `fit`'s
    arguments).

    Returns:
        (the atlas moved, `fit`'s dict); or (``prior`` as it is, None) where too few ROIs are named to fit it.
    """
    _, expected, found = pairs(labels, names, prior, prior_names, spacing_zyx)
    fitted = fit(expected, found, kind, **kw)
    if fitted is None:
        return prior, None
    return move(prior, fitted, spacing_zyx), fitted
