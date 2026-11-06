"""Fixed reference atlases: glomerulus labels from some other source (e.g. the Grabe et
al. 2016 in vivo atlas, or hemibrain ROIs), added to the atlas built from the training
flies.

A reference has labels but no odor responses, and any image it has comes from
a different preparation, so it is aligned to the training-fly atlas by its labels
rather than by image intensity: a linear fit (affine or similarity) to the centroids
of glomeruli named in both, then optionally a deformable (SyN) registration between the two sets of label
maps. The training-fly atlas is already registered to each new fly, so the reference
reaches new flies through it.

Reference format (as exported by ~/src/al_atlas): a folder with ``labels.tif``
((Z, Y, X) integer labels, voxel size in ImageJ metadata) and one ``.csv`` with
``label`` and ``glomerulus`` columns, and optionally ``side``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from .atlas import Atlas, _ants, _img
from .roi_io import _clean_name


@dataclass
class Reference:
    labels: np.ndarray  # (Z, Y, X) int
    names: dict[int, str]  # label -> glomerulus name, cleaned like ROI names (see `canonical`)
    spacing: tuple[float, float, float]  # (z, y, x) um
    side: str | None = None


def _tiff_spacing(path):
    with tifffile.TiffFile(path) as t:
        ij = t.imagej_metadata or {}
        page = t.pages[0]
        out = []
        for tag in ("YResolution", "XResolution"):
            num, den = page.tags[tag].value if tag in page.tags else (1, 1)
            out.append(den / num if num else 1.0)
        return (float(ij.get("spacing", 1.0)), *out)


def _table(folder) -> Path:
    tables = list(Path(folder).glob("*.csv"))
    if len(tables) != 1:
        raise ValueError(f"{folder}: expected one .csv of glomerulus names, found {len(tables)}")
    return tables[0]


def load_reference(folder, side: str | None = None, margin_um: float = 10.0) -> Reference:
    """Load a reference folder, keeping one side (e.g. "left") if given, cropped to
    the kept glomeruli plus ``margin_um``."""
    folder = Path(folder)
    table = _table(folder)
    with open(table, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("glomerulus")]
    if side is not None:
        rows = [r for r in rows if r.get("side") == side]
        if not rows:
            raise ValueError(f"{table}: no glomeruli with side {side!r}")
    spacing = _tiff_spacing(folder / "labels.tif")
    labels = tifffile.imread(folder / "labels.tif")
    names = {int(r["label"]): _clean_name(r["glomerulus"]) for r in rows}
    keep = np.isin(labels, list(names))
    lo = np.argwhere(keep).min(0)
    hi = np.argwhere(keep).max(0) + 1
    pad = np.ceil(margin_um / np.asarray(spacing)).astype(int)
    sl = tuple(slice(max(a - p, 0), b + p) for a, b, p in zip(lo, hi, pad))
    labels = np.where(keep, labels, 0)[sl].astype(np.int32)
    return Reference(labels, names, spacing, side)


def sides(folder) -> list[str | None]:
    """Sides listed in a reference folder's table ([None] if it has no side column)."""
    with open(_table(folder), newline="") as fh:
        found = sorted({r["side"] for r in csv.DictReader(fh) if r.get("side") and r.get("glomerulus")})
    return found or [None]


def _centroids(ref: Reference) -> dict[str, np.ndarray]:
    labs = sorted(ref.names)
    cents = ndi.center_of_mass(ref.labels > 0, ref.labels, labs)
    return {ref.names[l]: np.array(c) * ref.spacing for l, c in zip(labs, cents)
            if not np.isnan(c[0])}


def _atlas_centroids(atl: Atlas) -> dict[str, np.ndarray]:
    sp = np.asarray(atl.spacing)
    return {n: np.array(ndi.center_of_mass(p.astype(np.float32))) * sp
            for n, p in zip(atl.names, atl.prob) if p.max() > 0}


def _affine(src, dst):
    M, *_ = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)
    return M[:3].T, M[3]


def _similarity(src, dst):
    """Rotation (or reflection), one scale and a translation."""
    s0, d0 = src - src.mean(0), dst - dst.mean(0)
    U, S, Vt = np.linalg.svd(d0.T @ s0)
    A = U @ Vt * (S.sum() / (s0**2).sum())
    return A, dst.mean(0) - src.mean(0) @ A.T


_FITS = {"affine": _affine, "similarity": _similarity}


def _loo_error(fit, src, dst):
    """Median error predicting each point from a fit to the others."""
    err = []
    for i in range(len(src)):
        m = np.arange(len(src)) != i
        A, b = fit(src[m], dst[m])
        err.append(np.linalg.norm(src[i] @ A.T + b - dst[i]))
    return np.median(err)


def fit_transform(src: np.ndarray, dst: np.ndarray, n_reject: int = 2):
    """Least squares dst ~= src @ A.T + b (um), with A a full affine or a similarity
    transform, whichever predicts left-out points better (a similarity is more
    robust with few or noisy points, e.g. centroids from planes far apart). Points
    with residuals over 3x the median are dropped and the fit redone, up to
    ``n_reject`` times. Returns A, b, kept mask, and which kind of fit."""
    keep = np.ones(len(src), bool)
    for _ in range(n_reject + 1):
        kind = min(_FITS, key=lambda k: _loo_error(_FITS[k], src[keep], dst[keep]))
        A, b = _FITS[kind](src[keep], dst[keep])
        res = np.linalg.norm(src @ A.T + b - dst, axis=1)
        new = res <= max(3 * np.median(res[keep]), 1e-6)
        if new.sum() < 6 or np.array_equal(new, keep):
            break
        keep = new
    return A, b, keep, kind


def _fit(ref: Reference, atl: Atlas, ac=None):
    """Linear map from reference to atlas um, fit to glomeruli named in both. ``ac``:
    precomputed _atlas_centroids(atl). Returns (shared names, A, b, kept mask,
    residual per shared glomerulus, kind of fit)."""
    rc = _centroids(ref)
    ac = _atlas_centroids(atl) if ac is None else ac
    shared = sorted(set(rc) & set(ac))
    if len(shared) < 6:
        raise ValueError(f"only {len(shared)} glomeruli named in both the reference and the atlas")
    src, dst = np.array([rc[n] for n in shared]), np.array([ac[n] for n in shared])
    A, b, kept, kind = fit_transform(src, dst)
    return shared, A, b, kept, np.linalg.norm(src @ A.T + b - dst, axis=1), kind


def align_reference(ref: Reference, atl: Atlas, transform: str = "SyN", seed: int = 0, ac=None):
    """Resample the reference labels onto the atlas grid.

    transform: "Affine" (linear centroid fit only) or "SyN" (then deformable registration of
    the shared glomeruli's label maps). Returns (labels on the atlas grid, info).
    """
    shared, A, b, kept, res, kind = _fit(ref, atl, ac)

    # scipy maps output (atlas) voxel indices to input (reference) voxel indices
    sa, sr = np.asarray(atl.spacing), np.asarray(ref.spacing)
    Ainv = np.linalg.inv(A)
    mat = np.diag(1 / sr) @ Ainv @ np.diag(sa)
    off = -(Ainv @ b) / sr
    out = ndi.affine_transform(ref.labels, mat, off, output_shape=atl.template.shape, order=0)

    if transform != "Affine":
        fit = [n for n, k in zip(shared, kept) if k]
        out = _register_labels(out, atl, {n: l for l, n in ref.names.items()}, fit, seed)
    info = {"side": ref.side, "fit": kind, "n_shared": len(shared), "residual_median_um": float(np.median(res[kept])),
            "residual_max_um": float(res[kept].max()), "rejected": [n for n, k in zip(shared, kept) if not k],
            "mirrored": bool(np.linalg.det(A) < 0)}
    return out, info


def _register_labels(labels, atl, label_of, shared, seed):
    """SyN between label-map images: each shared glomerulus gets the same random
    intensity in both, so mean squares pulls matching glomeruli together."""
    ants = _ants()
    rng = np.random.default_rng(seed)
    vals = rng.uniform(1, 2, len(shared))
    idx = {n: k for k, n in enumerate(atl.names)}
    fixed = np.zeros(atl.template.shape, np.float32)
    lut = np.zeros(max(labels.max(), max(label_of.values())) + 1, np.float32)
    for v, n in zip(vals, shared):
        fixed += v * atl.prob[idx[n]].astype(np.float32)
        lut[label_of[n]] = v
    moving = ndi.gaussian_filter(lut[labels], 1.0)
    f, m = _img(fixed, atl.spacing), _img(moving, atl.spacing)
    reg = ants.registration(fixed=f, moving=m, type_of_transform="SyNOnly", syn_metric="meansquares")
    warped = ants.apply_transforms(fixed=f, moving=_img(labels.astype(np.float32), atl.spacing),
                                   transformlist=reg["fwdtransforms"], interpolator="genericLabel")
    return np.rint(warped.numpy()).astype(labels.dtype)


def add_to_atlas(atl: Atlas, ref: Reference, transform: str = "SyN", weight: float = 1.0,
                 smooth_um: float = 2.0, seed: int = 0, ac=None) -> tuple[Atlas, dict]:
    """Atlas with the reference's glomeruli added.

    Glomeruli also in the atlas are averaged in as if the reference were ``weight``
    more training flies. Glomeruli only in the reference get their own channels; the
    identity step can name ROIs after them from position and atlas overlap alone.
    ``smooth_um`` blurs the reference masks, which come from one fly and are only
    roughly aligned.
    """
    lab, info = align_reference(ref, atl, transform, seed, ac)
    sigma = smooth_um / np.asarray(atl.spacing)
    idx = {n: k for k, n in enumerate(atl.names)}
    counts = atl.counts if atl.counts is not None else np.ones(len(atl.names))
    prob = [p for p in atl.prob]
    names, counts = list(atl.names), list(counts)
    added = []
    objs = ndi.find_objects(lab)
    pad = np.ceil(4 * sigma).astype(int)
    for l, n in sorted(ref.names.items()):
        if l > len(objs) or objs[l - 1] is None:
            continue  # outside the atlas grid
        sl = tuple(slice(max(o.start - p, 0), o.stop + p) for o, p in zip(objs[l - 1], pad))
        q = np.zeros(lab.shape, np.float32)
        q[sl] = np.clip(ndi.gaussian_filter((lab[sl] == l).astype(np.float32), sigma), 0, 1)
        if n in idx:
            k = idx[n]
            prob[k] = ((counts[k] * prob[k].astype(np.float32) + weight * q)
                       / (counts[k] + weight)).astype(np.float16)
            counts[k] += weight
        else:
            prob.append(q.astype(np.float16))
            names.append(n)
            counts.append(weight)
            added.append(n)
    info["added"] = added
    return replace(atl, prob=np.stack(prob), names=names, counts=np.array(counts, float)), info


def canonical(ref: Reference, names) -> Reference:
    """The reference with each glomerulus named as in ``names`` (e.g. the atlas's: the
    database's) where they differ only in case (``DA4L`` -> ``DA4l``)."""
    folded = {n.casefold(): n for n in names}
    return replace(ref, names={l: folded.get(n.casefold(), n) for l, n in ref.names.items()})


def add_best_side(atl: Atlas, folder, side: str | None = None, **kw) -> tuple[Atlas, dict]:
    """add_to_atlas with the side (of a two-sided reference) whose centroids fit the
    atlas best, unless ``side`` is given. The reference's names are matched to the
    atlas's whatever their case (`canonical`)."""
    cands = [side] if side else sides(folder)
    ac = _atlas_centroids(atl)
    best = None
    for s in cands:
        ref = canonical(load_reference(folder, s), atl.names)
        try:
            *_, kept, res, _ = _fit(ref, atl, ac)
        except ValueError:
            continue
        r = float(np.median(res[kept]))
        if best is None or r < best[0]:
            best = (r, ref)
    if best is None:
        raise ValueError(f"{folder}: fewer than 6 glomeruli named in both the reference and the atlas")
    return add_to_atlas(atl, best[1], ac=ac, **kw)
