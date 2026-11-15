"""Cleaning steps that keep predicted ROIs off their neighbours by where pixels are, not what
they respond to: a margin, all round and from other ROIs (`margin`), or only the pixels Cellpose
is surest of (`core`).

Each takes a fly's labels ((Z, Y, X) ints) and the fly (as `alseg.fly.load_fly` gives it:
``spacing``, and for `core` ``cellprob``) and returns new labels of the same dtype, made plane
by plane (each plane's mask of an ROI on its own, as planes are several um apart), with pixels
only removed and every label kept, so that names carry over. Of a mask, only its largest
connected piece is kept. A mask that would fall under ``min_size_um2`` keeps instead its pixels
that best meet the step's rule, the fewest whose largest piece is that big (`_keep`), so no
ROI vanishes; a mask no bigger than that to begin with is left whole.

The defaults were picked on the odor response database's fit 7 (its 4 held-out flies, picked on
3 and scored on the 4th): of each step, the setting keeping the most area whose edges facing a
neighbouring ROI carry no more of it than the manual ROIs' do. On the 4 flies, the share of a
neighbour's tuning in an ROI's edge facing it is 0.30 as predicted and 0.15 manual; `margin`'s
defaults make it 0.13 keeping 97% of a mask's area (median), `core`'s 0.11 keeping 74%, and at
any area kept `margin` is the purer of the two.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi


def margin(labels: np.ndarray, fly: dict, erode_um: float = 0.0, gap_um: float = 2.0,
           min_size_um2: float = 10.0) -> np.ndarray:
    """Each in-plane mask without its pixels within ``erode_um`` of its own outline or within
    ``gap_um`` of another ROI's (as given, on its plane).

    Distances are from a pixel's centre to an outline, which runs half a pixel beyond the outermost
    pixels' centres: so 0.5 um takes one pixel of 0.5 um off, and where two ROIs touch they end
    2 * max(``erode_um``, ``gap_um``) apart. Outlines along the image's edge are not eroded.

    Args:
        labels: (Z, Y, X) ROIs.
        fly: with ``spacing`` ((z, y, x) um).
        erode_um: taken off all round, which alone keeps out blur from neighbours that got no
            ROI (which no score here sees); 0.5 um with the default gap lowers `leak` from 0.055
            to 0.036 (manual ROIs' 0.048) on fit 7's held-out flies, but keeps 80% of a mask's
            area, not 97%.
        gap_um: kept from other ROIs, on the side facing them (2 um: ROIs end about as far apart
            as manual ones, 2.0 um and 1.9 um median).
        min_size_um2: the least area a mask is left with (see the module's docstring): its
            pixels farthest inside both limits.

    Returns:
        (Z, Y, X) labels.
    """
    sy, sx = (float(s) for s in fly["spacing"][1:])
    half = (sy + sx) / 4
    pad = int(math.ceil(max(erode_um, gap_um, 0) / min(sy, sx))) + 2
    min_px = _pixels(min_size_um2, sy, sx)
    out = np.zeros_like(labels)
    for z, lab, sl in _masks(labels):
        box = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, labels.shape[1:]))
        crop = labels[z][box]
        m = crop == lab
        others = (crop > 0) & ~m
        score = _inside(m, (sy, sx)) - half - erode_um
        if others.any():
            score = np.minimum(score, _inside(~others, (sy, sx)) - half - gap_um)
        out[z][box][_keep(m, score, min_px)] = lab
    return out


def core(labels: np.ndarray, fly: dict, core_cellprob: float = -1.0, min_size_um2: float = 10.0) -> np.ndarray:
    """Each in-plane mask's pixels of a cell probability of at least ``core_cellprob``.

    A model trained on outlines eroded 1 px (fit 7's) is sure of little of its masks: at 0 they
    keep half their area, at +1 a fifth; and where two masks touch, the
    probability between them stays high, so at the default 9% of masks still touch another.

    Args:
        labels: (Z, Y, X) ROIs.
        fly: with ``spacing`` ((z, y, x) um) and ``cellprob``: Cellpose's cell probability map,
            (Z, Y, X) on the fly's grid (`alseg.segment.predict`'s, with ``return_cellprob``),
            in its logits (masks are of the pixels over ``Options.cellprob_threshold``, -2).
        core_cellprob: the least cell probability kept.
        min_size_um2: the least area a mask is left with (see the module's docstring): its
            pixels of the highest cell probability.

    Returns:
        (Z, Y, X) labels.

    Raises:
        ValueError: if the fly has no cell probability map, or one of another shape.
    """
    prob = fly.get("cellprob")
    if prob is None or np.shape(prob) != labels.shape:
        raise ValueError(f"{fly.get('name', 'the fly')}: needs Cellpose's cell probability map of its labels' shape "
                         f"{labels.shape} (segment.predict's, with return_cellprob), as 'cellprob'; it has "
                         f"{None if prob is None else np.shape(prob)}")
    sy, sx = (float(s) for s in fly["spacing"][1:])
    min_px = _pixels(min_size_um2, sy, sx)
    out = np.zeros_like(labels)
    for z, lab, sl in _masks(labels):
        m = labels[z][sl] == lab
        out[z][sl][_keep(m, prob[z][sl] - core_cellprob, min_px)] = lab
    return out


def _masks(labels):
    """(z, label, bounding box on plane z) of every in-plane mask."""
    for z in range(labels.shape[0]):
        for i, sl in enumerate(ndi.find_objects(labels[z])):
            if sl is not None:
                yield z, i + 1, sl


def _pixels(area_um2: float, sy: float, sx: float) -> int:
    return max(1, int(math.ceil(area_um2 / (sy * sx))))


def _inside(m: np.ndarray, sampling) -> np.ndarray:
    """Each pixel's distance (um) to the nearest pixel outside ``m`` (2D); inf if there is none."""
    if m.all():
        return np.full(m.shape, np.inf)
    return ndi.distance_transform_edt(m, sampling=sampling)


def _largest(m: np.ndarray) -> np.ndarray:
    cc, n = ndi.label(m)
    if n <= 1:
        return m
    return cc == 1 + int(np.argmax(np.bincount(cc.ravel())[1:]))


def _keep(m: np.ndarray, score: np.ndarray, min_px: int) -> np.ndarray:
    """Of mask ``m`` (2D), the largest connected piece of its pixels of ``score`` >= 0; if that has
    fewer than ``min_px`` pixels (or than ``m``, if it has fewer), the largest piece of the fewest
    best-scoring pixels whose largest piece has that many."""
    keep = _largest(m & (score >= 0))
    want = min(min_px, int(m.sum()))
    if keep.sum() >= want:
        return keep
    ranked = np.sort(score[m])[::-1]
    lo, hi = want, len(ranked)  # (the largest piece only grows as pixels are added)
    while lo < hi:
        mid = (lo + hi) // 2
        if _largest(m & (score >= ranked[mid - 1])).sum() >= want:
            hi = mid
        else:
            lo = mid + 1
    return _largest(m & (score >= ranked[lo - 1]))
