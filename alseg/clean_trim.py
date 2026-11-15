"""Cleaning ROIs by the tuning of their pixels: `trim` drops a mask's pixels that carry a neighbouring
glomerulus, `split` cuts masks holding two glomeruli in two.

Both work plane by plane (planes are ~10 um apart) on a fly's per-pixel evidence (`alseg.evidence`:
``fly["evidence"]``, (n, Z, Y, X); else its trial-mean ``responses``), with distances in um by its
``spacing`` (z, y, x).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _evidence(fly) -> np.ndarray:
    e = fly.get("evidence")
    return np.asarray(fly["responses"] if e is None else e)


def _masks(plane):
    """(label, slices) of each mask of a (Y, X) label image."""
    for i, sl in enumerate(ndi.find_objects(plane)):
        if sl is not None:
            yield i + 1, sl


def _box(sl, pad, shape):
    return tuple(slice(max(0, s.start - pad), min(n, s.stop + pad)) for s, n in zip(sl, shape))


def _smooth_within(E, m, sigma_px):
    """(n, Y, X) evidence smoothed by a Gaussian within the mask ``m`` (normalized by it), at the mask's
    pixels: (n_pixels, n)."""
    if sigma_px <= 0:
        return E[:, m].T
    w = ndi.gaussian_filter(m.astype(np.float32), sigma_px)
    s = ndi.gaussian_filter(np.where(m, E, 0).astype(np.float32), (0, sigma_px, sigma_px))
    return (s[:, m] / w[m]).T


def _unit(V):
    """Rows of ``V`` less their means, unit-normalized (so that dot products are correlations)."""
    V = V - V.mean(-1, keepdims=True)
    return V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-8)


def _nnls2(V, c, n):
    """Per row v of ``V`` (p, k), the non-negative least squares weights (a, b) of ``v = a c + b n``."""
    cc, nn, cn = c @ c, n @ n, c @ n
    rc, rn = V @ c, V @ n
    det = cc * nn - cn * cn
    if det > 1e-12 * cc * nn:
        a, b = (nn * rc - cn * rn) / det, (cc * rn - cn * rc) / det
    else:
        a, b = np.full(len(V), -1.0), np.full(len(V), -1.0)
    bad = (a < 0) | (b < 0)
    # on the boundary: whichever of a alone, b alone fits better
    a1, b1 = np.maximum(rc, 0) / cc, np.maximum(rn, 0) / nn
    use_a = a1 * rc >= b1 * rn
    a = np.where(bad, np.where(use_a, a1, 0), a)
    b = np.where(bad, np.where(use_a, 0, b1), b)
    return a, b


def _band_cut(X, d, px, own, nb, max_share, max_depth_um, min_band_px):
    """Pixels to drop (bool over the rows of ``X``, (p, n) evidence) facing a neighbour: bands of pixels
    1 px wide by distance ``d`` (um) to it (the first of its pixels next to it), cut from the nearest out
    while the band's mean carries the neighbour's share over ``max_share``, at most ``max_depth_um``
    worth of bands (rounded up to whole pixels). Bands of fewer than ``min_band_px`` pixels are judged
    with the next."""
    band = np.floor(d / px + 1e-6).astype(int) - 1
    n_bands = int(np.ceil(max_depth_um / px))
    drop = np.zeros(len(X), bool)
    lo = 0
    while lo < n_bands:
        hi = lo
        sel = band == lo
        while sel.sum() < min_band_px and hi + 1 < n_bands:
            hi += 1
            sel |= band == hi
        if sel.sum() < min_band_px:
            break
        a, b = _nnls2(X[sel].mean(0)[None], own, nb)
        if b[0] <= max_share * (a[0] + b[0]):
            break
        drop |= sel
        lo = hi + 1
    return drop


def _largest(m):
    cc, k = ndi.label(m)
    if k <= 1:
        return m
    return cc == 1 + np.argmax(ndi.sum(m, cc, range(1, k + 1)))


def _references(E, plane, sp, core_um, min_core_px):
    """Each mask's core (pixels over ``core_um`` from its edge, or None if under ``min_core_px``) and
    reference vector (its core's mean evidence, else its whole mask's), by label."""
    cores, refs = {}, {}
    for lab, sl in _masks(plane):
        m = plane[sl] == lab
        core = ndi.distance_transform_edt(np.pad(m, 1), sampling=sp)[1:-1, 1:-1] > core_um
        cores[lab] = (sl, core) if core.sum() >= min_core_px else None
        e = E[(slice(None),) + sl]
        refs[lab] = e[:, core].mean(1) if cores[lab] is not None else e[:, m].mean(1)
    return cores, refs


def trim(labels: np.ndarray, fly: dict, max_share: float = 0.1, max_depth_um: float = 3.5, reach_um: float = 2.0,
         mode: str = "band", core_um: float = 1.5, min_core_px: int = 8, min_band_px: int = 4,
         max_loss: float = 0.5, min_corr_pct: float = 0.0, sigma_um: float = 0.5, return_report: bool = False):
    """Each in-plane mask without its pixels that carry a neighbouring glomerulus more than its own.

    How much of a pixel's evidence is a neighbour's is the purity yardstick's regression:
    non-negative least squares ``evidence = a own core + b neighbour's core``, the neighbour's share
    ``b / (a + b)``. A mask's core is its pixels over ``core_um`` from its edge (if at least
    ``min_core_px``; a neighbour without one stands for itself by its whole mask). Masks without a core
    are left as they are. For each other mask on the plane within ``reach_um``:

    * ``mode="band"``: the mask's pixels are put in bands 1 px wide by their distance to the neighbour,
      and bands are cut from the nearest out while a band's mean evidence carries the neighbour's share
      over ``max_share``, at most ``max_depth_um`` worth of bands (rounded up to whole pixels; bands of
      under ``min_band_px`` pixels are judged with the next one out). Band means are of many pixels,
      so the shares are hardly noisy.
    * ``mode="pixel"``: each pixel outside the core within ``reach_um`` of the neighbour is judged on its
      own, its evidence smoothed within the mask (Gaussian of ``sigma_um``), and dropped where its share
      is over ``max_share``. Per-pixel shares are noisy (on fit 7's held-out flies, 11% of core pixels'
      shares were over 0.2), so this cuts less cleanly than "band" for the same area.

    With ``min_corr_pct`` > 0, pixels outside the core (smoothed as in "pixel") that correlate with
    their core's mean worse than that percentile of its own pixels do are dropped too, for neighbours
    that got no ROI. Off by default: edge pixels are noisier than cores, so even 1% took 8% of the
    masks' area on fit 7's held-out flies without cleaning their edges by either yardstick.

    What is left keeps its largest connected piece, holes filled. A mask that would lose more than
    ``max_loss`` of its area is left whole and flagged (a candidate for `split` or redrawing). Pixels
    are only removed; labels are kept.

    Defaults picked on fit 7's 4 held-out flies (leaving each out in turn, evidence of all recordings'
    trials): the neighbour's share of the edge facing it 0.30 -> 0.10 (manual ROIs 0.15), median area
    kept 92%, no mask lost.

    Args:
        labels: (Z, Y, X) int ROIs (joined across planes or each plane's own).
        fly: the fly (`alseg.fly.load_fly`): its ``spacing``, and ``evidence`` (`alseg.evidence.load`)
            or else ``responses``, (n, Z, Y, X).
        return_report: also return a report: ``flagged`` {label: [planes]} left whole, ``kept``
            {(label, z): fraction of the mask's area kept}, ``no_core`` [(label, z)] left as they are
            for want of a core.

    Returns:
        The trimmed labels (as ``labels``), and the report if ``return_report``.

    Raises:
        ValueError: if ``mode`` is neither.
    """
    if mode not in ("band", "pixel"):
        raise ValueError(f"trim mode {mode!r}: not 'band' or 'pixel'")
    E = _evidence(fly)
    sp = tuple(float(s) for s in fly["spacing"][1:])
    sigma_px = sigma_um / np.mean(sp)
    pad = int(np.ceil(max(reach_um, max_depth_um) / min(sp))) + 1
    out = labels.copy()
    report = {"flagged": {}, "kept": {}, "no_core": []}
    for z, plane in enumerate(labels):
        if not plane.any():
            continue
        Ez = E[:, z]
        cores, refs = _references(Ez, plane, sp, core_um, min_core_px)
        for lab, sl in _masks(plane):
            if cores[lab] is None:
                report["no_core"].append((lab, z))
                continue
            box = _box(sl, pad, plane.shape)
            P = plane[box]
            m = P == lab
            core = np.zeros_like(m)
            core[tuple(slice(s.start - b.start, s.stop - b.start) for s, b in zip(sl, box))] = cores[lab][1]
            Eb = Ez[(slice(None),) + box]
            V = _smooth_within(Eb, m, sigma_px) if mode == "pixel" or min_corr_pct > 0 else None
            own = refs[lab]
            drop = np.zeros(int(m.sum()), bool)
            near = ndi.distance_transform_edt(~m, sampling=sp) <= reach_um
            for nb in np.unique(P[near & (P != lab)]):
                if nb == 0:
                    continue
                d = ndi.distance_transform_edt(P != nb, sampling=sp)[m]
                if mode == "band":
                    drop |= _band_cut(Eb[:, m].T, d, min(sp), own, refs[int(nb)], max_share, max_depth_um, min_band_px)
                    continue
                test = (d <= reach_um) & ~core[m]
                if not test.any():
                    continue
                a, b = _nnls2(V[test], own, refs[int(nb)])
                share = b / np.maximum(a + b, 1e-12)
                drop[np.flatnonzero(test)[share > max_share]] = True
            if min_corr_pct > 0:
                r = _unit(V) @ _unit(own)
                thr = np.percentile(r[core[m]], min_corr_pct)
                drop |= (r < thr) & ~core[m]
            keep = np.zeros_like(m)
            keep[m] = ~drop
            keep = _largest(ndi.binary_fill_holes(keep) & m)
            frac = keep.sum() / m.sum()
            if frac < 1 - max_loss:
                report["flagged"].setdefault(int(lab), []).append(z)
                frac = 1.0
            else:
                out[z][box][m & ~keep] = 0
            report["kept"][int(lab), z] = float(frac)
    return (out, report) if return_report else out


def _two_clusters(U, n_iter: int = 30):
    """2-means of unit rows ``U`` (p, n) by correlation, started from the sign of the first principal
    component: bool (p,), True for the second cluster."""
    X = U - U.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    lab = X @ vt[0] > 0
    for _ in range(n_iter):
        if lab.all() or not lab.any():
            break
        c0, c1 = _unit(U[~lab].mean(0)), _unit(U[lab].mean(0))
        new = U @ c1 > U @ c0
        if (new == lab).all():
            break
        lab = new
    return lab


def _coherence(U, mean):
    return float((U @ _unit(mean)).mean())


def split(masks: np.ndarray, fly: dict, max_corr: float = 0.5, min_area_um2: float = 10.0, min_connected: float = 0.8,
          edge_um: float = 1.0, center: bool = True, sigma_um: float = 0.5, min_gain: float | None = None,
          min_coherence: float | None = 0.5, return_report: bool = False):
    """Each in-plane mask cut in two where its pixels' evidence falls into two places tuned unlike.

    Per mask, the evidence is first ``center``ed: each value (trial) less its mean over the plane's
    other masks (if any), so that what every glomerulus shares (the AL's response to an odor, a trial
    stronger than the next) does not make two glomeruli look alike. Then its interior (pixels
    over ``edge_um`` from its edge, so that its neighbours' light at the edge does not make a ring of
    it; the whole mask if the interior is too small), evidence smoothed within it (Gaussian of
    ``sigma_um``), is split into 2 clusters by correlation (2-means). The clusters' largest connected
    pieces become the parts, the rest of the mask going to the nearer. The mask is split where:

    * each cluster's largest piece holds at least ``min_connected`` of it (the clusters are places,
      not noise);
    * each part is at least ``min_area_um2``;
    * the parts' mean evidence correlates under ``max_corr``;
    * if ``min_gain`` is given, each part's coherence (mean correlation of its pixels with its mean)
      beats the whole mask's by at least that (off by default: it did not tell merges from single
      glomeruli).

    The larger part keeps the mask's label; the other gets a new one (above every label in
    ``masks``). Masks not split whose interior's coherence is under ``min_coherence`` (None: none)
    are flagged. Masks under twice ``min_area_um2`` are left as they are.

    On fit 7's 4 held-out flies (evidence of all recordings' trials), of the 11 predicted in-plane
    masks holding >= 40% of two manual ROIs, 4 were split, each part on its own manual ROI; of the
    266 holding one, 12 were split, the parts cut off lying 89% outside any drawn ROI (an undrawn
    glomerulus, or none), and 1 of the 81 lying >= 80% in their manual ROI. The 7 merges not split
    are tuned alike (their parts correlate 0.55-0.96) and as coherent as single glomeruli, so neither
    test, nor ``min_coherence``, picks them out.

    Args:
        masks: (Z, Y, X) int, each plane's own masks (Cellpose's, before `alseg.segment.join_planes`),
            or ROIs joined across planes (each in-plane mask is split on its own).
        fly: as for `trim`.
        return_report: also return a report: ``split`` {(label, z): new label}, ``flagged`` {label:
            [planes]} (incoherent, not split), and ``stats`` {(label, z): the tests' values: ``corr``,
            ``gain``, ``connected``, ``areas`` (um^2), ``coherence``}.

    Returns:
        The masks (as ``masks``), and the report if ``return_report``.
    """
    E = _evidence(fly)
    sp = tuple(float(s) for s in fly["spacing"][1:])
    px_area = sp[0] * sp[1]
    sigma_px = sigma_um / np.mean(sp)
    out = masks.copy()
    nxt = int(masks.max()) + 1
    report = {"split": {}, "flagged": {}, "stats": {}}
    for z, plane in enumerate(masks):
        if not plane.any():
            continue
        Ez = E[:, z]
        total, count = Ez[:, plane > 0].sum(1, dtype=np.float64), int((plane > 0).sum())
        for lab, sl in _masks(plane):
            box = _box(sl, 1, plane.shape)
            m = plane[box] == lab
            if m.sum() * px_area < 2 * min_area_um2:
                continue
            Eb = Ez[(slice(None),) + box]
            if center and count > m.sum():
                # less the mean of the plane's other masks (its own would make its halves unlike)
                other = (total - Eb[:, m].sum(1, dtype=np.float64)) / (count - m.sum())
                Eb = Eb - other.astype(np.float32)[:, None, None]
            inner = ndi.distance_transform_edt(np.pad(m, 1), sampling=sp)[1:-1, 1:-1] > edge_um if edge_um > 0 else m
            if inner.sum() * px_area < 2 * min_area_um2:
                inner = m
            U = _unit(_smooth_within(Eb, inner, sigma_px))
            whole = _coherence(U, Eb[:, inner].mean(1))
            c = np.zeros_like(m)
            c[inner] = _two_clusters(U)
            st = dict(coherence=whole, connected=0.0, areas=(0.0, 0.0), corr=np.nan, gain=np.nan)
            if c.any() and (inner & ~c).any():
                p1 = _largest(c)
                p2 = _largest(inner & ~p1)
                st["connected"] = float(min(p1.sum() / c.sum(), p2.sum() / (inner & ~c).sum()))
                t1, t2 = Eb[:, p1].mean(1), Eb[:, p2].mean(1)
                st["corr"] = float(_unit(t1) @ _unit(t2))
                st["gain"] = min(_coherence(U[p1[inner]], t1), _coherence(U[p2[inner]], t2)) - whole
                rest = m & ~p1 & ~p2
                if rest.any():
                    closer = ndi.distance_transform_edt(~p1) <= ndi.distance_transform_edt(~p2)
                    p1, p2 = p1 | (rest & closer), p2 | (rest & ~closer)
                st["areas"] = (float(p1.sum() * px_area), float(p2.sum() * px_area))
            report["stats"][int(lab), z] = st
            if (st["connected"] >= min_connected and min(st["areas"]) >= min_area_um2 and st["corr"] < max_corr
                    and (min_gain is None or st["gain"] >= min_gain)):
                small = p2 if p2.sum() <= p1.sum() else p1
                out[z][box][small] = nxt
                report["split"][int(lab), z] = nxt
                nxt += 1
            elif min_coherence is not None and whole < min_coherence:
                report["flagged"].setdefault(int(lab), []).append(z)
    return (out, report) if return_report else out
