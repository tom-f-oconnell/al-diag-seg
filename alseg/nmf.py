"""Glomeruli a fly's ROIs miss, looked for in its movies by non-negative matrix factorization (NMF).

The model's ROIs are Cellpose's, drawn from the fly's average image and its odors' mean responses. A
glomerulus it left out may still show in the movies, as a place whose light changes together and unlike
its neighbours'. `find` looks for those, plane by plane (planes are ~10 um apart):

1. The plane's movie (`alseg.evidence.frames`: every time point of every trial less its baseline, each
   pixel over its noise, lightly blurred) is factorized as footprints (pixels) times time courses, both
   non-negative. Each ROI on the plane has a footprint, started at its mask and kept within
   ``reach_um`` of it, so that its own light, and what spills round it, is its own. Free footprints are
   added one at a time, each started where the others so far leave the most unexplained, only inside the
   hull of the ROIs (of the plane and the planes next to it: the lobe, as far as the ROIs show it) and
   ``seed_gap_um`` off them, and kept off the ROIs' pixels. All are then fit together (HALS).
2. A free footprint's place is its pixels where it gives more light than any other footprint and at least
   ``theta`` of its most, opened by a disk (slivers between ROIs go), the piece with its peak.
3. A place is a candidate glomerulus if it is big and round enough (``min_area_um2``,
   ``min_radius_um``: the largest circle in it), lies in the hull (``min_in_hull``), responds to odors
   reliably (``min_eta2``: the share of the variance of its trial responses that the odors explain) and
   not as a mixture of its neighbours (``min_novelty``: the share of its tuning that the ROIs within
   ``near_um``, weighted as best fits, leave unexplained). The last is what tells a glomerulus from the
   light round its neighbours, which is all the others together let through.
4. Candidates of adjacent planes that overlap and change together are one (as `alseg.segment.join_planes`
   joins masks), and one doing so with an ROI that has no mask on its plane is more of that ROI
   (``kind`` "extended"), else an ROI of its own ("added").

Masks that hold two glomeruli are not found so: a factorization of a mask's own pixels in two did not
tell them from masks over one (below). `find` proposes cutting those that `alseg.clean_trim.split` would
(by the per-trial dF/F maps, `alseg.evidence.load`), the part cut off a candidate too (``kind`` "split",
``of`` the ROI it is cut from).

`check` does all of it for a model's ROIs of a fly, cleans the candidates as the ROIs were
(`alseg.clean`) and names them: with the ROIs' names kept (the candidates named among the glomeruli
left), and with every ROI named again together. `name` and `subset` do the naming for some of the
candidates.

On the odor response database's fit 7 (its 4 held-out flies, 36 planes: 325 manual in-plane outlines,
of which its ROIs match 183 at IoU >= 0.5 and leave 57 under 0.2; the database's design.md has more):

* glomeruli added: 25 candidates, 8 of them on a manual outline the ROIs did not match (IoU >= 0.3
  with it; 5 at >= 0.5) and 17 where none is drawn. Outlines are drawn on some of a glomerulus' planes
  only, and not of every glomerulus, so those 17 are not known to be wrong. Without the tests of 3,
  901 places, 30 on such an outline: most are the light round the lobe and between ROIs.
* masks cut: of the 11 holding at least 40% of two manual outlines, 4 (3 with each part on its own
  outline), and 23 others (12 over one outline, 11 over none). A rank 2 NMF of each mask's pixels
  proposed parts for 5 of the 11 (2 right) and 159 others, by no measure apart.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

DEFAULTS = {
    "n_free": 25,  # free footprints tried per plane
    "sigma_um": 0.5,  # the movie's blur (s.d.) before it is factorized
    "seed_um": 1.5,  # the blur (s.d.) of what is left unexplained, where a free footprint is started at its most
    "seed_gap_um": 1.5,  # how far from the ROIs' masks a free footprint is started at least
    "win_um": 10.0,  # how far from where it started a free footprint may reach
    "reach_um": 2.0,  # how far from its mask an ROI's footprint may reach
    "iters": 30,  # passes of the fit of all footprints together
    "theta": 0.35,  # the share of a footprint's most that its place's pixels have at least
    "open_um": 1.0,  # the radius of the disk a place is opened by
    "min_masks": 3,  # the ROIs a plane must have to be looked at (fewer do not show where the lobe is)
    "min_area_um2": 20.0,
    "min_radius_um": 2.5,
    "min_in_hull": 0.8,
    "min_eta2": 0.5,
    "min_novelty": 0.5,
    "near_um": 4.0,  # the ROIs whose tunings a candidate's is compared with: those this near it
    "join_overlap": 0.2,  # as `alseg.segment.join_planes`: of the smaller mask's area
    "join_corr": 0.7,  # of the two masks' mean time courses
}


def _blur(x, sigma: float):
    """A (N, Y, X) tensor blurred in-plane by a Gaussian of s.d. ``sigma`` pixels (0 beyond the edges)."""
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(x[:, None], k.view(1, 1, -1, 1), padding=(r, 0))
    return F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))[:, 0]


def _over_noise(D):
    """A (T, Y, X) movie with each pixel over its noise (s.d., from the median difference of successive
    time points; at least the 20th percentile of the pixels', so that dark pixels are not blown up)."""
    sd = 1.4826 * (D[1:] - D[:-1]).abs().median(0).values / math.sqrt(2)
    floor = sd.flatten().kthvalue(max(1, int(0.2 * sd.numel()))).values
    return D / sd.clamp(min=max(float(floor), 1e-6))


def _update_c(X, A, C=None, passes: int = 3):
    """Time courses C (K, T) >= 0 of footprints A (P, K) in X (P, T): least squares, then HALS."""
    import torch

    G, B = A.T @ A, A.T @ X
    if C is None:
        eye = torch.eye(len(G), device=A.device, dtype=A.dtype)
        C = torch.linalg.solve(G + 1e-6 * G.diagonal().mean() * eye, B).clamp(min=0)
    for _ in range(passes):
        for k in range(len(G)):
            if G[k, k] > 0:
                C[k] = (C[k] + (B[k] - G[k] @ C) / G[k, k]).clamp(min=0)
    return C


def _update_a(X, A, C, S):
    """Footprints A (P, K) >= 0, each within its support (S, bool), given time courses C: a HALS pass."""
    G, H = C @ C.T, X @ C.T
    for k in range(A.shape[1]):
        if G[k, k] > 0:
            A[:, k] = ((A[:, k] + (H[:, k] - A @ G[:, k]) / G[k, k]).clamp(min=0)) * S[:, k]
    return A


def _factorize(X, shape, masks, seed_ok, free_ok, n_free, seed_px, win_px, reach_px, iters):
    """Factorizes a plane's movie X ((P, T) tensor, P = Y * X of ``shape``) as footprints times time
    courses: one footprint per mask of ``masks`` ((K0, Y, X) bool), then up to ``n_free`` free ones,
    each started at the pixel of ``seed_ok`` where the rest so far (blurred by ``seed_px``) is largest.

    Returns:
        (A: (K, Y, X) float32, C: (K, T) float32), numpy, the masks' footprints first.
    """
    import torch

    dev, (ny, nx), T = X.device, shape, X.shape[1]
    K0 = len(masks)
    seed_ok = torch.as_tensor(seed_ok, device=dev).reshape(-1).clone()
    free_ok = torch.as_tensor(free_ok, device=dev).reshape(-1)
    A = torch.as_tensor(masks.reshape(K0, -1).T.astype(np.float32), device=dev)
    S = torch.as_tensor(np.stack([ndi.distance_transform_edt(~m) <= reach_px for m in masks]).reshape(K0, -1).T, device=dev)
    C = _update_c(X, A)
    for _ in range(3):  # (the masks' footprints settle before the free ones take what they leave)
        A = _update_a(X, A, C, S)
        C = _update_c(X, A, C)
    R = X - A @ C
    yy, xx = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    for _ in range(n_free):
        Rb = _blur(R.T.reshape(T, ny, nx), seed_px).reshape(T, -1).T
        E = Rb.clamp(min=0).pow(2).sum(1) * seed_ok
        p = int(E.argmax())
        if float(E[p]) <= 0:
            break
        s = (((yy - p // nx) ** 2 + (xx - p % nx) ** 2) <= win_px ** 2).reshape(-1) & free_ok
        c = Rb[p].clamp(min=0)
        a = torch.zeros(len(X), device=dev)
        for _ in range(5):
            a = (R @ c / (c @ c + 1e-12)).clamp(min=0) * s
            if float(a @ a) == 0:
                break
            c = ((a @ R) / (a @ a)).clamp(min=0)
        if float(a @ a) == 0 or float(c @ c) == 0:
            seed_ok[p] = False
            continue
        R = R - a[:, None] * c[None]
        A, C, S = torch.cat([A, a[:, None]], 1), torch.cat([C, c[None]], 0), torch.cat([S, s[:, None]], 1)
    for _ in range(iters):
        C = _update_c(X, A, C)
        A = _update_a(X, A, C, S)
    C = _update_c(X, A, C)
    return A.T.reshape(-1, ny, nx).cpu().numpy(), C.cpu().numpy()


def _disk(r: float) -> np.ndarray:
    k = int(math.ceil(r))
    y, x = np.ogrid[-k:k + 1, -k:k + 1]
    return x * x + y * y <= r * r


def _piece(m: np.ndarray, through=None) -> np.ndarray:
    """The connected piece of ``m`` that holds the pixel ``through`` (if it is in one), else its largest."""
    cc, n = ndi.label(m)
    if n <= 1:
        return m
    if through is not None and cc[through] > 0:
        return cc == cc[through]
    return cc == 1 + int(np.argmax(np.bincount(cc.ravel())[1:]))


def _places(A, C, K0, plane, theta, open_px, min_px) -> list[np.ndarray]:
    """The free footprints' places (see the module's docstring, 2), those of at least ``min_px`` pixels."""
    contrib = A * np.sqrt((C ** 2).mean(1))[:, None, None]
    owner = contrib.argmax(0)
    out = []
    for k in range(K0, len(A)):
        a = A[k]
        if a.max() <= 0:
            continue
        m = (a >= theta * a.max()) & (owner == k) & (plane == 0)
        if open_px > 0:
            m = ndi.binary_opening(m, _disk(open_px))
        if m.sum() < min_px:
            continue
        m = ndi.binary_fill_holes(_piece(m, np.unravel_index(a.argmax(), a.shape))) & (plane == 0)
        if m.sum() >= min_px:
            out.append(m)
    return out


def _corr(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class _Trials:
    """A fly's trials, to make of a time course (over `alseg.evidence.frames`' time points) its trial
    responses (its mean over each trial's time points with odor), and of those its tuning (their mean per
    odor of a recording)."""

    def __init__(self, movies: dict):
        trial, on = np.asarray(movies["trial"]), np.asarray(movies["on"], bool)
        self.frames = [np.flatnonzero((trial == k) & on) for k in range(len(movies["trials"]))]
        ids = {t: i for i, t in enumerate(dict.fromkeys(movies["trials"]))}
        self.odor = np.array([ids[t] for t in movies["trials"]])

    def responses(self, trace) -> np.ndarray:
        return np.array([trace[f].mean() for f in self.frames])

    def tuning(self, trace) -> np.ndarray:
        r = self.responses(trace)
        return np.array([r[self.odor == i].mean() for i in np.unique(self.odor)])

    def eta2(self, trace) -> float:
        """The share of the variance of a time course's trial responses that the odors explain."""
        r = self.responses(trace)
        total = ((r - r.mean()) ** 2).sum()
        within = sum(((r[self.odor == i] - r[self.odor == i].mean()) ** 2).sum() for i in np.unique(self.odor))
        return float(1 - within / total) if total > 0 else 0.0


def _novelty(y: np.ndarray, others: list[np.ndarray]) -> float:
    """The share of a tuning ``y`` (its sum of squares) that the best non-negative mixture of ``others``
    leaves unexplained (1 with none)."""
    from scipy.optimize import nnls

    if not others:
        return 1.0
    _, r = nnls(np.stack(others, 1), y)
    return float(r ** 2 / (y @ y + 1e-12))


def hulls(labels: np.ndarray) -> np.ndarray:
    """Per plane of ``labels`` ((Z, Y, X)), the convex hull of the ROIs of it and of the planes next to
    it ((Z, Y, X) bool): the lobe there, as far as the ROIs show it."""
    from skimage.morphology import convex_hull_image

    out = np.zeros(labels.shape, bool)
    for z in range(len(labels)):
        m = (labels[max(0, z - 1):z + 2] > 0).any(0)
        if m.sum() > 2:
            out[z] = convex_hull_image(m)
    return out


def _join(labels: np.ndarray, found: list[dict], traces: dict, overlap: float, min_corr: float) -> None:
    """Gives each of ``found`` (in-plane places: ``z``, ``mask``, ``trace``) its ``group`` (those of
    one glomerulus share it) and ``kind``: "extended", with ``of`` (an ROI's label), where it is more of
    an ROI of ``labels`` on a plane next to its own; else "added". ``traces``: {(label, z): time course}
    of the ROIs' masks."""
    def alike(c, m, t):
        both = (c["mask"] & m).sum()
        return both / min(c["mask"].sum(), m.sum()) if both and _corr(c["trace"], t) >= min_corr else 0.0

    # more of an ROI: the best pairs first, an ROI taking one candidate per plane, and going on from it
    masks = {k: labels[k[1]] == k[0] for k in traces}
    on = {}  # label -> the planes it has a mask on
    for lab, z in traces:
        on.setdefault(lab, set()).add(z)
    while True:
        best = (overlap, None, None)
        for c in found:
            if c.get("of") is not None:
                continue
            for dz in (-1, 1):
                for (lab, z), m in masks.items():
                    if z == c["z"] + dz and c["z"] not in on[lab]:
                        f = alike(c, m, traces[lab, z])
                        if f >= best[0]:
                            best = (f, c, lab)
        if best[1] is None:
            break
        _, c, lab = best
        c["of"], c["kind"] = lab, "extended"
        masks[lab, c["z"]], traces[lab, c["z"]] = c["mask"], c["trace"]
        on[lab].add(c["z"])
    # the rest among themselves, each joined to one at most on the plane before
    rest = [c for c in found if c.get("of") is None]
    for i, c in enumerate(rest):
        c["group"], c["kind"] = i, "added"
    pairs = sorted(((alike(a, b["mask"], b["trace"]), i, j) for i, a in enumerate(rest) for j, b in enumerate(rest)
                    if b["z"] == a["z"] + 1), reverse=True)
    up, down = set(), set()
    for f, i, j in pairs:
        if f >= overlap and i not in down and j not in up:
            down.add(i)
            up.add(j)
            old, new = rest[j]["group"], rest[i]["group"]
            for c in rest:
                if c["group"] == old:
                    c["group"] = new


def find(labels: np.ndarray, fly: dict, movies: dict | None = None, gpu: bool | None = None, progress=None,
         split: bool = True, split_params: dict | None = None, **params) -> dict:
    """Looks in a fly's movies for glomeruli its ROIs miss (see the module's docstring).

    Args:
        labels: (Z, Y, X) int, its ROIs.
        fly: as `alseg.fly.load_fly` gives it: its ``spacing`` and, to read its movies and its trials'
            dF/F maps, ``dir`` (without those maps, masks are cut by its ``responses``).
        movies: its movies (`alseg.evidence.frames`), if read already.
        gpu: whether to factorize on the GPU (None: if PyTorch sees one).
        progress: called as ``progress(stage, done, total)``: "read" (recordings), then "factorize"
            (planes).
        split: also propose cutting the masks that `alseg.clean_trim.split` would, with
            ``split_params``.
        params: any of `DEFAULTS`.

    Returns:
        dict with ``labels``: ``labels`` with every candidate in ((Z, Y, X), their dtype: new ROIs
        numbered from above its largest label; more of an ROI under that ROI's label; a part cut off a
        mask under a new label, in place of that mask's pixels), ``candidate``: (Z, Y, X) int, the
        candidate each voxel is of (0: none), ``candidates``: of each, numbered from 1, its ``id``,
        ``kind`` ("added", "extended", "split"), ``label`` (in the labels returned), ``of`` (the ROI
        it is more of, or cut from; else None), ``planes``, ``area_um2`` (of all its planes) and
        what it was taken by: ``eta2`` and ``novelty`` (the least of its planes'), or of a part cut
        off, ``corr`` (of its and the rest's mean evidence); ``planes``: those factorized,
        ``params``, and ``rois``: of each in-plane mask of ``labels`` on those planes (``label``,
        ``z``), the same measures as a candidate is taken by (``area_um2``, ``radius_um``, ``eta2``,
        ``novelty`` against the other ROIs near it), to test the ROIs themselves by: with ``n_free``
        0 nothing is factorized, and those are all it gives (and the masks cut, with ``split``).

    Raises:
        ValueError: for a parameter that is not one, or a fly with no movie to read.
    """
    import torch

    from . import clean, clean_trim, evidence
    from .model import use_gpu

    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"nmf.find: no parameter {', '.join(sorted(unknown))} (it has {', '.join(DEFAULTS)})")
    p = {**DEFAULTS, **params}
    if movies is None:
        movies = evidence.frames(fly["dir"], labels.shape, progress=(lambda a, b: progress("read", a, b)) if progress else None)
    dff, trials = movies["dff"], _Trials(movies)
    dev = torch.device("cuda" if use_gpu(gpu) else "cpu")
    px = float(np.mean(fly["spacing"][1:]))
    um = lambda k: p[k] / px  # noqa: E731
    min_px = max(1, int(round(p["min_area_um2"] / px ** 2)))
    hull = hulls(labels)
    found, traces, planes, rois = [], {}, [], []
    for z, plane in enumerate(labels):
        labs = [int(l) for l in np.unique(plane) if l]
        if len(labs) >= max(1, p["min_masks"]) and hull[z].any():
            planes.append(z)
            # (only the hull and as far round it as a footprint started in it may reach)
            pad = int(math.ceil(um("win_um"))) + 2
            ys, xs = np.nonzero(hull[z])
            box = (slice(max(0, ys.min() - pad), ys.max() + pad + 1), slice(max(0, xs.min() - pad), xs.max() + pad + 1))
            pl, hl = plane[box], hull[z][box]
            D = _over_noise(torch.as_tensor(np.asarray(dff[(slice(None), z) + box], np.float32), device=dev))
            X = _blur(D, um("sigma_um")).reshape(len(D), -1).T.contiguous()
            del D
            masks = np.stack([pl == l for l in labs])
            mean = lambda m: X[torch.as_tensor(np.flatnonzero(m.ravel()), device=dev)].mean(0).cpu().numpy()  # noqa: E731
            for l, m in zip(labs, masks):
                traces[l, z] = mean(m)
            tunings = {l: trials.tuning(traces[l, z]) for l in labs}
            for l, m in zip(labs, masks):  # (the ROIs' own masks by the same tests)
                near = ndi.distance_transform_edt(~m) <= um("near_um")
                rois.append({"label": l, "z": z, "area_um2": float(m.sum() * px ** 2), "radius_um": float(ndi.distance_transform_edt(m).max() * px),
                             "eta2": trials.eta2(traces[l, z]),
                             "novelty": _novelty(tunings[l], [tunings[k] for k, q in zip(labs, masks) if k != l and (q & near).any()])})
            places = []
            if p["n_free"] > 0:
                A, C = _factorize(X, pl.shape, masks, hl & (ndi.distance_transform_edt(pl == 0) >= um("seed_gap_um")), pl == 0,
                                  p["n_free"], um("seed_um"), um("win_um"), um("reach_um"), p["iters"])
                places = _places(A, C, len(labs), pl, p["theta"], um("open_um"), min_px)
            for m in places:
                trace = mean(m)
                near = ndi.distance_transform_edt(~m) <= um("near_um")
                full = np.zeros(plane.shape, bool)
                full[box] = m
                c = {"z": z, "mask": full, "trace": trace, "area_um2": float(m.sum() * px ** 2),
                     "radius_um": float(ndi.distance_transform_edt(m).max() * px), "in_hull": float((m & hl).sum() / m.sum()),
                     "eta2": trials.eta2(trace),
                     "novelty": _novelty(trials.tuning(trace), [tunings[l] for l, q in zip(labs, masks) if (q & near).any()])}
                if (c["area_um2"] >= p["min_area_um2"] and c["radius_um"] >= p["min_radius_um"] and c["in_hull"] >= p["min_in_hull"]
                        and c["eta2"] >= p["min_eta2"] and c["novelty"] >= p["min_novelty"]):
                    found.append(c)
            del X
        if progress:
            progress("factorize", z + 1, len(labels))
    _join(labels, found, traces, p["join_overlap"], p["join_corr"])
    out, cand, cands = labels.copy(), np.zeros(labels.shape, np.int32), []
    nxt = int(labels.max()) + 1
    groups = {}
    for c in found:
        groups.setdefault((c["kind"], c["of"] if c["kind"] == "extended" else c["group"]), []).append(c)
    if split:
        clean.ensure_evidence(fly, say=lambda *_: None)
        cut, report = clean_trim.split(labels, fly, return_report=True, **(split_params or {}))
        for (lab, z), new in sorted(report["split"].items(), key=lambda kv: (kv[0][1], kv[0][0])):
            m = cut[z] == new
            groups["split", new] = [{"z": z, "mask": m, "kind": "split", "of": int(lab), "area_um2": float(m.sum() * px ** 2),
                                     "corr": float(report["stats"][lab, z]["corr"])}]
    for (kind, _), cs in sorted(groups.items(), key=lambda kv: (min(c["z"] for c in kv[1]), kv[0][0], kv[0][1])):
        if kind == "extended":
            label = cs[0]["of"]
        else:
            label, nxt = nxt, nxt + 1
        i = len(cands) + 1
        for c in cs:
            out[c["z"]][c["mask"]] = label
            cand[c["z"]][c["mask"]] = i
        cands.append({"id": i, "kind": kind, "label": int(label), "of": None if cs[0].get("of") is None else int(cs[0]["of"]),
                      "planes": sorted({c["z"] for c in cs}), "area_um2": float(sum(c["area_um2"] for c in cs)),
                      **({"corr": cs[0]["corr"]} if kind == "split" else
                         {"eta2": float(min(c["eta2"] for c in cs)), "novelty": float(min(c["novelty"] for c in cs))})})
    return {"labels": out, "candidate": cand, "candidates": cands, "planes": planes, "params": p, "rois": rois}


def subset(labels: np.ndarray, merged: np.ndarray, candidate: np.ndarray, chosen) -> np.ndarray:
    """``merged`` (`find`'s ``labels``) with only the candidates ``chosen`` (their ids) in: the voxels of
    the others (by ``candidate``) as in ``labels``, the ROIs they were found for."""
    return np.where(np.isin(candidate, list(chosen)), merged, labels)


def name(model, fly: dict, labels: np.ndarray, prior=None, prior_names=None, keep: dict | None = None):
    """Names a fly's ROIs as ``model`` names those it predicts (`alseg.model.Model.predict`'s "pred":
    with its atlas on the fly's grid, ``prior``, of the glomeruli ``prior_names``, if it has one).

    Args:
        fly: its ``responses`` (in the model's odors, or those of them it has: ``odors``) and ``spacing``.
        keep: {label: name or None} of ROIs that keep their names (`alseg.identity.assign_identities`'s
            ``fixed``).

    Returns:
        ({label: (name or None, cost, confidence)}, what the costs are made of), as
        `alseg.identity.assign_identities` with ``return_terms``.
    """
    from . import identity

    kw = {} if prior is not None else {"w_shape": 0, "w_layout": 0}
    # (a fly that lacks some of the model's odors: by those it has, as the model names its ROIs)
    template = model.template_of(fly) if hasattr(model, "template_of") else model.template
    return identity.assign_identities(fly["responses"], labels, template, fly["spacing"], prior, prior_names,
                                      max_cost=model.options.max_cost, return_terms=True, fixed=keep, **kw)


def check(model, fly: dict, labels: np.ndarray, names: dict, prior=None, steps=(), clean_params: dict | None = None,
          movies: dict | None = None, progress=None, named: np.ndarray | None = None, **params) -> dict:
    """Looks for glomeruli a model's ROIs of a fly miss (`find`), cleans the candidates as the ROIs were
    cleaned, and names them.

    Args:
        model: the `alseg.model.Model` whose ROIs they are (its template, and how it names).
        fly: as `alseg.fly.load_fly` gives it.
        labels: (Z, Y, X) its ROIs, and ``names``: {label: name or None}, as the model named them.
        prior: the model's atlas on the fly's grid (`alseg.atlas.atlas_to_fly`: (glomeruli, Z, Y, X)),
            if it has one.
        steps: the steps the ROIs were cleaned by (`alseg.clean.STEPS`), with ``clean_params``:
            applied to the ROIs with the candidates in, of which only the candidates' own pixels are
            taken (the ROIs stay as they are). Those that cannot run on the fly are left out, and a
            candidate left under `DEFAULTS`' ``min_area_um2`` is dropped.
        movies, params: as `find` takes them.
        progress: as `find`'s, then ("name", done, 2).
        named: the ROIs as the model named them ((Z, Y, X), the same labels), if ``labels`` are those
            cleaned since (`Model.predict` names Cellpose's ROIs, and the cleaned ones keep those
            names): ROIs are then named by these with the candidates in (`subset`), so that naming
            them all again with no candidate gives the names they have.

    Returns:
        `find`'s dict (its ``labels`` and candidates as cleaned), with ``cleaned`` (the steps
        applied) and ``names``: of every ROI of ``labels`` returned, {label: name or None},
        ``kept`` (the ROIs' names as given, the candidates named among the glomeruli left) and
        ``joint`` (all named again together); ``terms``: what each of the two namings cost
        (`alseg.identity.assign_identities`).
    """
    from . import clean

    res = find(labels, fly, movies, gpu=model.options.gpu, progress=progress, **params)
    merged, cand, px = res["labels"], res["candidate"], float(np.mean(fly["spacing"][1:]))
    res["cleaned"] = [s for s in clean.ordered(steps) if clean.usable(s, fly)] if res["candidates"] else []
    if res["cleaned"]:
        cleaned = clean.apply(merged, fly, res["cleaned"], clean_params)
        merged[(cand > 0) & (cleaned != merged)] = 0
        kept = []
        for c in res["candidates"]:
            own = (cand == c["id"]) & (merged == c["label"])
            if own.sum() * px ** 2 < res["params"]["min_area_um2"]:
                merged[cand == c["id"]] = labels[cand == c["id"]]
                cand[cand == c["id"]] = 0
                continue
            c["area_um2"], c["planes"] = float(own.sum() * px ** 2), [int(z) for z in np.flatnonzero(own.any((1, 2)))]
            kept.append(c)
        res["candidates"] = kept
    prior_names = list(model.atlas.names) if prior is not None else None
    res["names"], res["terms"] = {}, {}
    to_name = merged if named is None else subset(named, merged, cand, [c["id"] for c in res["candidates"]])
    for i, (key, keep) in enumerate((("kept", {int(l): names.get(int(l)) for l in np.unique(labels) if l}), ("joint", None))):
        ids, res["terms"][key] = name(model, fly, to_name, prior, prior_names, keep)
        res["names"][key] = {int(l): v[0] for l, v in ids.items()}
        if progress:
            progress("name", i + 1, 2)
    return res
