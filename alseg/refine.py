"""Use odor tuning to tighten ROIs, and measure crosstalk between neighbors.

Distances are in um and use each fly's voxel size (``spacing``, (z, y, x)), so ROIs
only grow into another plane when the planes are closer than the growth distance.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels


def _voxel_vectors(responses):
    """(n_odors, Z, Y, X) -> (Z, Y, X, n_odors), each voxel's vector unit-normalized."""
    v = np.moveaxis(responses, 0, -1).astype(np.float32)
    v = v - v.mean(-1, keepdims=True)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def roi_tuning(responses: np.ndarray, labels: np.ndarray, core_erode: int = 1) -> dict[int, np.ndarray]:
    """Median 24-odor tuning vector of each ROI's core (eroded in-plane)."""
    out = {}
    for lab in np.unique(labels):
        if lab == 0:
            continue
        m = labels == lab
        core = ndi.binary_erosion(m, structure=_inplane(core_erode)) if core_erode else m
        if core.sum() < 5:
            core = m
        out[int(lab)] = np.median(responses[:, core], axis=1)
    return out


def _inplane(r):
    """Structuring element that erodes in x/y only (r pixels)."""
    return _disk(r)[None]


def _disk(r):
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r


def neighbors(labels: np.ndarray, gap_um: float = 3.0, spacing=(1.0, 1.0, 1.0)) -> dict[int, set[int]]:
    """Pairs of ROIs within ``gap_um`` of each other (ROIs on adjacent planes count
    as neighbors if they overlap after growing in-plane)."""
    nb: dict[int, set[int]] = {int(l): set() for l in np.unique(labels) if l}
    grown = expand_labels(labels, gap_um, spacing=spacing)
    for lab in nb:
        ring = ndi.binary_dilation(grown == lab) & (grown != lab) & (grown > 0)
        nb[lab] = set(int(x) for x in np.unique(grown[ring]))
    return nb


def refine_by_tuning(
    responses: np.ndarray,
    labels: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
    grow_um: float = 2.0,
    min_corr: float = 0.3,
    margin: float = 0.05,
    min_rel_amp: float = 0.5,
    min_size_um3: float = 20.0,
) -> np.ndarray:
    """Re-assign voxels to the ROI whose tuning they match, and drop ambiguous ones.

    1. Each ROI is allowed to grow by ``grow_um`` (candidates).
    2. A candidate voxel is kept for ROI A only if its tuning correlates with
       A's core tuning at >= ``min_corr`` and beats every neighbor's
       correlation by ``margin``. Otherwise it is dropped (left unassigned),
       so mixed border voxels do not contaminate either trace.
    3. A voxel must also respond with at least ``min_rel_amp`` of its ROI's
       core amplitude. Correlation ignores amplitude, so without this, dim
       voxels that only see blurred-in signal (e.g. out of focus in z) pass.
    """
    tun = roi_tuning(responses, labels)
    labs = np.array(sorted(tun))
    T = np.stack([tun[l] for l in labs])
    T = T - T.mean(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8

    cand = expand_labels(labels, grow_um, spacing=spacing)
    region = cand > 0
    V = _voxel_vectors(responses)[region]  # (n, n_odors)
    raw = np.moveaxis(responses, 0, -1)[region]
    amp = np.linalg.norm(raw - raw.mean(1, keepdims=True), axis=1)
    core_amp = np.array([np.linalg.norm(tun[l] - tun[l].mean()) for l in labs])
    C = V @ T.T  # correlation with each ROI's tuning

    own_idx = np.searchsorted(labs, cand[region])
    nb = neighbors(labels, grow_um + max(spacing[1:]), spacing)
    allowed = np.zeros((len(labs), len(labs)), bool)
    for i, l in enumerate(labs):
        allowed[i, i] = True
        for n in nb[l]:
            if n in tun:
                allowed[i, np.searchsorted(labs, n)] = True

    n = len(V)
    own = C[np.arange(n), own_idx]
    rival = np.where(allowed[own_idx], C, -np.inf)
    rival[np.arange(n), own_idx] = -np.inf
    best_rival = rival.max(1)
    best_rival_idx = rival.argmax(1)

    out_vals = np.zeros(n, labels.dtype)
    bright = amp >= min_rel_amp * core_amp[own_idx]
    keep = bright & (own >= min_corr) & (own >= best_rival + margin)
    out_vals[keep] = labs[own_idx[keep]]
    # a voxel inside A that clearly belongs to neighbor B moves to B
    rival_bright = amp >= min_rel_amp * core_amp[best_rival_idx]
    move = (~keep) & rival_bright & (best_rival >= min_corr) & (best_rival >= own + margin)
    out_vals[move] = labs[best_rival_idx[move]]

    out = np.zeros_like(labels)
    out[region] = out_vals
    return _clean_fragments(out, min_size_um3 / np.prod(spacing))


def _clean_fragments(out, min_size):
    """Keep the largest 3D component of each ROI; drop ROIs below min_size."""
    for lab, sl in _objects(out):
        m = out[sl] == lab
        cc, k = ndi.label(m)
        if k > 1:
            sizes = ndi.sum(m, cc, range(1, k + 1))
            out[sl][m & (cc != (np.argmax(sizes) + 1))] = 0
        if (out[sl] == lab).sum() < min_size:
            out[sl][out[sl] == lab] = 0
    return out


def _objects(labels):
    for i, sl in enumerate(ndi.find_objects(labels)):
        if sl is not None:
            yield i + 1, sl


def refine_with_prior(
    responses: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str | None],
    prior: np.ndarray,
    prior_names: list[str],
    spacing=(1.0, 1.0, 1.0),
    w_tuning: float = 1.0,
    w_prior: float = 0.5,
    w_smooth: float = 0.15,
    z_smooth: float = 0.5,
    min_corr: float = 0.3,
    min_rel_amp: float = 0.5,
    grow_um: float = 2.0,
    n_iter: int = 5,
    min_size_um3: float = 20.0,
) -> np.ndarray:
    """Refinement as energy minimization, with an atlas prior.

    Each candidate voxel x gets one of: unassigned (0), its ROI, or an
    adjacent ROI. The cost of giving x to ROI l is

        w_tuning * (1 - corr(r_x, tuning_l))       odor tuning match
      + w_prior  * -log(p_l(x) + eps)              atlas: is l expected here?
      + w_smooth * (# 6-neighbors with another label; z-neighbors x z_smooth,
                    times the xy/z voxel size ratio when z is coarser)

    and leaving x unassigned costs the same as an ROI with corr = min_corr
    and a neutral prior p = 0.5. So a voxel where the atlas strongly expects
    glomerulus l needs less tuning evidence to join it, and a voxel the atlas
    says l should not reach needs more. Voxels dimmer than ``min_rel_amp``
    of l's core cannot join l. Solved by iterated conditional modes (ICM),
    starting from the best per-voxel label.

    names : {label: glomerulus name or None}, e.g. from identity.assign_identities.
        ROIs without a name get the neutral prior.
    prior : (K, Z, Y, X) atlas warped into this fly (atlas.atlas_to_fly),
        rows ordered as ``prior_names``.
    """
    eps = 0.05
    tun = roi_tuning(responses, labels)
    labs = np.array(sorted(tun))
    K = len(labs)
    T = np.stack([tun[l] for l in labs])
    T = T - T.mean(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8

    cand = expand_labels(labels, grow_um, spacing=spacing)
    region = cand > 0
    V = _voxel_vectors(responses)[region]
    raw = np.moveaxis(responses, 0, -1)[region]
    amp = np.linalg.norm(raw - raw.mean(1, keepdims=True), axis=1)
    core_amp = np.array([np.linalg.norm(tun[l] - tun[l].mean()) for l in labs])
    n = len(V)

    neutral = -np.log(0.5 + eps)
    U = np.empty((K + 1, n), np.float32)  # row 0 = unassigned
    U[0] = w_tuning * (1 - min_corr) + w_prior * neutral
    U[1:] = w_tuning * (1 - (V @ T.T).T)
    pidx = {nm: i for i, nm in enumerate(prior_names)}
    for k, l in enumerate(labs):
        nm = names.get(int(l))
        if nm in pidx:
            U[k + 1] += w_prior * -np.log(prior[pidx[nm]][region] + eps)
        else:
            U[k + 1] += w_prior * neutral
        U[k + 1][amp < min_rel_amp * core_amp[k]] = np.inf

    # only own ROI and adjacent ROIs are allowed per voxel
    own_k = np.searchsorted(labs, cand[region]) + 1
    nb = neighbors(labels, grow_um + max(spacing[1:]), spacing)
    allowed = np.zeros((K + 1, K + 1), bool)
    allowed[:, 0] = True
    for k, l in enumerate(labs):
        allowed[k + 1, k + 1] = True
        for m in nb[int(l)]:
            if m in tun:
                allowed[k + 1, np.searchsorted(labs, m) + 1] = True
    U[~allowed[own_k].T] = np.inf

    # 6-neighborhood, z-neighbors down-weighted for anisotropic voxels
    kern = np.zeros((3, 3, 3), np.float32)
    kern[1, 0, 1] = kern[1, 2, 1] = kern[1, 1, 0] = kern[1, 1, 2] = 1.0
    kern[0, 1, 1] = kern[2, 1, 1] = z_smooth * min(1.0, max(spacing[1:]) / spacing[0])
    total = kern.sum()

    vals = np.concatenate([[0], labs])
    L = np.zeros_like(labels)
    L[region] = vals[U.argmin(0)]
    for _ in range(n_iter):
        cost = U.copy()
        for k, v in enumerate(vals):
            same = ndi.convolve((L == v).astype(np.float32), kern, mode="constant")[region]
            cost[k] += w_smooth * (total - same)
        new = vals[cost.argmin(0)]
        changed = (new != L[region]).sum()
        L[region] = new
        if changed == 0:
            break
    return _clean_fragments(L, min_size_um3 / np.prod(spacing))


# ---------------------------------------------------------------- metrics


def crosstalk_report(responses: np.ndarray, labels: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict:
    """Signal-isolation metrics for a set of ROIs.

    coherence[A]   mean correlation of A's voxels with A's mean tuning
                   (high = homogeneous ROI)
    neighbor_corr  {(A, B): corr of A's and B's mean tuning} for adjacent
                   pairs. Compare against manual ROIs: lower or equal is good.
    leak[(A, B)]   least-squares weight of B's tuning when explaining A's
                   border voxels as a*A + b*B. Near 0 = A's border is clean.
    """
    V = _voxel_vectors(responses)
    mean_t = {}
    coherence = {}
    for lab in np.unique(labels):
        if lab == 0:
            continue
        m = labels == lab
        t = responses[:, m].mean(1)
        mean_t[int(lab)] = t
        tc = (t - t.mean()) / (np.linalg.norm(t - t.mean()) + 1e-8)
        coherence[int(lab)] = float((V[m] @ tc).mean())

    nb = neighbors(labels, spacing=spacing)
    neighbor_corr, leak = {}, {}
    for a, bs in nb.items():
        border = labels == a
        border = border & ~ndi.binary_erosion(border, structure=_inplane(1))
        ya = responses[:, border].mean(1)
        for b in bs:
            if a < b:
                neighbor_corr[(a, b)] = float(np.corrcoef(mean_t[a], mean_t[b])[0, 1])
            X = np.stack([mean_t[a], mean_t[b]], 1)
            coef, *_ = np.linalg.lstsq(X, ya, rcond=None)
            leak[(a, b)] = float(coef[1] / (abs(coef[0]) + abs(coef[1]) + 1e-8))
    return {"coherence": coherence, "neighbor_corr": neighbor_corr, "leak": leak}
