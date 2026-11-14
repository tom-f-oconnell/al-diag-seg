"""Name predicted ROIs by matching odor tuning, position, shape and layout to a template.

Cost of naming ROI i as glomerulus j:

    w_tuning   * (1 - corr(tuning_i, tuning_j))
  + w_position * centroid distance to j's expected position (in template SDs)
  + w_shape    * Mahalanobis distance of (log volume, log axis lengths)
  + w_atlas    * (1 - mean atlas probability of j over ROI i's voxels)
  + w_layout   * how well i's offsets to its neighbors match the offsets
                 between their assigned glomeruli in the template

The layout term depends on the other assignments, so matching is iterated:
solve with Hungarian, recompute the layout term from the current
assignment, solve again (a simple approximation to graph matching).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment

from .refine import neighbors, roi_tuning

# the terms of the cost of naming an ROI as a glomerulus (see above), in the order they are added
TERMS = ("tuning", "position", "shape", "atlas", "layout")


def _z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-8)


def shape_descriptor(mask: np.ndarray, spacing_zyx) -> np.ndarray:
    """[log volume (um^3), log of 3 principal axis lengths (um), largest first]."""
    pts = np.argwhere(mask) * np.asarray(spacing_zyx, float)
    vol = len(pts) * float(np.prod(spacing_zyx))
    if len(pts) < 4:
        return np.array([np.log(vol + 1e-6), 0.0, 0.0, 0.0])
    ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
    return np.concatenate([[np.log(vol)], 0.5 * np.log(np.maximum(ev, 1e-3))])


def roi_features(responses, labels, spacing_zyx=(1.0, 1.0, 1.0), origin=None):
    """{label: {"tuning", "centroid", "shape"}} for one fly.

    Centroids (um) are relative to ``origin``, default the centroid of all
    ROIs. When an atlas is warped into the fly, expected positions come from
    the atlas instead (see :func:`atlas_centroids`), which is better.
    """
    tun = roi_tuning(responses, labels)
    labs = sorted(tun)
    sp = np.asarray(spacing_zyx, float)
    cents = np.array(center_of_mass(labels > 0, labels, labs)) * sp
    if origin is None:
        origin = cents.mean(0)
    return {
        l: {"tuning": _z(tun[l]), "centroid": c - origin, "shape": shape_descriptor(labels == l, sp)}
        for l, c in zip(labs, cents)
    }


def build_template(flies):
    """flies: [{"responses", "labels", "lut": {label: name}, "spacing"}, ...]

    Returns {name: {"tuning", "centroid", "centroid_sd", "shape", "shape_sd", "n"}}
    """
    acc: dict[str, list] = {}
    for f in flies:
        feats = roi_features(f["responses"], f["labels"], f.get("spacing", (1, 1, 1)))
        for lab, ft in feats.items():
            name = f["lut"].get(lab)
            if name:
                acc.setdefault(name, []).append(ft)
    all_shape = np.stack([ft["shape"] for rows in acc.values() for ft in rows])
    shape_floor = 0.25 * all_shape.std(0) + 1e-3  # keeps SDs sane with few flies
    tmpl = {}
    for name, rows in acc.items():
        C = np.stack([r["centroid"] for r in rows])
        S = np.stack([r["shape"] for r in rows])
        tmpl[name] = {
            "tuning": np.stack([r["tuning"] for r in rows]).mean(0),
            "centroid": C.mean(0),
            "centroid_sd": float(C.std(0).mean()) if len(C) > 1 else 10.0,
            "shape": S.mean(0),
            "shape_sd": np.maximum(S.std(0), shape_floor) if len(S) > 1 else shape_floor * 4,
            "n": len(rows),
        }
    return tmpl


def atlas_centroids(prior, prior_names, spacing_zyx):
    """Expected centroid (um, fly voxel grid) of each glomerulus from the warped atlas."""
    sp = np.asarray(spacing_zyx, float)
    out = {}
    for p, n in zip(prior, prior_names):
        if p.sum() > 0:
            out[n] = np.array(center_of_mass(p)) * sp
    return out


def assign_identities(
    responses,
    labels,
    template,
    spacing_zyx=(1.0, 1.0, 1.0),
    prior=None,
    prior_names=None,
    w_tuning: float = 1.0,
    w_position: float = 0.5,
    w_shape: float = 0.3,
    w_atlas: float = 1.0,
    w_layout: float = 0.5,
    n_layout_iter: int = 3,
    max_cost: float = 0.6,
    unknown_cost: float = 0.5,
    return_terms: bool = False,
    n_best: int = 5,
):
    """Match predicted ROIs to template glomeruli.

    prior, prior_names : optional atlas warped into this fly
        (atlas.atlas_to_fly). Enables the atlas-overlap term and puts the
        position and layout terms in registered coordinates. Glomeruli in the
        atlas but not the template (e.g. added from a fixed reference) are also
        candidates, with their tuning and shape costs set to ``unknown_cost``
        (on the 0-1 scale of the shape term; tuning cost 0.5 is corr 0.5).
    The total cost is divided by the sum of the active weights, so it is on
    the same scale whichever terms are on; pairs with cost > ``max_cost``
    are left unassigned instead of forced.

    Returns {label: (name or None, cost, confidence)}; confidence is the gap
    between the best and second-best glomerulus for that ROI. With
    ``return_terms``, also what the costs are made of (a tuple of the two):
    ``weights`` (of each of `TERMS`; None for one that is off: the atlas term
    without an atlas), ``weight_sum``, ``max_cost``, ``unknown_cost`` and
    ``rois``: {label: {"assigned", "cost", "confidence", "candidates"}}, the
    candidates the ``n_best`` glomeruli of least cost for that ROI (and the
    one it was named, if not among them), least first, each with its ``name``,
    ``cost`` (as compared with ``max_cost``), the value of each term before it
    is weighted (None where it has none), ``reference`` (not in the template:
    its tuning and shape terms are ``unknown_cost``) and ``taken_by`` (the
    label of another ROI given that name, else None).
    """
    sp = np.asarray(spacing_zyx, float)
    feats = roi_features(responses, labels, sp)
    labs = list(feats)
    names = list(template)

    # positions: registered (atlas) if available, else relative to AL center
    if prior is not None:
        expected = atlas_centroids(prior, prior_names, sp)
        roi_pos = {l: np.array(center_of_mass(labels == l)) * sp for l in labs}
        names += [n for n in prior_names if n not in template and n in expected]
    else:
        expected = {n: template[n]["centroid"] for n in names}
        roi_pos = {l: feats[l]["centroid"] for l in labs}
    nL, nN = len(labs), len(names)
    default_sd = np.median([g["centroid_sd"] for g in template.values()])
    exp_pos = np.array([expected.get(n, template.get(n, {}).get("centroid")) for n in names])
    pos_sd = np.array([template[n]["centroid_sd"] if n in template else default_sd for n in names])

    base = np.zeros((nL, nN))
    parts = {k: np.full((nL, nN), np.nan) for k in TERMS}
    for i, l in enumerate(labs):
        f = feats[l]
        for j, n in enumerate(names):
            g = template.get(n)
            d = np.linalg.norm(roi_pos[l] - exp_pos[j]) / (pos_sd[j] + 1e-8)
            if g is None:
                tuning_cost = shape_cost = unknown_cost
            else:
                tuning_cost = 1 - np.corrcoef(f["tuning"], g["tuning"])[0, 1]
                s = np.sqrt(np.mean(((f["shape"] - g["shape"]) / g["shape_sd"]) ** 2))
                shape_cost = min(s, 4.0) / 4.0
            base[i, j] = (w_tuning * tuning_cost
                          + w_position * min(d, 4.0) / 4.0
                          + w_shape * shape_cost)
            parts["tuning"][i, j], parts["position"][i, j], parts["shape"][i, j] = tuning_cost, min(d, 4.0) / 4.0, shape_cost
    if prior is not None and w_atlas:
        pidx = {n: k for k, n in enumerate(prior_names)}
        for i, l in enumerate(labs):
            m = labels == l
            for j, n in enumerate(names):
                if n in pidx:
                    parts["atlas"][i, j] = 1 - float(prior[pidx[n]][m].mean())
                    base[i, j] += w_atlas * parts["atlas"][i, j]

    active = w_tuning + w_position + w_shape + w_layout + (w_atlas if prior is not None else 0)
    weights = {"tuning": w_tuning, "position": w_position, "shape": w_shape,
               "atlas": w_atlas if prior is not None else None, "layout": w_layout}
    base /= active
    w_layout = w_layout / active

    nb = neighbors(labels, spacing=sp)
    assign = _solve(base, max_cost)
    for _ in range(n_layout_iter if w_layout else 0):
        layout = _layout_cost(labs, nb, roi_pos, exp_pos, pos_sd, assign)
        new = _solve(base + w_layout * layout, max_cost)
        if np.array_equal(new, assign):
            break
        assign = new
    if w_layout:
        parts["layout"] = _layout_cost(labs, nb, roi_pos, exp_pos, pos_sd, assign)
    cost = base + (w_layout * parts["layout"] if w_layout else 0)

    out = {}
    for i, l in enumerate(labs):
        srt = np.sort(cost[i])
        conf = float(srt[1] - srt[0]) if nN > 1 else np.inf
        j = assign[i]
        out[l] = (names[j] if j >= 0 else None, float(cost[i, j]) if j >= 0 else max_cost, conf)
    if not return_terms:
        return out
    taken = {int(j): labs[i] for i, j in enumerate(assign) if j >= 0}
    rois = {}
    for i, l in enumerate(labs):
        best = [int(j) for j in np.argsort(cost[i], kind="stable")[:n_best]]
        if assign[i] >= 0 and int(assign[i]) not in best:
            best.append(int(assign[i]))
        rois[int(l)] = {"assigned": out[l][0], "cost": out[l][1], "confidence": out[l][2], "candidates": [
            {"name": names[j], "cost": float(cost[i, j]),
             **{k: None if np.isnan(parts[k][i, j]) else float(parts[k][i, j]) for k in TERMS},
             "reference": names[j] not in template,
             "taken_by": int(taken[j]) if j in taken and taken[j] != l else None} for j in best]}
    return out, {"weights": weights, "weight_sum": float(active), "max_cost": max_cost, "unknown_cost": unknown_cost,
                 "rois": rois}


def _solve(cost, max_cost):
    """Hungarian with a dummy "unassigned" option per ROI. Returns assignment per row (-1 = none)."""
    nL, nN = cost.shape
    padded = np.hstack([cost, np.full((nL, nL), max_cost)])
    rows, cols = linear_sum_assignment(padded)
    a = np.full(nL, -1)
    for i, j in zip(rows, cols):
        if j < nN:
            a[i] = j
    return a


def _layout_cost(labs, nb, roi_pos, exp_pos, pos_sd, assign):
    """For ROI i as glomerulus j: mean mismatch between i's offsets to its
    assigned neighbors and the template offsets between j and theirs."""
    li = {l: i for i, l in enumerate(labs)}
    nN = len(exp_pos)
    L = np.zeros((len(labs), nN))
    for i, l in enumerate(labs):
        terms = []
        for m in nb.get(l, ()):
            k = li.get(m)
            if k is None or assign[k] < 0:
                continue
            off = roi_pos[l] - roi_pos[m]  # observed offset to neighbor
            exp_off = exp_pos - exp_pos[assign[k]]  # (nN, 3) template offsets
            scale = np.sqrt(pos_sd**2 + pos_sd[assign[k]] ** 2) + 1e-8
            terms.append(np.minimum(np.linalg.norm(exp_off - off, axis=1) / scale, 4.0) / 4.0)
        if terms:
            L[i] = np.mean(terms, 0)
    return L
