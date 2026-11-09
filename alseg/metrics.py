"""Geometry metrics: predicted vs. manual ROIs."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt: np.ndarray, pred: np.ndarray):
    g_ids = np.unique(gt)[1:]
    p_ids = np.unique(pred)[1:]
    # contingency table via a joint histogram
    idx = gt.astype(np.int64) * (pred.max() + 1) + pred
    counts = np.bincount(idx.ravel(), minlength=(gt.max() + 1) * (pred.max() + 1))
    counts = counts.reshape(gt.max() + 1, pred.max() + 1)
    inter = counts[np.ix_(g_ids, p_ids)].astype(float)
    area_g = counts.sum(1)[g_ids][:, None]
    area_p = counts.sum(0)[p_ids][None, :]
    return inter / (area_g + area_p - inter + 1e-8), g_ids, p_ids


def _compact(a: np.ndarray):
    """Labels numbered 1..n (0 stays 0), and the original label of each."""
    ids = np.unique(a)
    ids = ids[ids > 0]
    lut = np.zeros(int(a.max()) + 1 if a.size else 1, np.int64)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[a], ids


def iou_matrix_partly_drawn(gt: np.ndarray, pred: np.ndarray):
    """`iou_matrix` for manual ROIs drawn on some of a glomerulus' planes only (often its
    clearest): a predicted object's voxels on planes where the manual ROI has no outline count
    against it only where they lie in another manual ROI (elsewhere, the glomerulus may well
    be there, undrawn). Also returns the fraction of each predicted object's voxels that lie in
    any manual ROI."""
    g, g_ids = _compact(gt)
    p, p_ids = _compact(pred)
    ng, npr, nz = len(g_ids) + 1, len(p_ids) + 1, gt.shape[0]
    idx = (np.arange(nz)[:, None, None] * ng + g) * npr + p
    c = np.bincount(idx.ravel(), minlength=nz * ng * npr).reshape(nz, ng, npr)  # (plane, manual, predicted)
    inter = c[:, 1:, 1:].sum(0).astype(float)
    area_g = c[:, 1:, :].sum((0, 2))[:, None]
    per_plane = c[:, :, 1:].sum(1)  # (plane, predicted): each one's voxels
    in_manual = c[:, 1:, 1:].sum(1)  # ... in any manual ROI
    drawn = (c[:, 1:, :].sum(2) > 0).T.astype(float)  # (manual, plane)
    counted = drawn @ per_plane + (1 - drawn) @ in_manual
    frac_in = in_manual.sum(0) / np.maximum(per_plane.sum(0), 1)
    return inter / (area_g + counted - inter + 1e-8), g_ids, p_ids, frac_in


def _scored(n_pred: int, matched, frac_in: np.ndarray | None, min_in: float = 0.5) -> int:
    """Predicted ROIs scored: all, or (``frac_in`` given) those matched and those that lie mostly
    in manual ROIs. One lying mostly where none is drawn may be an undrawn glomerulus."""
    if frac_in is None:
        return n_pred
    m = np.zeros(n_pred, bool)
    m[list(matched)] = True
    return int((m | (frac_in >= min_in)).sum())


def match_report_2d(gt: np.ndarray, pred: np.ndarray, thresh: float = 0.5, partly_drawn: bool = False) -> dict:
    """`match_report`'s recall, precision and mean Dice of matched ROIs, plane by plane: each
    label's outline on a plane is an ROI, as ROIs are drawn and used (and as Cellpose predicts
    them, before its planes' masks are stitched into 3D objects). Not penalized, unlike the 3D
    match, by an object spanning other planes than the manual one, planes being far apart.
    ``partly_drawn``: precision of only the predicted ROIs matched or lying mostly in manual
    ROIs (``n_scored``), as glomeruli are often drawn on some of their planes only."""
    tp = n_gt = n_pred = n_scored = 0
    dice = []
    for z in range(gt.shape[0]):
        iou, g_ids, p_ids = iou_matrix(gt[z], pred[z])
        n_gt, n_pred = n_gt + len(g_ids), n_pred + len(p_ids)
        matched = []
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            ok = iou[r, c] >= thresh
            v, matched = iou[r, c][ok], c[ok]
            tp += len(v)
            dice += list(2 * v / (1 + v))
        frac_in = None
        if partly_drawn and len(p_ids):
            area = np.bincount(pred[z].ravel(), minlength=int(p_ids.max()) + 1)[p_ids]
            frac_in = np.bincount(pred[z][gt[z] > 0], minlength=int(p_ids.max()) + 1)[p_ids] / area
        n_scored += _scored(len(p_ids), matched, frac_in)
    return {"n_gt": n_gt, "n_pred": n_pred, "n_scored": n_scored, "tp": tp, "precision": tp / max(n_scored, 1),
            "recall": tp / max(n_gt, 1), "mean_dice_matched": float(np.mean(dice)) if dice else 0.0}


def match_report(gt: np.ndarray, pred: np.ndarray, thresh: float = 0.5, partly_drawn: bool = False) -> dict:
    """Matched IoU/Dice per manual ROI, plus counts of misses, false
    positives, splits (one manual ROI covered by >1 prediction) and merges.
    ``partly_drawn``: as `iou_matrix_partly_drawn` matches them, and precision of only the
    predicted objects matched or lying mostly in manual ROIs (``n_scored``), as glomeruli are
    often drawn on some of their planes only."""
    if partly_drawn:
        iou, g_ids, p_ids, frac_in = iou_matrix_partly_drawn(gt, pred)
    else:
        (iou, g_ids, p_ids), frac_in = iou_matrix(gt, pred), None
    if iou.size == 0:
        return {"n_gt": len(g_ids), "n_pred": len(p_ids), "n_scored": _scored(len(p_ids), [], frac_in), "tp": 0}
    r, c = linear_sum_assignment(-iou)
    good = iou[r, c] >= thresh
    per_gt = {int(g_ids[i]): float(iou[i, j]) for i, j in zip(r, c)}
    dice = {k: 2 * v / (1 + v) for k, v in per_gt.items()}
    tp = int(good.sum())
    n_scored = _scored(len(p_ids), c[good], frac_in)
    splits = int(((iou > 0.1).sum(1) > 1).sum())
    merges = int(((iou > 0.1).sum(0) > 1).sum())
    return {
        "n_gt": len(g_ids),
        "n_pred": len(p_ids),
        "n_scored": n_scored,
        "tp": tp,
        "precision": tp / max(n_scored, 1),
        "recall": tp / max(len(g_ids), 1),
        "mean_dice_matched": float(np.mean([dice[int(g_ids[i])] for i in r[good]])) if tp else 0.0,
        "dice_per_gt": dice,
        "splits": splits,
        "merges": merges,
    }
