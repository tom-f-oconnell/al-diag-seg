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


def match_report(gt: np.ndarray, pred: np.ndarray, thresh: float = 0.5) -> dict:
    """Matched IoU/Dice per manual ROI, plus counts of misses, false
    positives, splits (one manual ROI covered by >1 prediction) and merges."""
    iou, g_ids, p_ids = iou_matrix(gt, pred)
    if iou.size == 0:
        return {"n_gt": len(g_ids), "n_pred": len(p_ids), "tp": 0}
    r, c = linear_sum_assignment(-iou)
    good = iou[r, c] >= thresh
    per_gt = {int(g_ids[i]): float(iou[i, j]) for i, j in zip(r, c)}
    dice = {k: 2 * v / (1 + v) for k, v in per_gt.items()}
    tp = int(good.sum())
    splits = int(((iou > 0.1).sum(1) > 1).sum())
    merges = int(((iou > 0.1).sum(0) > 1).sum())
    return {
        "n_gt": len(g_ids),
        "n_pred": len(p_ids),
        "tp": tp,
        "precision": tp / max(len(p_ids), 1),
        "recall": tp / max(len(g_ids), 1),
        "mean_dice_matched": float(np.mean([dice[int(g_ids[i])] for i in r[good]])) if tp else 0.0,
        "dice_per_gt": dice,
        "splits": splits,
        "merges": merges,
    }
