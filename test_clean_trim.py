"""Tests of `alseg.evidence` and `alseg.clean_trim` on small synthetic planes (a few seconds, no GPU).

Usage: python -m pytest test_clean_trim.py, or python test_clean_trim.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from alseg import clean_trim, evidence

N = 80  # trials
SP = (10.0, 0.5, 0.5)


def _tunings(rng, k, corr=0.0):
    """``k`` trial-response vectors (k, N), correlated ``corr`` with the first."""
    t = np.clip(rng.normal(0, 1, (k, N)), 0, None) ** 1.5
    for i in range(1, k):
        t[i] = corr * t[0] + (1 - corr) * t[i]
    return t


def _fly(truth, tunings, rng, blur_px=1.5, noise=0.3, spacing=SP):
    """A one-plane fly whose glomeruli (``truth``, (Y, X) labels 1..k) respond with ``tunings``, blurred
    across their borders, plus noise: {"evidence": (N, 1, Y, X), "spacing"}."""
    E = np.zeros((N,) + truth.shape, np.float32)
    for g, t in enumerate(tunings):
        E += t[:, None, None] * (truth == g + 1)
    E = ndi.gaussian_filter(E, (0, blur_px, blur_px)) + rng.normal(0, noise, E.shape)
    return {"evidence": E[:, None].astype(np.float32), "spacing": spacing, "responses": None}


def _share(E, edge, a, b):
    """Neighbour b's share in the mean of ``edge`` pixels (non-negative least squares of a's and b's)."""
    from scipy.optimize import nnls
    (x, y), _ = nnls(np.stack([E[:, a].mean(1), E[:, b].mean(1)], 1), E[:, edge].mean(1))
    return y / (x + y)


def test_trim_cuts_overshoot():
    """Glomeruli A (left) and B (right), A's mask 3 px into B: trim cuts the part facing B, keeps
    A's labels, core and most of its area, and removes nothing elsewhere."""
    rng = np.random.default_rng(1)
    truth = np.zeros((40, 64), int)
    truth[5:35, 4:32], truth[5:35, 32:60] = 1, 2
    f = _fly(truth, _tunings(rng, 2), rng)
    pred = np.zeros((1,) + truth.shape, np.int32)
    pred[0, 5:35, 4:35], pred[0, 5:35, 35:60] = 7, 9  # A's mask overshoots into B
    out, rep = clean_trim.trim(pred, f, return_report=True)
    assert out.dtype == pred.dtype and set(np.unique(out)) == {0, 7, 9}
    assert ((out == 0) | (out == pred)).all(), "trim may only remove pixels"
    a = out[0] == 7
    assert not a[:, 33:].any(), "A kept pixels of B"
    assert a[8:32, 8:28].all(), "A lost its core"
    assert 0.7 < rep["kept"][7, 0] < 1 and not rep["flagged"]
    E = f["evidence"][:, 0]
    core_a, core_b = ndi.binary_erosion(truth == 1, iterations=5), ndi.binary_erosion(truth == 2, iterations=5)
    edge = lambda m: m & ~ndi.binary_erosion(m, iterations=2) & (np.arange(64) > 20)  # noqa: E731
    before, after = _share(E, edge(pred[0] == 7), core_a, core_b), _share(E, edge(a), core_a, core_b)
    assert after < before / 2 and after < 0.2, (before, after)


def test_trim_band_and_pixel_modes_agree_far_from_neighbours():
    """A mask alone on its plane (no neighbour, coherent) is left as it is by either mode."""
    rng = np.random.default_rng(2)
    truth = np.zeros((40, 40), int)
    truth[8:32, 8:32] = 1
    f = _fly(truth, _tunings(rng, 1), rng)
    pred = truth[None].astype(np.int32) * 3
    for mode in ("band", "pixel"):
        out = clean_trim.trim(pred, f, mode=mode)
        assert (out == 3).sum() >= 0.9 * (pred == 3).sum(), mode


def test_trim_unlabelled_neighbour():
    """A's mask spills into C, a glomerulus with no ROI: by default it stays (no neighbour's core to
    judge it by); with ``min_corr_pct``, the pixels outside A's core that do not follow it go."""
    rng = np.random.default_rng(3)
    truth = np.zeros((40, 64), int)
    truth[5:35, 4:36], truth[5:35, 36:60] = 1, 2
    f = _fly(truth, _tunings(rng, 2), rng, blur_px=0.8)
    pred = np.zeros((1,) + truth.shape, np.int32)
    pred[0, 5:35, 4:42] = 1  # 6 px into C
    assert (clean_trim.trim(pred, f) == pred).all()
    out = clean_trim.trim(pred, f, min_corr_pct=5)
    assert (out[0, :, 39:] == 0).all() and (out[0, :, 8:30] == 1).sum() > 0.9 * 30 * 22


def test_trim_in_um():
    """The same fly at half the pixel size (twice the pixels): distances are in um, so trimming
    reaches as deep in um."""
    rng = np.random.default_rng(4)
    t = _tunings(rng, 2)
    truth = np.zeros((40, 64), int)
    truth[5:35, 4:32], truth[5:35, 32:60] = 1, 2
    pred = np.zeros((1,) + truth.shape, np.int32)
    pred[0, 5:35, 4:35], pred[0, 5:35, 35:60] = 1, 2
    lost = []
    for k, sp in ((1, (10.0, 1.0, 1.0)), (2, (10.0, 0.5, 0.5))):
        tr = np.kron(truth, np.ones((k, k), int))
        p = np.kron(pred[0], np.ones((k, k), int))[None]
        f = _fly(tr, t, np.random.default_rng(5), blur_px=1.5 * k, spacing=sp)
        out = clean_trim.trim(p, f)
        lost.append(((p == 1) & (out == 0)).sum() * sp[1] * sp[2])
    assert abs(lost[0] - lost[1]) <= 0.35 * max(lost), lost


def test_trim_flags_masks_it_would_halve():
    """A mask that would lose more than ``max_loss`` of its area is left whole and flagged."""
    rng = np.random.default_rng(6)
    truth = np.zeros((40, 64), int)
    truth[5:35, 4:32], truth[5:35, 32:60] = 1, 2
    f = _fly(truth, _tunings(rng, 2), rng)
    pred = np.zeros((1,) + truth.shape, np.int32)
    pred[0, 5:35, 4:35], pred[0, 5:35, 35:60] = 1, 2
    out, rep = clean_trim.trim(pred, f, max_loss=0.01, return_report=True)
    assert rep["flagged"] == {1: [0]} and (out[0] == 1).sum() == (pred[0] == 1).sum()


def test_trim_falls_back_to_responses():
    """Without ``evidence``, the fly's trial-mean ``responses`` are the evidence."""
    rng = np.random.default_rng(7)
    truth = np.zeros((40, 64), int)
    truth[5:35, 4:32], truth[5:35, 32:60] = 1, 2
    f = _fly(truth, _tunings(rng, 2), rng)
    pred = np.zeros((1,) + truth.shape, np.int32)
    pred[0, 5:35, 4:35], pred[0, 5:35, 35:60] = 1, 2
    a = clean_trim.trim(pred, f)
    b = clean_trim.trim(pred, {"responses": f["evidence"], "spacing": SP})
    assert (a == b).all()


def _two_in_one(rng, corr=0.0, noise=0.3):
    truth = np.zeros((40, 64), int)
    truth[5:35, 4:32], truth[5:35, 32:60] = 1, 2
    f = _fly(truth, _tunings(rng, 2, corr), rng, noise=noise)
    masks = np.zeros((2,) + truth.shape, np.int32)
    masks[0, 5:35, 4:60] = 4  # one mask over both
    masks[1, 5:35, 4:60] = 4  # and the same on a second plane (as joined labels)
    f["evidence"] = np.concatenate([f["evidence"]] * 2, axis=1)
    return truth, masks, f


def test_split_merged_mask():
    """A mask over two glomeruli tuned unlike is cut between them; each plane's part gets its own new
    label above every other, and the larger part keeps the old one."""
    truth, masks, f = _two_in_one(np.random.default_rng(8))
    masks[1, 0, 0] = 11  # (the highest label, elsewhere)
    out, rep = clean_trim.split(masks, f, return_report=True)
    assert set(rep["split"]) == {(4, 0), (4, 1)} and sorted(rep["split"].values()) == [12, 13]
    for z in (0, 1):
        new = rep["split"][4, z]
        parts = [(out[z] == 4) & (masks[z] == 4), out[z] == new]
        assert parts[0].sum() >= parts[1].sum() and ((out[z] == 0) == (masks[z] == 0)).all()
        for p in parts:
            g = np.bincount(truth[p], minlength=3)[1:]
            assert g.max() >= 0.9 * g.sum(), g  # each part one glomerulus


def test_split_leaves_one_glomerulus_whole():
    """Masks of one glomerulus each, however noisy, are not split; nor one over two tuned alike."""
    rng = np.random.default_rng(9)
    truth = np.zeros((40, 64), int)
    truth[5:35, 4:32], truth[5:35, 32:60] = 1, 2
    for noise in (0.3, 1.0):
        f = _fly(truth, _tunings(rng, 2), rng, noise=noise)
        out, rep = clean_trim.split(truth[None].astype(np.int32), f, return_report=True)
        assert not rep["split"] and (out[0] == truth).all(), (noise, rep["stats"])
    _, masks, f = _two_in_one(np.random.default_rng(10), corr=0.9)
    out, rep = clean_trim.split(masks, f, return_report=True)
    assert not rep["split"], rep["stats"]


def test_split_flags_incoherent():
    """A mask over two glomeruli tuned too alike to split (parts correlate over ``max_corr``) but
    less coherent than ``min_coherence`` is flagged."""
    _, masks, f = _two_in_one(np.random.default_rng(11), corr=0.6, noise=0.05)
    out, rep = clean_trim.split(masks, f, max_corr=0.0, min_coherence=0.99, return_report=True)
    assert not rep["split"] and rep["flagged"] == {4: [0, 1]} and (out == masks).all()


def _recording(d: Path, x, n_odors=None):
    d.mkdir(parents=True)
    tifffile.imwrite(d / evidence.TRIAL_DFF, x, imagej=True, metadata={"axes": "TZYX"})
    trials = [{"odors": f"o{i % (n_odors or len(x))}"} for i in range(len(x))]
    (d / evidence.TRIALS).write_text(json.dumps(trials))


def test_evidence_load():
    """Every recording's trials on the fly's grid, joined in folder name order; others left out;
    NaN as 0, values clipped; names and exclude pick recordings."""
    rng = np.random.default_rng(12)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        a = rng.normal(0, 1, (6, 2, 8, 9)).astype(np.float32)
        a[0, 0, 0, 0], a[1, 0, 0, 0] = np.nan, 50
        b = rng.normal(0, 1, (3, 2, 8, 9)).astype(np.float32)
        _recording(d / "diag1", a, 2)
        _recording(d / "megamat", b)
        _recording(d / "other_fov", rng.normal(0, 1, (4, 2, 8, 10)).astype(np.float32))
        (d / "empty").mkdir()
        assert [r.name for r in evidence.recordings(d, (2, 8, 9))] == ["diag1", "megamat"]
        e = evidence.load(d, (2, 8, 9))
        assert e.shape == (9, 2, 8, 9) and e.dtype == np.float32
        assert e[0, 0, 0, 0] == 0 and e[1, 0, 0, 0] == 5.0 and np.allclose(e[6:], b)
        assert evidence.load(d, (2, 8, 9), names=["megamat"]).shape[0] == 3
        assert evidence.load(d, (2, 8, 9), exclude=["megamat"]).shape[0] == 6
        try:
            evidence.load(d, (3, 8, 9))
        except ValueError:
            pass
        else:
            raise AssertionError("no recording on that grid, but no error")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
