"""Per-pixel evidence of which glomerulus a pixel belongs to, over all of a fly's recordings, for the
steps that clean ROIs (`alseg.clean_trim`).

The model reads only the diagnostic recordings' trial means (20 values per pixel), which are too noisy
to tell a pixel's glomerulus from its neighbour's. Here every recording of the fly counts (megamat,
validation and other panels too), trial by trial, as al_analysis leaves them in each recording's
folder (``<analysis>/<date>/<fly>/<recording>/``):

* ``kind="trials"``: ``trial_dff.tif``, (trial, Z, Y, X) float32 (its axes labelled TZYX), each
  trial's dF/F map, in the order of ``trial_frames_and_odors.json`` (each odor's
  ``trialmean_dff.tif`` is the mean of its trials). On fit 7's held-out flies 117-153 values per
  pixel from 5-6 recordings (the trial means the model reads: 20), 145-250 MB per fly, read in
  5-10 s (0.3-2 s when cached).

There is no kind of the movies (``mocorr.tif``) yet: with trials, `alseg.clean_trim.trim` already
cleans edges as well as the manual ROIs are, even on evidence of the non-diagnostic recordings alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

TRIAL_DFF = "trial_dff.tif"
TRIALS = "trial_frames_and_odors.json"
KINDS = ("trials",)


def recordings(mocorr_dir, shape_zyx, kind: str = "trials") -> list[Path]:
    """A fly's recording folders with evidence of ``kind`` on its grid ((Z, Y, X) ``shape_zyx``), in
    name order. Recordings on another grid (another field of view or number of planes) are left out."""
    if kind not in KINDS:
        raise ValueError(f"evidence kind {kind!r}: not one of {', '.join(KINDS)}")
    out = []
    for r in sorted(p for p in Path(mocorr_dir).iterdir() if p.is_dir()):
        f = r / TRIAL_DFF
        if not (f.is_file() and (r / TRIALS).is_file()):
            continue
        with tifffile.TiffFile(f) as t:
            shape = t.series[0].shape
        if tuple(shape[-3:]) == tuple(shape_zyx):
            out.append(r)
    return out


def load(mocorr_dir, shape_zyx, kind: str = "trials", names=None, exclude=(), clip: float = 5.0) -> np.ndarray:
    """A fly's per-pixel evidence over its recordings.

    Args:
        mocorr_dir: its motion-corrected data folder (as `alseg.fly.load_fly` takes).
        shape_zyx: its grid, (Z, Y, X) (e.g. ``fly["average"].shape``).
        kind: ``"trials"`` (see the module's docstring).
        names: only the recordings of these folder names (default every one on the grid).
        exclude: recordings (folder names) left out.
        clip: dF/F values beyond +-``clip`` are clipped (al_analysis' dF/F is huge at the odd pixel
            whose baseline is near 0), and values that are not finite set to 0.

    Returns:
        (n, Z, Y, X) float32: n values per pixel, all recordings' joined (e.g. 123 trials of 5
        recordings).

    Raises:
        ValueError: if no recording has evidence of ``kind`` on the grid.
    """
    recs = [r for r in recordings(mocorr_dir, shape_zyx, kind)
            if (names is None or r.name in names) and r.name not in exclude]
    if not recs:
        raise ValueError(f"{mocorr_dir}: no recording with {TRIAL_DFF} of shape (trials, {', '.join(map(str, shape_zyx))})")
    out = []
    for r in recs:
        x = tifffile.imread(r / TRIAL_DFF).astype(np.float32)
        x = x.reshape((-1,) + tuple(shape_zyx))
        x[~np.isfinite(x)] = 0
        out.append(np.clip(x, -clip, clip, out=x))
    return np.concatenate(out)
