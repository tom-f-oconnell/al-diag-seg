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

The cleaning steps do not read the movies: with trials, `alseg.clean_trim.trim` already cleans edges
as well as the manual ROIs are, even on evidence of the non-diagnostic recordings alone. `frames`
reads them, for `alseg.nmf` (which looks for glomeruli the ROIs miss): every time point of every
trial, 840-2,460 per pixel on those flies, 0.6-1.4 GB per fly as it keeps them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

TRIAL_DFF = "trial_dff.tif"
TRIALS = "trial_frames_and_odors.json"
MOVIE = "mocorr.tif"  # a recording's motion-corrected movie, (T, Z, Y, X), as al_analysis leaves it
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


def _movie(path: Path, shape_zyx):
    """A recording's movie as (T, Z, Y, X), read as it is asked for where the file allows (else read whole);
    None if it is on another grid. Its planes are a dimension of their own (as ImageJ hyperstacks have
    them, and al_analysis writes them); pages that do not say what they are, are taken as time points'
    planes in turn."""
    with tifffile.TiffFile(path) as t:
        series = t.series[0]
        shape, axes, mapped = tuple(series.shape), series.axes, series.dataoffset is not None
    zyx = tuple(shape_zyx)
    if len(shape) == 4:
        ok = shape[1:] == zyx
    else:
        ok = len(shape) == 3 and shape[1:] == zyx[1:] and (zyx[0] == 1 or (axes[0] not in "TZC" and shape[0] % zyx[0] == 0))
    if not ok:
        return None
    m = tifffile.memmap(path, mode="r") if mapped else tifffile.imread(path)
    return m.reshape((-1,) + zyx)


def frames(mocorr_dir, shape_zyx, names=None, exclude=(), progress=None) -> dict:
    """A fly's motion-corrected movies as evidence, time point by time point: of every recording on its
    grid with a movie (`MOVIE`) and trials (`TRIALS`), every time point of every trial less the mean of
    the trial's time points before its odor (dF, in the movie's units, not over F: light adds).

    Args:
        mocorr_dir: its motion-corrected data folder (as `alseg.fly.load_fly` takes).
        shape_zyx: its grid, (Z, Y, X).
        names: only the recordings of these folder names (default every one on the grid).
        exclude: recordings (folder names) left out.
        progress: called as ``progress(done, total)`` (recordings read).

    Returns:
        dict with ``dff``: (n, Z, Y, X) float16, n time points in all (e.g. 2,460 of 5 recordings);
        ``trial``: (n,) the trial each is of (an index into ``trials``); ``on``: (n,) whether the
        odor is on (from the trial's first time point with it); ``trials``: of each trial, the
        (recording, odor) it is of, its odor as recorded; and ``recordings``: the folders read.

    Raises:
        ValueError: if no recording has a movie and trials on the grid.
    """
    recs = [r for r in sorted(p for p in Path(mocorr_dir).iterdir() if p.is_dir())
            if (r / MOVIE).is_file() and (r / TRIALS).is_file() and (names is None or r.name in names) and r.name not in exclude]
    dff, trial, on, trials, read = [], [], [], [], []
    for i, r in enumerate(recs):
        movie = _movie(r / MOVIE, shape_zyx)
        if movie is not None:
            for t in json.loads((r / TRIALS).read_text(encoding="utf-8")):
                start, onset, end = int(t["start_frame"]), int(t["first_odor_frame"]), min(int(t["end_frame"]), len(movie) - 1)
                if onset <= start or end < onset:
                    continue
                v = np.asarray(movie[start:end + 1], np.float32)
                v -= v[:onset - start].mean(0)
                dff.append(v.astype(np.float16))
                trial += [len(trials)] * len(v)
                on += [False] * (onset - start) + [True] * (end + 1 - onset)
                trials.append((r.name, str(t.get("odors"))))
            read.append(r.name)
        if progress:
            progress(i + 1, len(recs))
    if not dff:
        raise ValueError(f"{mocorr_dir}: no recording with {MOVIE} and {TRIALS} of planes ({', '.join(map(str, shape_zyx))})")
    return {"dff": np.concatenate(dff), "trial": np.array(trial), "on": np.array(on), "trials": trials, "recordings": read}
