"""Turn odor-response imaging into per-voxel features and CNN input channels.

Two input formats are supported:

* a raw movie ``(T, Z, Y, X)`` plus odor onset frames, via
  :func:`response_vectors_from_movie`, or
* precomputed response volumes ``(n_odors, Z, Y, X)`` (e.g. mean dF/F per
  odor), which you can pass straight to :func:`build_channels`.

Motion-correct the movie before any of this; otherwise the local-correlation
and PCA maps blur across glomerulus borders.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def response_vectors_from_movie(
    movie: np.ndarray,
    onsets: list[list[int]],
    baseline: tuple[int, int] = (-10, 0),
    window: tuple[int, int] = (2, 12),
) -> np.ndarray:
    """Trial-averaged mean dF/F per odor.

    movie : (T, Z, Y, X) volumes over time
    onsets : onsets[k] = list of onset volume indices for odor k (one per trial)
    baseline, window : frame offsets relative to onset

    Returns (n_odors, Z, Y, X) float32.
    """
    out = []
    for trials in onsets:
        per_trial = []
        for t0 in trials:
            f0 = movie[t0 + baseline[0] : t0 + baseline[1]].mean(0)
            f = movie[t0 + window[0] : t0 + window[1]].mean(0)
            per_trial.append((f - f0) / (f0 + 1e-3 * f0.mean() + 1e-6))
        out.append(np.mean(per_trial, 0))
    return np.stack(out).astype(np.float32)


def local_correlation(movie: np.ndarray, size: tuple[int, int] = (3, 3)) -> np.ndarray:
    """In-plane local correlation image (Suite2p/CaImAn style) per z-slice.

    movie : (T, Z, Y, X). Can also be the (n_odors, Z, Y, X) response stack,
    in which case it measures similarity of odor tuning to neighbors, which is
    exactly the cue that separates adjacent glomeruli.
    Returns (Z, Y, X) in [-1, 1].
    """
    x = movie.astype(np.float32)
    x = x - x.mean(0)
    x /= np.sqrt((x**2).sum(0)) + 1e-8
    k = (1, 1) + tuple(size)
    n = np.prod(size)
    # sum over neighbors of <x_i, x_j>, excluding self, divided by n - 1
    s = uniform_filter(x, size=k, mode="nearest") * n
    corr = ((x * s).sum(0) - 1.0) / (n - 1)
    return corr


class SharedPCA:
    """PCA on per-voxel response vectors, fit once on training flies.

    Fitting the basis on pooled training data (not per fly) keeps the
    channels comparable across flies.
    """

    def __init__(self, n_components: int = 3):
        self.k = n_components

    def fit(self, response_stacks, masks=None, max_voxels: int = 200_000, seed: int = 0):
        rng = np.random.default_rng(seed)
        rows = []
        for i, r in enumerate(response_stacks):
            v = _zscore_vectors(r)
            if masks is not None:
                v = v[masks[i].ravel()]
            rows.append(v)
        X = np.concatenate(rows)
        if len(X) > max_voxels:
            X = X[rng.choice(len(X), max_voxels, replace=False)]
        self.mean_ = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = vt[: self.k]
        return self

    def transform(self, r: np.ndarray) -> np.ndarray:
        """(n_odors, Z, Y, X) -> (k, Z, Y, X)"""
        v = _zscore_vectors(r)
        p = (v - self.mean_) @ self.components_.T
        return p.T.reshape((self.k,) + r.shape[1:])


def _zscore_vectors(r):
    """(n_odors, Z, Y, X) -> (n_voxels, n_odors), each odor z-scored within fly."""
    v = r.reshape(r.shape[0], -1).T
    return (v - v.mean(0)) / (v.std(0) + 1e-8)


def normalize01(img: np.ndarray, lo: float = 1, hi: float = 99.5) -> np.ndarray:
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / (b - a + 1e-8), 0, 1).astype(np.float32)


def build_channels(anatomy: np.ndarray, responses: np.ndarray, pca: SharedPCA) -> np.ndarray:
    """Stack CNN input channels: (C, Z, Y, X), each scaled to [0, 1].

    C = 3 to match Cellpose-SAM's 3-channel input:
      0: anatomy (mean / baseline fluorescence)
      1: local correlation of odor tuning
      2: first shared PCA component of the tuning vectors
    Swap channel 2 for another PC, or use PCs 1-3 as an RGB "tuning color"
    image, if that separates your glomeruli better. With a custom U-Net you
    can simply feed all of them.
    """
    lc = local_correlation(responses)
    pcs = pca.transform(responses)
    chans = [normalize01(anatomy), normalize01(lc), normalize01(pcs[0])]
    return np.stack(chans)
