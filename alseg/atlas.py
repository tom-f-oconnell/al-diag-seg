"""Probabilistic glomerulus atlas: where each glomerulus is expected to be.

Training flies are registered to a reference fly (ANTsPy), their labeled
glomeruli are warped into that space and averaged into one probability map
per glomerulus. For a new fly, the template is registered to the fly's
anatomy and the probability maps are warped into the fly's own voxel grid,
where they act as soft priors for refinement and identity assignment.

pip install antspyx
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


def _ants():
    import ants

    return ants


def _img(a, spacing):
    return _ants().from_numpy(np.ascontiguousarray(a, dtype=np.float32), spacing=tuple(float(s) for s in spacing))


def register(moving, fixed, spacing, transform: str = "SyN", seed: int = 0):
    """Register ``moving`` anatomy onto ``fixed``. Returns a function that
    warps any volume on the moving grid onto the fixed grid.

    transform: "Affine" for flies that differ mainly by pose, "SyN"
    (affine + deformable) when glomerulus layout also varies.
    """
    ants = _ants()
    f, m = _img(fixed, spacing), _img(moving, spacing)
    reg = ants.registration(fixed=f, moving=m, type_of_transform=transform, random_seed=seed)
    tx = reg["fwdtransforms"]

    def warp(vol, interpolator="linear"):
        out = ants.apply_transforms(fixed=f, moving=_img(vol, spacing), transformlist=tx,
                                    interpolator=interpolator)
        return out.numpy()

    return warp


@dataclass
class Atlas:
    template: np.ndarray  # (Z, Y, X) mean registered anatomy
    prob: np.ndarray  # (K, Z, Y, X) float16, P(glomerulus k at voxel)
    names: list[str]
    spacing: tuple[float, float, float]


def build_atlas(flies, spacing, reference: int = 0, transform: str = "SyN", smooth: float = 1.0) -> Atlas:
    """flies: [{"anatomy", "labels", "lut": {label: name}}, ...] (training flies only).

    ``smooth`` (voxels, in-plane) blurs each warped mask a little so the
    prior is soft at borders even with few flies.
    All flies must share one grid (same Z, Y, X); resample beforehand if not.
    For a less reference-biased template, run this twice, using the first
    template as the reference the second time.
    """
    ref = flies[reference]["anatomy"]
    names = sorted({n for f in flies for n in f["lut"].values()})
    idx = {n: i for i, n in enumerate(names)}
    acc = np.zeros((len(names),) + ref.shape, np.float32)
    seen = np.zeros(len(names), np.float32)
    anat_sum = np.zeros(ref.shape, np.float32)
    for i, f in enumerate(flies):
        if i == reference:
            warp = lambda v, interpolator="linear": np.asarray(v, np.float32)
        else:
            warp = register(f["anatomy"], ref, spacing, transform)
        anat_sum += warp(f["anatomy"])
        wl = np.rint(warp(f["labels"].astype(np.float32), "genericLabel")).astype(np.int32)
        for lab, name in f["lut"].items():
            m = (wl == lab).astype(np.float32)
            if smooth:
                m = ndi.gaussian_filter(m, (smooth / 2, smooth, smooth))
            acc[idx[name]] += m
            seen[idx[name]] += 1
    prob = acc / np.maximum(seen, 1)[:, None, None, None]
    return Atlas(anat_sum / len(flies), prob.astype(np.float16), names, tuple(spacing))


def atlas_to_fly(atlas: Atlas, anatomy, transform: str = "SyN") -> np.ndarray:
    """Warp the atlas into a fly's voxel grid. Returns (K, Z, Y, X) float32."""
    warp = register(atlas.template, anatomy, atlas.spacing, transform)
    return np.stack([np.clip(warp(p.astype(np.float32)), 0, 1) for p in atlas.prob])
