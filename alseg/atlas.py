"""Probabilistic glomerulus atlas: where each glomerulus is expected to be.

Training flies are registered to a template (ANTsPy), their labeled glomeruli are
warped into that space and averaged into one probability map per glomerulus. For a
new fly, the template is registered to the fly's anatomy and the probability maps are
warped onto the fly's own voxel grid, where they act as soft priors for refinement
and identity assignment.

Registration is in physical space (um), with each image's own voxel size, so flies
can differ in resolution, and functional recordings can be a few planes many um
apart. Each fly is registered between flies through its "registration image": a
higher resolution anatomical stack of the same fly (``fly["hires"]``) if there is
one, since that registers much better than a few planes, otherwise the functional
anatomy itself. The functional planes are placed in the fly's high-res stack by a
rigid (or affine) registration within the fly. So labels go functional planes ->
high-res stack -> template, and priors go back the other way.

The template is the registration image of one training fly (by default the one
sampled most finely in z), resampled so no axis is finer than ``atlas_um``.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .responses import normalize01


def _ants():
    import ants

    return ants


def _img(a, spacing):
    return _ants().from_numpy(np.ascontiguousarray(a, dtype=np.float32), spacing=tuple(float(s) for s in spacing))


def register(moving, fixed, moving_spacing, fixed_spacing, transform: str = "SyN", seed: int = 0) -> list[str]:
    """ANTs transform files that warp ``moving`` onto ``fixed``, in physical space.

    transform: "Affine" for flies that differ mainly by pose, "SyN" (affine +
    deformable) when glomerulus layout also varies, "Rigid" within a fly.
    """
    reg = _ants().registration(fixed=_img(fixed, fixed_spacing), moving=_img(moving, moving_spacing),
                               type_of_transform=transform, random_seed=seed)
    return reg["fwdtransforms"]


def _invert(transforms: list[str]) -> list[tuple[str, bool]]:
    """Chain for the inverse of linear transforms (e.g. from a Rigid registration)."""
    if not all(t.endswith(".mat") for t in transforms):
        raise ValueError("only linear (Rigid, Similarity, Affine) transforms can be inverted here")
    return [(t, True) for t in reversed(transforms)]


def _identity_file() -> str:
    ants = _ants()
    path = tempfile.NamedTemporaryFile(suffix=".mat", delete=False).name
    ants.write_transform(ants.create_ants_transform(transform_type="AffineTransform", dimension=3), path)
    return path


class Warp:
    """Resamples volumes onto a fixed grid through a chain of (transform file,
    invert) pairs, in ANTs order: the last one is applied to the image first."""

    def __init__(self, shape, spacing, chain: list[tuple[str, bool]]):
        self.fixed = _img(np.zeros(shape, np.float32), spacing)
        self.chain = chain or [(_identity_file(), False)]

    def __call__(self, vol, spacing, interpolator: str = "linear") -> np.ndarray:
        out = _ants().apply_transforms(
            fixed=self.fixed, moving=_img(vol, spacing), transformlist=[t for t, _ in self.chain],
            whichtoinvert=[i for _, i in self.chain], interpolator=interpolator)
        return out.numpy()


def registration_image(fly) -> tuple[np.ndarray, tuple]:
    """(anatomy, spacing) a fly is registered between flies with: its high-res stack
    if it has one, otherwise its functional anatomy."""
    h = fly.get("hires")
    return (h["anatomy"], h["spacing"]) if h else (fly["anatomy"], fly["spacing"])


def hires_to_functional(fly, transform: str = "Rigid") -> list[str]:
    """Transforms that warp a fly's high-res stack onto its functional grid ([] if it
    has none). The functional planes are the fixed image, so the fit only uses
    where they actually are."""
    h = fly.get("hires")
    if not h:
        return []
    return register(h["anatomy"], fly["anatomy"], h["spacing"], fly["spacing"], transform)


def template_grid(anatomy, spacing, atlas_um: float) -> tuple[np.ndarray, tuple]:
    """``anatomy`` resampled so no axis is finer than ``atlas_um``."""
    sp = np.maximum(np.asarray(spacing, float), atlas_um)
    if np.allclose(sp, spacing):
        return anatomy.astype(np.float32), tuple(float(s) for s in spacing)
    img = _ants().resample_image(_img(anatomy, spacing), tuple(sp), use_voxels=False, interp_type=0)
    return img.numpy(), tuple(float(s) for s in img.spacing)


@dataclass
class Atlas:
    template: np.ndarray  # (Z, Y, X) mean registered anatomy, each fly scaled to [0, 1]
    prob: np.ndarray  # (K, Z, Y, X) float16, P(glomerulus k at voxel)
    names: list[str]
    spacing: tuple[float, float, float]
    counts: np.ndarray | None = None  # (K,) number of flies behind each map


def build_atlas(flies, transform: str = "SyN", hires_transform: str = "Rigid", atlas_um: float = 1.0,
                smooth_um: float = 1.0, reference: int | None = None) -> Atlas:
    """flies: [{"anatomy", "labels", "lut": {label: name}, "spacing", optional "hires":
    {"anatomy", "spacing"}}, ...] (training flies only).

    ``reference`` (index) is the fly whose registration image becomes the template;
    by default the one sampled most finely in z. ``smooth_um`` blurs each warped mask
    a little, so the prior is soft at borders even with few flies.
    For a less reference-biased template, run this twice, using the first
    template as the reference the second time.
    """
    if reference is None:
        reference = min(range(len(flies)), key=lambda i: registration_image(flies[i])[1][0])
    template, t_sp = template_grid(*registration_image(flies[reference]), atlas_um)
    names = sorted({n for f in flies for n in f["lut"].values()})
    idx = {n: i for i, n in enumerate(names)}
    acc = np.zeros((len(names),) + template.shape, np.float32)
    seen = np.zeros(len(names), np.float32)
    anat_sum = np.zeros(template.shape, np.float32)
    sigma = smooth_um / np.asarray(t_sp)
    for i, f in enumerate(flies):
        anat, sp = registration_image(f)
        # the reference's registration image is the template, so it needs no transform
        to_t = [] if i == reference else [(t, False) for t in register(anat, template, sp, t_sp, transform)]
        anat_sum += Warp(template.shape, t_sp, to_t)(normalize01(anat), sp)
        chain = to_t + _invert(hires_to_functional(f, hires_transform))
        wl = Warp(template.shape, t_sp, chain)(f["labels"].astype(np.float32), f["spacing"], "genericLabel")
        wl = np.rint(wl).astype(np.int32)
        for lab, name in f["lut"].items():
            m = (wl == lab).astype(np.float32)
            if smooth_um:
                m = ndi.gaussian_filter(m, sigma)
            acc[idx[name]] += m
            seen[idx[name]] += 1
    prob = acc / np.maximum(seen, 1)[:, None, None, None]
    return Atlas(anat_sum / len(flies), prob.astype(np.float16), names, t_sp, seen)


def atlas_to_fly(atlas: Atlas, fly, transform: str = "SyN", hires_transform: str = "Rigid") -> np.ndarray:
    """Warp the atlas onto a fly's functional grid (through its high-res stack, if it
    has one). Returns (K, Z, Y, X) float32."""
    anat, sp = registration_image(fly)
    chain = ([(t, False) for t in hires_to_functional(fly, hires_transform)]
             + [(t, False) for t in register(atlas.template, anat, atlas.spacing, sp, transform)])
    warp = Warp(fly["anatomy"].shape, fly["spacing"], chain)
    return np.stack([np.clip(warp(p.astype(np.float32), atlas.spacing), 0, 1) for p in atlas.prob])
