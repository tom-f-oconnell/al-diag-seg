# Design notes

## Voxel sizes: few planes far apart, and differences between flies

Functional recordings are typically 8–12 planes 8–12 µm apart, with pixels 10–20×
finer in-plane, and the resolution differs somewhat between flies. Some flies also have
a higher resolution, more isotropic anatomical stack (`hires/`).

### What the pipeline does

1. **Voxel sizes per fly**, from ThorImage `Experiment.xml` files (via hong2p) or the
   command line, never both, and from XML for all flies or none. The XML has to match
   the TIFF's shape, since its pixel size would be wrong for cropped or binned data.
2. **Each fly stays on its own grid, and is never resampled in z.** Geometry is in µm:
   identity matching (centroids, shapes, layout), ROI growth during refinement (which
   then stays in-plane when planes are further apart than the growth distance), the
   weight of z-neighbors in the smoothness term, and minimum ROI size.
3. **Segmentation is 2D, at one in-plane pixel size** (`--xy-um`, default the median
   across flies). Every plane is resampled in-plane for the model and its masks
   resampled back, so glomeruli are the same size in pixels for every fly. The pixel
   size is saved next to the model for `--skip-train`.
4. **Registration is in physical space** (ANTs, with each image's voxel size), and
   goes through each fly's `hires/` stack if it has one: the functional planes are
   placed in the stack by a rigid registration within the fly (`--hires-reg`), with the
   planes as the fixed image so the fit only uses real data. Stacks are registered
   between flies (`--reg`, SyN by default). Flies without a stack register their
   functional anatomy directly. The template is the most finely sampled (in z)
   registration image, coarsened to `--atlas-um`.
5. **Reference atlases** (`--ref`) are fit to the atlas with an affine or a similarity
   transform, whichever predicts left-out glomeruli better. Centroids from planes far
   apart are biased in z, which a full affine fit to a dozen glomeruli overfits.

On the synthetic data (`make_synthetic.py`, which samples flies like this):

| Check | Result |
|---|---|
| High-res stack placed on the functional planes (label Dice vs. truth) | rigid 0.94–0.95; affine 0.77–0.94; no registration 0.42–0.64 |
| Atlas prior argmax accuracy, held-out fly with a stack | 0.85 using stacks vs. 0.80 ignoring them |
| Same, held-out fly without a stack (better atlas only) | 0.78 vs. 0.77 |
| Reference fit residual (synthetic reference) | similarity 3.7 µm median vs. affine 5.1 µm; on Grabe → hemibrain, affine is chosen (5.1 vs. 6.1 µm) |

### Alternatives

* **Resample everything to an isotropic grid first.** One grid for everything is the
  simplest code. But at ~10 µm between planes, 80–90% of the voxels would be
  interpolated, memory and time grow about 10×, the model would see interpolated planes
  that look nothing like real ones, and ROIs drawn on real planes have no counterpart
  on the made-up ones.
* **Treat every plane as an independent 2D image.** Robust to any z spacing, and
  simplest of all. But a glomerulus spanning 2–3 planes then gets named 2–3 times
  independently, and there is no 3D atlas: plane depths differ between flies, so a
  per-plane 2D atlas has nothing consistent to average.
* **A 3D model** (Cellpose `do_3D` with its anisotropy setting, or a 3D U-Net). It
  could use context across planes, but a glomerulus spans only 1–3 planes, the labels
  are drawn per plane anyway, and Cellpose's 3D mode upsamples z to near-isotropic
  first (about 10–20× here).
* **Tell Cellpose each fly's scale via `diameter`** instead of resampling. Equivalent in
  principle, since Cellpose rescales by diameter internally, but training on mixed
  scales then needs per-image diameters. Explicit resampling keeps training and
  prediction consistent.
* **Place the functional planes in the stack from stage positions** instead of
  registering. Nothing to fail to converge, but it needs accurate positions in the
  metadata and ignores movement between the two recordings. Could serve as an
  initialization if rigid registration fails.
* **Register each functional plane into the stack separately** (slice-to-volume).
  Handles tilted or unevenly spaced planes and drift between planes, which one rigid
  transform can't, at the cost of many more parameters fit to little data. Worth it
  only if rigid placement visibly fails.
* **Build the atlas on one fly's functional planes** instead of a high-res template.
  Needs no stacks, but every fly's position between the template's planes is then only
  known to ~10 µm, and registration between few-plane volumes is poorly constrained in
  z. This is what happens anyway when no training fly has a stack.
* **Require a stack for every fly.** Better registration, but the pipeline has to work
  without them, so stacks are used where they exist.
