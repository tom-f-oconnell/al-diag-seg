# Antennal lobe glomerulus segmentation: example code

Example pipeline for training a model to draw glomerulus ROIs from existing
per-z-slice ImageJ ROIs and responses to the diagnostic odor panel. It follows
the approach in the "Glomerulus ROI segmentation: recommended approach" doc.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:
```
uv sync   # Python 3.13 env with cellpose 4.x; training needs a GPU
```
Run the commands below with `uv run python ...`, or activate the env first. uv puts
it in `./.venv`, or in `./venv` if `UV_PROJECT_ENVIRONMENT=venv` is set. Python 3.14
won't work until antspyx ships wheels for it.

## Files

| File | What it does |
|---|---|
| `alseg/roi_io.py` | RoiSet.zip → `(Z, Y, X)` label volume. ROIs sharing a glomerulus name across slices become one 3D object. Falls back to linking by slice-to-slice overlap if ROIs have ImageJ's default numeric names. Overlapping voxels are returned separately. |
| `alseg/responses.py` | Per-voxel 24-odor response vectors (from a raw movie + onsets, or precomputed), local-correlation map, a PCA basis shared across flies, and the 3 input channels. |
| `alseg/metadata.py` | Voxel sizes from ThorImage `Experiment.xml` files, via [hong2p](https://github.com/ejhonglab/hong2p). |
| `alseg/segment.py` | Builds per-slice training data (eroded ROI cores), fine-tunes Cellpose-SAM, predicts slice by slice and stitches into 3D. Every fly's planes are resampled in-plane (never in z) to one pixel size for the model. |
| `alseg/refine.py` | Distances (ROI growth, neighbors, minimum size) are in µm, so ROIs don't grow into planes far away. `refine_with_prior`: refinement as energy minimization over voxel labels, with costs for tuning mismatch, `-log` atlas prior and neighbor disagreement (smoothness), solved by ICM. `refine_by_tuning`: the simpler tuning-only version: grows each ROI a little, then keeps only voxels that match its own tuning better than any neighbor's and are bright enough, and drops mixed border voxels. Also computes crosstalk metrics (coherence, neighbor tuning correlation, border leak). |
| `alseg/atlas.py` | Probabilistic atlas: registers training flies to a template (ANTsPy, Affine or SyN, in physical space with each fly's voxel size, through its high-res stack if it has one), averages each glomerulus's warped mask into a probability map, and warps the atlas into a new fly. |
| `alseg/reference.py` | Adds a fixed reference atlas (labels only, e.g. Grabe et al. 2016 or hemibrain) to the training-fly atlas, aligned by its labels; see [below](#fixed-reference-atlases). |
| `alseg/identity.py` | Builds a per-glomerulus template (tuning, position, shape) from labeled flies and names new ROIs. Cost = tuning + position + shape (Mahalanobis on log volume and axis lengths) + atlas overlap + layout (offsets to neighbors vs. template offsets, re-solved iteratively). Poor matches stay unassigned. |
| `alseg/metrics.py` | IoU/Dice matching, precision/recall, split and merge counts. |
| `run_pipeline.py` | All of the above, with held-out flies, writing label TIFFs, per-fly identity CSVs and `summary.csv`. |
| `make_synthetic.py`, `smoke_test.py` | Fake flies in the expected layout plus a fake reference atlas, and runs of the pipeline with Cellpose swapped for a stand-in: as is, with two glomeruli unlabeled in the training flies, and that plus the reference (`python make_synthetic.py syn 8 && python smoke_test.py`, no GPU needed). |

## Data layout

```
data/fly01/anatomy.tif            (Z, Y, X)
data/fly01/responses.tif          (n_odors, Z, Y, X)  mean dF/F per odor, motion-corrected
data/fly01/RoiSet.zip
data/fly01/Experiment.xml         optional: ThorImage metadata of that recording
data/fly01/hires/anatomy.tif      optional: higher resolution anatomical stack of the fly
data/fly01/hires/Experiment.xml   its metadata (required if fly01/Experiment.xml exists)
```

```
python run_pipeline.py data/ --val fly07 fly13 fly21 --out results/
```

Voxel sizes come from the `Experiment.xml` files if there are any. Then every fly needs
one, as does every `hires/` stack, and `--spacing`/`--hires-spacing` are errors. The
XML must describe the TIFF as is (not cropped or binned). Without XML files, give
`--spacing Z Y X` (and `--hires-spacing` for `hires/` stacks) in µm. Flies can differ in
voxel size, and planes can be much further apart than pixels (e.g. 8–12 planes
~10 µm apart); `hires/` stacks are optional but improve registration between flies. See
[design.md](design.md) for how this is handled, and the alternatives.

`python run_pipeline.py -h` lists every option, with defaults and examples. Per held-out fly it writes `*_pred.tif`, `*_refined_tuning.tif`, `*_refined_prior.tif`, the warped atlas (`*_atlas_maxprob.tif`), identity CSVs for both refinements, and one `summary.csv` row per ROI set. `--no-atlas` skips the priors.

## Anatomical priors: what they do and how to tune them

* **Order of operations.** Predict → name the raw ROIs (needs to know which glomerulus each ROI is before it can apply that glomerulus's prior) → refine with the prior → name again.
* **`w_prior`** sets how much the atlas can overrule the tuning. Leaving a voxel unassigned costs the same as joining an ROI at `min_corr` with a neutral prior of 0.5, so a high prior lowers the tuning evidence needed and a low prior raises it.
* **Identity weights** (`w_tuning`, `w_position`, `w_shape`, `w_atlas`, `w_layout`) are normalized by their sum, so `max_cost` keeps its meaning when you turn terms on or off.
* **Keep priors soft.** An atlas from a few flies is blurry; on synthetic data a strong prior (`w_prior` ≥ 1) shrinks or drops ROIs. Tune weights on held-out flies against both Dice and crosstalk.

Synthetic results (2 held-out flies, 6-fly atlas, ROIs that overshoot by 2 px as the
"prediction"; flies have 8–12 planes 8–12 µm apart, 0.9–1.2 µm pixels, and half have a
`hires/` stack):

| Refinement | Recall | Dice | Border leak |
|---|---|---|---|
| none (overshooting prediction) | 1.0 | 0.86–0.87 | 0.13–0.14 |
| tuning only | 0.75–0.88 | 0.93–0.94 | 0.04–0.07 |
| tuning + atlas prior (`w_prior` 0.5) | 1.0 | 0.89–0.91 | 0.06–0.08 |

Manual ROIs have leak ≈ 0.03–0.04. The prior keeps every ROI (recall 1.0), and pure
tuning is stricter about borders but drops some ROIs. In identity matching, tuning alone swapped the two nearly identically tuned glomeruli, while any position term (centroid, atlas, or layout) fixed it; all terms together got 32/32. The synthetic data are too easy to tell those terms apart, so compare them on your flies.

With 20–30 flies, hold out 4–5 flies per fold and rotate (5-fold over flies). Never split slices of the same fly between train and validation.

## Fixed reference atlases

`--ref` adds atlases from other sources to the one built from your training flies, e.g.
the exports in [al_atlas](https://github.com/ejhonglab/al_atlas):
```
python run_pipeline.py data/ --val fly07 --ref ~/src/al_atlas/grabe2016
```
A reference folder has `labels.tif` (voxel size in its ImageJ metadata) and one `.csv`
with `label` and `glomerulus` columns, and optionally `side`.

Of the al_atlas exports, `grabe2016` is the better one: several hemibrain meshes in
that export are fragments.

* **Alignment is by labels, not images.** A linear map is fit to the centroids of
  glomeruli named in both (needs ≥ 6; outliers are dropped and listed): a full affine
  or a similarity transform, whichever predicts left-out glomeruli better. Then
  `--ref-reg SyN` (default) registers the two sets of label maps. Your atlas is
  already registered to each new fly, so the reference follows it. Voxel sizes must
  be right, since the fit is in µm.
* **Side.** For a reference with both ALs, the side that fits better is used, mirrored
  if needed (both are printed); `--ref-side` forces one.
* **Glomeruli you never labeled** get their own atlas channels, and can be named from
  position, layout and atlas overlap alone (tuning and shape costs set to a neutral
  `unknown_cost`). Shared ones are averaged in as `--ref-weight` (default 1) extra flies.
* **Names must match** after the same cleanup as ROI names (`DA4l` → `DA4L`). Atlases
  differ in places: Grabe has VC3 and VP1 where the hemibrain has VC3l/m and VP1d/l/m.

On synthetic data, with G03 and G11 unlabeled in the training flies, atlas-based
naming got 28/32 ROIs right (0/4 of the two dropped glomeruli); with the reference,
30/32 (G03 in both flies; G11 left unassigned, as its shape varies a lot between the
fake flies, so one reference fly misplaces it by 10–30 µm). Aligning the Grabe atlas
to the hemibrain right AL (two different flies, in vivo vs. EM) left 5 µm median
centroid error, and shared glomeruli overlapped with median Dice 0.46 after the
affine and 0.63 after SyN.

## Things to check on your data

* **ROI names.** `roi_io._clean_name` maps names like `DM2_z12` or `DM2 (3)` to `DM2`. Edit it to match your naming. ROIs in one zip need unique names (ImageJ enforces this), so the slice suffix is expected.
* **Slice position.** Read from `z_position` (hyperstacks) or `position` (plain stacks), 1-based. An ROI with no position raises an error.
* **Channels.** Cellpose-SAM takes 3 channels: anatomy, local correlation of tuning, PC1 of tuning. Try PCs 1–3 instead if anatomy is weak.
* **Refinement trades size for purity.** On the synthetic data, tuning-only refinement brings border leak from ~0.13 (overshooting ROIs) to 0.04–0.07 (manual: 0.03–0.04), but drops ROIs whose cores are too small or dim. Tune `min_corr`, `margin` and `min_rel_amp` against crosstalk metrics, not Dice alone.
* **Voxel sizes.** Check the per-fly voxel sizes `run_pipeline.py` prints. Refinement distances (`grow_um`, `min_size_um3`) are in µm, so they carry over between flies with different resolutions.
