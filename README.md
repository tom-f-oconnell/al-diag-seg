# Antennal lobe glomerulus segmentation: example code

Example pipeline for training a model to draw glomerulus ROIs from existing
per-z-slice ImageJ ROIs and responses to the diagnostic odor panel. It follows
the approach in the "Glomerulus ROI segmentation: recommended approach" doc. Its input
is the odor response database's ([hong-data](https://github.com/ejhonglab/hong-data))
ROI training download, as al_analysis leaves the data on tensor.

The odor response database can also fit models itself (its AL Segmentation tab), and
gives each fit's model to download, to run on your own flies as below.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:
```
uv sync   # Python 3.13 env with cellpose 4.x; training needs a GPU
```
Run the commands below with `uv run python ...`, or activate the env first. uv puts
it in `./.venv`, or in `./venv` if `UV_PROJECT_ENVIRONMENT=venv` is set. Python 3.14
won't work until antspyx ships wheels for it. Or install it into another environment with
pip (hong2p comes from GitHub too):
```
pip install "hong2p @ git+https://github.com/ejhonglab/hong2p" "al-diag-seg @ git+https://github.com/tom-f-oconnell/al-diag-seg"
```
which gives the commands `alseg-fit` (the same as `python run_pipeline.py`) and
`alseg-predict`.

## Running a model on your flies

A fit (`alseg-fit`, or the database's) writes its model to a folder (`model/`). To segment
flies of your own with it, give their motion-corrected data folders as al_analysis leaves
them (their diagnostic recordings' `trialmean_dff.tif` and `trial_frames_and_odors.json`,
and `mocorr_concat_avg.tif` or `mocorr_concat.tif`), and their raw data folders (voxel
sizes from `Experiment.xml`, and an `anat/` stack if there is one):
```
alseg-predict model/ --fly /local/tensor/Ann/analysis/2026-03-31/1 --raw /local/tensor/Ann/raw/2026-03-31/1 --out pred/
```
Each fly's `<fly>_RoiSet.zip` has its ROIs named by glomerulus (numbered where none
fits), per plane, as ImageJ and al_analysis read them. Its odors are matched to the
model's by their names as recorded; it needs all of them (the ones it lacks are listed).
It runs on a CPU too (slower). Flies with a `RoiSet.zip` of their own are also scored
against it. `alseg-predict -h` has the rest (e.g. `--export`, for flies of a database
download).

## Files

| File | What it does |
|---|---|
| `alseg/model.py` | `fit`: the whole model (Cellpose weights, PCA basis, identity template, atlas, odors); `Model.save`/`load` (a folder); `Model.predict` (a fly → ROIs of each kind, named); `Model.evaluate` (scores against a fly's own ROIs, per fly and per glomerulus). |
| `alseg/fly.py` | A fly from its folders as al_analysis leaves them (not a download), with its odors matched to a model's, to run the model on. |
| `alseg/cli.py` | The commands: `alseg-fit` (`run_pipeline.py`) and `alseg-predict`. |
| `alseg/olfdb.py` | Reads the database's ROI training download: per fly, its diagnostic recordings' `trialmean_dff.tif` one after another, lined up by odor with every other fly's (odors named alike by the database), its anatomy, its RoiSet's glomeruli (as the database tells them from other ROIs), voxel sizes from `Experiment.xml`, and its anatomical stack from ThorImage's `ChanA_*.tif` files. |
| `alseg/roi_io.py` | RoiSet.zip → `(Z, Y, X)` label volume, and back (`write_roiset`: a polygon per label and plane, named, which reads back to the same voxels). ROIs sharing a glomerulus name across slices become one 3D object. Falls back to linking by slice-to-slice overlap if ROIs have ImageJ's default numeric names. Overlapping voxels are returned separately. |
| `alseg/responses.py` | Per-voxel 24-odor response vectors (from a raw movie + onsets, or precomputed), local-correlation map, a PCA basis shared across flies, and the 3 input channels. |
| `alseg/metadata.py` | Voxel sizes from ThorImage `Experiment.xml` files, via [hong2p](https://github.com/ejhonglab/hong2p). |
| `alseg/segment.py` | Builds per-slice training data (eroded ROI cores), fine-tunes Cellpose-SAM, predicts slice by slice and stitches into 3D. Every fly's planes are resampled in-plane (never in z) to one pixel size for the model. |
| `alseg/refine.py` | Distances (ROI growth, neighbors, minimum size) are in µm, so ROIs don't grow into planes far away. `refine_with_prior`: refinement as energy minimization over voxel labels, with costs for tuning mismatch, `-log` atlas prior and neighbor disagreement (smoothness), solved by ICM. `refine_by_tuning`: the simpler tuning-only version: grows each ROI a little, then keeps only voxels that match its own tuning better than any neighbor's and are bright enough, and drops mixed border voxels. Also computes crosstalk metrics (coherence, neighbor tuning correlation, border leak). |
| `alseg/atlas.py` | Probabilistic atlas: registers training flies to a template (ANTsPy, Affine or SyN, in physical space with each fly's voxel size, through its anatomical stack if it has one, mirrored left-right first if that fits better), averages each glomerulus's warped mask into a probability map, and warps the atlas into a new fly. |
| `alseg/reference.py` | Adds a fixed reference atlas (labels only, e.g. Grabe et al. 2016 or hemibrain) to the training-fly atlas, aligned by its labels; see [below](#fixed-reference-atlases). |
| `alseg/identity.py` | Builds a per-glomerulus template (tuning, position, shape) from labeled flies and names new ROIs. Cost = tuning + position + shape (Mahalanobis on log volume and axis lengths) + atlas overlap + layout (offsets to neighbors vs. template offsets, re-solved iteratively). Poor matches stay unassigned. |
| `alseg/metrics.py` | IoU/Dice matching, precision/recall, split and merge counts. |
| `run_pipeline.py` | All of the above, with held-out flies: the model (`model/`), and per held-out fly its ROIs (label TIFFs and named RoiSets), identity CSVs; `summary.csv`, `glomeruli.csv`. |
| `alseg/synthetic.py` (`make_synthetic.py`), `smoke_test.py` | Fake flies as a database download (with numbered and tentative ROIs, an odor one fly lacks, and stacks of right ALs) plus a fake reference atlas, and runs of the pipeline with Cellpose swapped for a stand-in: as is, with two glomeruli unlabeled in the training flies, and that plus the reference; then a saved model run with `alseg-predict` on a fly as a download's and from its own folders (`python make_synthetic.py syn 8 && python smoke_test.py`, no GPU needed). |

## Data

Download it from the database's AL Segmentation tab (keeping the anatomical stacks, in it
unless unticked: flies register to each other through them), unzip it, and give its folder:
```
python run_pipeline.py olfdb_roi_training_2026-11-04/ --val Tom_2023-11-19_1 Sam_2026-03-31_1 --out results/
```

It has every file where it is on tensor, and `roi_training.json`, which says which file
is what. Per fly:

```
<its motion-corrected data folder>/<diagnostic recording>/trialmean_dff.tif            (n_odors, Z, Y, X) mean dF/F per odor
<its motion-corrected data folder>/<diagnostic recording>/trial_frames_and_odors.json  each trial's odor
<its motion-corrected data folder>/<a recording>/RoiSet.zip                            ImageJ ROIs, per plane
<its motion-corrected data folder>/mocorr_concat_avg.tif                               (Z, Y, X) average image
<its raw data folder>/<diagnostic recording>/Experiment.xml                            voxel sizes
<its raw data folder>/anat/                                                            optional: its anatomical stack
```

Flies are named `<top folder of their data on tensor>_<date>_<fly number>`. A fly's
responses are its diagnostic recordings' one after another, put in one order for every
fly by odor, keeping the odors every fly has (those left out are printed). ROIs are
taken as the database takes them: only glomeruli are trained on (named as one of the
glomeruli the database accepts, or several joined by `/`), and tentative ones (`DA1_t0`)
as that glomerulus, unless the fly has an ROI of it without the mark (`--no-tentative`
leaves them out). Numbered ROIs, and names ending in `+` or `?`, are left out.

Voxel sizes come from the `Experiment.xml` files. Flies can differ in voxel size, and
planes can be much further apart than pixels (e.g. 8–12 planes ~10 µm apart);
anatomical stacks are optional but improve registration between flies. They are as
recorded, while al_analysis mirrors right ALs' functional data left-right, so each is
placed on its fly's planes both ways and kept the way that fits better (printed). See
[design.md](design.md) for how this is handled, and the alternatives.

`python run_pipeline.py -h` lists every option, with defaults and examples. It saves the model to `model/` (see [above](#running-a-model-on-your-flies)). Per held-out fly it writes `*_pred.tif`, `*_refined_tuning.tif`, `*_refined_prior.tif` and each as a RoiSet named by glomerulus (`*_<kind>_RoiSet.zip`), the warped atlas (`*_atlas_maxprob.tif`), identity CSVs for both refinements; one `summary.csv` row per fly and ROI set (and its manual ROIs' crosstalk, to compare), and `glomeruli.csv`: per glomerulus, in how many held-out flies the final ROIs found it (IoU ≥ 0.5) and named it right. `--no-atlas` skips the priors.

A few voxels of some recordings' `trialmean_dff.tif` are NaN (al_analysis' dF/F where the
baseline is 0); they are taken as 0, and counted as they are read.

## Atlas priors: what they do and how to tune them

* **Order of operations.** Predict → name the raw ROIs (needs to know which glomerulus each ROI is before it can apply that glomerulus's prior) → refine with the prior → name again.
* **`w_prior`** sets how much the atlas can overrule the tuning. Leaving a voxel unassigned costs the same as joining an ROI at `min_corr` with a neutral prior of 0.5, so a high prior lowers the tuning evidence needed and a low prior raises it.
* **Identity weights** (`w_tuning`, `w_position`, `w_shape`, `w_atlas`, `w_layout`) are normalized by their sum, so `max_cost` keeps its meaning when you turn terms on or off.
* **Keep priors soft.** An atlas from a few flies is blurry; on synthetic data a strong prior (`w_prior` ≥ 1) shrinks or drops ROIs. Tune weights on held-out flies against both Dice and crosstalk.

Synthetic results (2 held-out flies, 6-fly atlas, ROIs that overshoot by 2 px as the
"prediction"; flies have 8–12 planes 8–12 µm apart, 0.9–1.2 µm pixels, and half have an
anatomical stack):

| Refinement | Recall | Dice | Border leak |
|---|---|---|---|
| none (overshooting prediction) | 1.0 | 0.86–0.87 | 0.13–0.14 |
| tuning only | 0.75–0.88 | 0.93–0.94 | 0.04–0.07 |
| tuning + atlas prior (`w_prior` 0.5) | 0.94–1.0 | 0.90–0.91 | 0.06–0.08 |

Manual ROIs have leak ≈ 0.03–0.04. The prior keeps (nearly) every ROI, and pure
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
* **Names must match** the training flies', but for upper and lower case (a reference's
  `DA4L` is taken as `DA4l`). Atlases differ in places: Grabe has VC3 and VP1 where the
  hemibrain has VC3l/m and VP1d/l/m. Glomeruli only in a reference whose names the
  database does not accept (Grabe's VP1) are listed when it is added: ROIs named after
  them would not be glomeruli to it.

On synthetic data, with G03 and G11 unlabeled in the training flies, atlas-based
naming got 28/32 ROIs right (0/4 of the two dropped glomeruli); with the reference,
30/32 (G03 in both flies; G11 left unassigned, as its shape varies a lot between the
fake flies, so one reference fly misplaces it by 10–30 µm). Aligning the Grabe atlas
to the hemibrain right AL (two different flies, in vivo vs. EM) left 5 µm median
centroid error, and shared glomeruli overlapped with median Dice 0.46 after the
affine and 0.63 after SyN.

## Things to check on your data

* **ROI names.** ROIs of one glomerulus on several planes (one name) are one 3D object; `roi_io._clean_name` also maps names like `DM2_z12` or `DM2 (3)` to `DM2`. Case is kept, as the database's names are case-sensitive (`DA4l`, `VA1d`).
* **Slice position.** Read from `z_position` (hyperstacks) or `position` (plain stacks), 1-based. An ROI with no position raises an error.
* **Channels.** Cellpose-SAM takes 3 channels: the average image, local correlation of tuning, PC1 of tuning. Try PCs 1–3 instead if the average image is weak.
* **Refinement trades size for purity.** On the synthetic data, tuning-only refinement brings border leak from ~0.13 (overshooting ROIs) to 0.04–0.07 (manual: 0.03–0.04), but drops ROIs whose cores are too small or dim. Tune `min_corr`, `margin` and `min_rel_amp` against crosstalk metrics, not Dice alone.
* **Voxel sizes.** Check the per-fly voxel sizes `run_pipeline.py` prints. Refinement distances (`grow_um`, `min_size_um3`) are in µm, so they carry over between flies with different resolutions.
* **Run-to-run differences.** ANTs registration uses all cores by default, which makes it differ slightly between runs, so close calls (whether a stack gets mirrored, which glomeruli a `--ref` fit leaves out) can change. `--deterministic` makes registration repeatable (a fixed seed and one thread): two runs of `python smoke_test.py --deterministic` wrote identical files, but took 2.4× as long (~230 s vs. 98 s).
