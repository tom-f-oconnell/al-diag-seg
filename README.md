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

## What a fit and a prediction read from each fly

A fit reads six kinds of file per fly (the database's fits in place on tensor, `alseg-fit`
from a download of them) and reduces them to a 3-channel image per plane plus label
volumes; prediction reads the same files minus the ROIs. As of 2026-11-13 the database's
AL Segmentation tab lists the same 86 flies its fit 7 used (62 Sam's, 24 Tom's; 82 trained
on, 4 held out), so the numbers below are from fit 7 and the files as they were that day.

### Training: files read per fly

| File | Dimensions | Becomes |
|---|---|---|
| `<analysis>/<date>/<fly>/<recording>/trialmean_dff.tif`, one per diagnostic recording (3 per fly) | (odor, Z, Y, X) float32, mean dF/F per odor. 10, 10 and 4–5 odors; Z = 7–11 planes (10 in 58 flies); Y = X = 192 | Joined along odor (24–25), cut to the 20 odors every fly has: `responses` (20, Z, 192, 192) |
| `<recording>/trial_frames_and_odors.json` | one entry per trial | Odor names only, to label the odor axis. The database reads it; al-diag-seg does not open it in a fit |
| `<fly>/mocorr_concat_avg.tif` | (Z, Y, X) float32 | `average`. Only Tom's 24 flies have it; for Sam's 62, the database takes the mean over time of `mocorr_concat.tif` (time, Z, Y, X) and caches it in its `data/roi_training/` |
| `<recording>/RoiSet.zip` (the real file, in a diagnostic recording folder) | ImageJ polygons, each on one plane; 75–175 per fly | Two label volumes (Z, Y, X), below |
| `<raw>/<date>/<fly>/<recording>/Experiment.xml` | – | Voxel size (z, y, x): 10 µm × 0.47–0.58 µm (median 0.553) |
| `<raw>/<date>/<fly>/anat/ChanA_*.tif` and its `Experiment.xml` (14 flies, all Tom's, Apr 2023–Jan 2024) | 181–281 planes × 5 frames of 384 × 384 uint16, averaged over frames to (Z, Y, X) at 0.5 × ~0.25 µm | Registration image for the atlas |

The TIFFs' own axis labels call the odor axis T. NaN/inf dF/F values are set to 0.

### Training: what is built from them

- **`channels` (3, Z, Y, X)**, each scaled to 0–1 by the volume's 1st–99.5th percentiles:
  - 0: the average image.
  - 1: correlation of each voxel's 20-odor response with its 3 × 3 in-plane neighbours.
  - 2: the first principal component of the per-voxel response vectors.
- **`labels` (Z, Y, X)**: named glomeruli and `_t<n>` tentative ones, same name across
  planes as one 3D label. This is 7,013 of the 10,730 outlines, 34–43 glomeruli per fly.
- **`seg_labels` (Z, Y, X)**: those plus numbered and `?` ROIs (8,240 outlines). `+` ROIs
  and `AL`/`AL_plane<n>` outlines are left out.

Each step then uses:

1. **PCA basis** (mean (20,), components (3, 20)): training flies' `responses` at voxels
   inside `labels`.
2. **Cellpose fine-tuning**: one example per plane, a (Y, X, 3) image from `channels` and a
   (Y, X) mask from `seg_labels`, eroded 1 px.
   - Planes are resampled in-plane to 0.553 µm (192 px becomes 163–202) and never mixed
     across Z.
   - The 82 training flies give 792 planes, doubled to 1,584 by one flipped/rotated copy
     each.
   - The 4 held-out flies' 39 planes give only the validation loss.
3. **Naming template**: per glomerulus (47), mean tuning (20,), centroid (3,) in µm and
   shape (4,), from training flies' `responses`, `labels` and voxel size.
4. **Atlas**: template (91, 96, 96) at 1 µm and probability maps (55, 91, 96, 96).
   - Each training fly's anat stack, or its `average` if it has none, is registered to one
     fly's stack.
   - Its `labels` are warped in.
   - 8 of the 55 glomeruli come from the Grabe 2016 reference, not from flies.

Held-out flies are in none of steps 1, 3 and 4.

### Prediction: files read

You give the fly's motion-corrected folder and its raw folder (`load_fly` in
`alseg/fly.py`).

- **Responses**: every subfolder with `trialmean_dff.tif` and
  `trial_frames_and_odors.json` where more than half the odors are the model's 20. They
  are reordered to (20, Z, Y, X); extra odors are ignored, and a fly may lack some of the
  model's (it needs 3 of them: [missing_diagnostics.md](missing_diagnostics.md)).
- **Average**: `mocorr_concat_avg.tif`, else the mean of `mocorr_concat.tif`, computed in
  memory.
- **Voxel size**: `Experiment.xml` of one of those recordings in the raw folder.
- **Anat stack**: the first `anat*/` folder in the raw folder, if there is one.
- **`RoiSet.zip`**: optional, used only to score the result, never as input.

From the model folder it uses `segmenter` (the weights), `pca.npz`, `template.json`,
`atlas.npz` and `model.json`.

The same `channels` go through Cellpose plane by plane at 0.553 µm. `responses` are used
again to join planes into 3D ROIs and, with the template and atlas, to name them.

### Never read

By a fit, or a prediction as the model makes it: the movies themselves (apart from that
mean), per-trial `trial_dff.tif`, non-diagnostic recordings such as megamat, and the 4–5
odors per fly outside the common 20. What is run on a prediction's ROIs afterwards reads
more, of every recording on the fly's grid: the cleaning steps `trim` and `--split`, each
recording's `trial_dff.tif`; and the look for glomeruli the ROIs miss (`alseg.nmf`,
[below](#glomeruli-the-rois-miss)), each recording's `mocorr.tif` (every time point of every
trial).

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
fits: none costing `--max-cost` or less, 0.6 unless the model was fit with another), per
plane, as ImageJ and al_analysis read them; `<fly>_identity_terms.json` has what each name
cost (each term's value for the ROI's 5 best glomeruli, and the weights), and how the atlas was
moved to where the fly's ROIs say its glomeruli are, before they were named with it (`alignment`). Its odors are matched to the
model's by their names as recorded; it needs all of them (the ones it lacks are listed).
A fly may lack some of the model's odors (it needs 3 of them): its ROIs are then joined and named by those it has, exactly
as by a model fit to the same flies with those odors, and found as well as by such a model if it has the odors Cellpose's
images are made from (`Options.image_odors`). [missing_diagnostics.md](missing_diagnostics.md) has how, what it costs to lack
those too, and what it would take to do without the diagnostic panel.
It runs on a CPU too (slower). Flies with a `RoiSet.zip` of their own are also scored
against it. `alseg-predict -h` has the rest (e.g. `--export`, for flies of a database
download).

## Files

| File | What it does |
|---|---|
| `alseg/model.py` | `fit`: the whole model (Cellpose weights, PCA basis, identity template, atlas, odors; `Options.image_odors`: the odors Cellpose's images are made from, if not all); `Model.save`/`load` (a folder); `Model.predict` (a fly → ROIs of each kind, named, by the odors of the model it has); `Model.evaluate` (scores against a fly's own ROIs, per fly and per glomerulus). |
| `alseg/fly.py` | A fly from its folders as al_analysis leaves them (not a download), with its odors matched to a model's, to run the model on. |
| `alseg/cli.py` | The commands: `alseg-fit` (`run_pipeline.py`) and `alseg-predict`. |
| `alseg/olfdb.py` | Reads the database's ROI training download: per fly, its diagnostic recordings' `trialmean_dff.tif` one after another, lined up by odor with every other fly's (odors named alike by the database), its anatomy, its RoiSet's glomeruli (as the database tells them from other ROIs), voxel sizes from `Experiment.xml`, and its anatomical stack from ThorImage's `ChanA_*.tif` files. |
| `alseg/roi_io.py` | RoiSet.zip → `(Z, Y, X)` label volume, and back (`write_roiset`: a polygon per label and plane, named, which reads back to the same voxels). ROIs sharing a glomerulus name across slices become one 3D object. Falls back to linking by slice-to-slice overlap if ROIs have ImageJ's default numeric names. Overlapping voxels are returned separately. |
| `alseg/responses.py` | Per-voxel 24-odor response vectors (from a raw movie + onsets, or precomputed), local-correlation map, a PCA basis shared across flies, and the 3 input channels. |
| `alseg/metadata.py` | Voxel sizes from ThorImage `Experiment.xml` files, via [hong2p](https://github.com/ejhonglab/hong2p). |
| `alseg/segment.py` | Builds per-slice training data (eroded ROI cores), fine-tunes Cellpose-SAM, predicts slice by slice and joins the slices' masks into 3D objects (by overlap and odor tuning). Every fly's planes are resampled in-plane (never in z) to one pixel size for the model. |
| `alseg/clean.py`, `clean_trim.py`, `clean_margin.py`, `evidence.py` | Steps that keep neighbouring glomeruli out of Cellpose's ROIs (`--clean`), only removing pixels, on each plane: `trim` cuts an ROI's edge back, in bands by distance from its neighbour, while the band carries more than a share of the neighbour's core's tuning (by every recording's per-trial dF/F, `trial_dff.tif`: `evidence.py`); `margin` keeps every ROI a distance from the others' outlines (and inside its own); `core` keeps the pixels Cellpose's cell probability is higher at. `--split` cuts a plane's masks that hold two glomeruli tuned differently in two, and joins the planes' masks again (`<fly>_split`, named again; Cellpose's own stay as `<fly>_pred`), before the other steps. Each step alone is written and scored too (`<fly>_clean_<step>`), and the steps asked for are the final ROIs (`clean`). `Model.refine` does the same later, from Cellpose's ROIs and their names (as a prediction keeps them), with the atlas on the fly's grid (to name ROIs split again) and Cellpose's cell probabilities (for `core`). |
| `alseg/nmf.py` | Looks in a fly's movies for glomeruli its ROIs miss (`find`; [below](#glomeruli-the-rois-miss)): each plane's movie factorized (non-negative, on the GPU if there is one) as a footprint per ROI and free ones for what those leave unexplained, a free one's place a candidate if it is round enough, in the lobe, responds reliably and not as a mixture of its neighbours; joined across planes, and to the ROI it is more of. `check` also cleans the candidates as the ROIs were and names them, with the ROIs' names kept or all named again together (`identity.assign_identities`'s `fixed`). |
| `alseg/refine.py` | Distances (ROI growth, neighbors, minimum size) are in µm, so ROIs don't grow into planes far away. `refine_with_prior`: refinement as energy minimization over voxel labels, with costs for tuning mismatch, `-log` atlas prior and neighbor disagreement (smoothness), solved by ICM. `refine_by_tuning`: the simpler tuning-only version: grows each ROI a little, then keeps only voxels that match its own tuning better than any neighbor's and are bright enough, and drops mixed border voxels. Also computes crosstalk metrics (coherence, neighbor tuning correlation, border leak), and (`purity_report`, `breadth`) how far ROIs keep from each other on a plane, how much of its nearest neighbor's tuning an ROI's edge facing it and its whole carry, and how broadly predicted ROIs respond against the manual ones they match: all in `summary.csv`, for each kind of ROIs and the manual ones. |
| `alseg/atlas.py` | Probabilistic atlas: registers training flies to a template (ANTsPy, Affine or SyN, in physical space with each fly's voxel size, through its anatomical stack if it has one, mirrored left-right first if that fits better), averages each glomerulus's warped mask into a probability map, and warps the atlas into a new fly. |
| `alseg/reference.py` | Adds a fixed reference atlas (labels only, e.g. Grabe et al. 2016 or hemibrain) to the training-fly atlas, aligned by its labels; see [below](#fixed-reference-atlases). |
| `alseg/align.py` | Moves the atlas on a fly's grid to where the fly's ROIs say its glomeruli are, before they are named with it (`--align`): an affine map fit to the ROIs that odor tuning alone names (or to those named with the atlas, after a shift), whatever a few wrong ones say; see [below](#things-to-check-on-your-data). |
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

`python run_pipeline.py -h` lists every option, with defaults and examples. It saves the model to `model/` (see [above](#running-a-model-on-your-flies)). Per held-out fly it writes `*_pred.tif`, `*_refined_tuning.tif`, `*_refined_prior.tif` and each as a RoiSet named by glomerulus (`*_<kind>_RoiSet.zip`), the warped atlas (`*_atlas_maxprob.tif`), identity CSVs for each; one `summary.csv` row per fly and ROI set (matched to the manual ROIs at IoU ≥ 0.5 as 3D objects, and plane by plane: `*_2d`, scored as drawn on some planes only: see [below](#things-to-check-on-your-data); and its manual ROIs' crosstalk, to compare), and `glomeruli.csv`: per glomerulus, in how many held-out flies the final ROIs (Cellpose's own, named, unless `--final refined`) found it (IoU ≥ 0.5) and named it right. `--no-atlas` skips the priors.

A few voxels of some recordings' `trialmean_dff.tif` are NaN (al_analysis' dF/F where the
baseline is 0); they are taken as 0, and counted as they are read.

## Atlas priors: what they do and how to tune them

* **Order of operations.** Predict → name the raw ROIs → move the atlas to where the ROIs say the fly's glomeruli are, and name them again (`--align`, [below](#things-to-check-on-your-data)) → refine with the prior (it needs to know which glomerulus each ROI is before it can apply that glomerulus's prior) → name again.
* **`w_prior`** sets how much the atlas can overrule the tuning. Leaving a voxel unassigned costs the same as joining an ROI at `min_corr` with a neutral prior of 0.5, so a high prior lowers the tuning evidence needed and a low prior raises it.
* **Identity weights** (`w_tuning`, `w_position`, `w_shape`, `w_atlas`, `w_layout`) are normalized by their sum, so `max_cost` (`--max-cost`, saved with a model: an ROI no glomerulus costs less for stays numbered) keeps its meaning when you turn terms on or off. `<fly>_identity_terms*.json` has each term's value, before weighting, for every ROI's 5 best glomeruli, and which other ROI was given a name it did not get.
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
tuning is stricter about borders but drops some ROIs. In identity matching, tuning alone swapped the two nearly identically tuned glomeruli, while any position term (centroid, atlas, or layout) fixed it; all terms together got 32/32 (with the atlas as registered, `--align none`; moved onto the fly's ROIs, as by default, 31/32 in one run: these flies' atlas is registered well already, so moving it only adds a little noise). The synthetic data are too easy to tell those terms apart, so compare them on your flies.

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
* **What Cellpose trains on.** Every ROI outlining one glomerulus, named or not (`olfdb.segment_keeper`): numbered ROIs and uncertain names (`DL2d?`) too, which in the database's flies are as many as the named ones; left out, they would be taught as background. Not hong2p's maximal-extent ROIs (`DM5+`, which may hold neighbours' signal), nor outlines of the whole AL (`AL_plane3`). Naming, the atlas and scores use the named ones only.
* **Slice position.** Read from `z_position` (hyperstacks) or `position` (plain stacks), 1-based. An ROI with no position raises an error.
* **Channels.** Cellpose-SAM takes 3 channels: the average image, local correlation of tuning, PC1 of tuning. Try PCs 1–3 instead if the average image is weak.
* **Refinement trades size for purity.** On the synthetic data, tuning-only refinement brings border leak from ~0.13 (overshooting ROIs) to 0.04–0.07 (manual: 0.03–0.04), but drops ROIs whose cores are too small or dim. Tune `min_corr`, `margin` and `min_rel_amp` against crosstalk metrics, not Dice alone. On real flies (the database's first fit, to 82 flies), Cellpose's ROIs already leaked little (0.04–0.06; manual 0.03–0.08), and refinement only shrank them and broke them into pieces (of the 4 held-out flies' 325 manual ROIs, plane by plane, 46% found at IoU ≥ 0.5 before, 29% after; 380 pieces per plane became 1,289): voxel by voxel, trial-mean tuning is noisy, so many inside an ROI fail `min_corr`, and its dimmer edges fail `min_rel_amp`. So the final ROIs are Cellpose's own, named, unless `--final refined`.
* **Glomeruli drawn on some planes only.** ROIs are often drawn only on a glomerulus' clearest planes, so a predicted ROI where nothing is drawn may be right. Scores count against an ROI only what the manual ones can say is wrong: matched as a 3D object, its voxels on planes where the manual glomerulus has no outline count only where they lie in another manual glomerulus' outline; and "right" (precision), as 3D objects and plane by plane, is of the predicted ROIs that are matched or lie mostly in manual outlines (`n_scored`, `n_scored_2d` in `summary.csv`). "Found" (recall) is as before. Naming is checked against the manual glomerulus an ROI overlaps by the same rule. `Model.evaluate(partly_drawn=False)` scores strictly, as before.
* **How Cellpose's masks are made.** With Cellpose's own thresholds (cell probability 0, flow 0.4), a model trained on eroded ROI cores makes too few ROIs, too small. The defaults are a cell probability threshold of −2 (`--cellprob-threshold`) and a flow threshold of 0.8 (`--flow-threshold`), with masks not grown back (`--dilate`), picked together with how the planes' masks are joined (below) for a later fit of the database (the same 82 flies, 30 epochs), on its 4 held-out flies and 8 it trained on, scored as partly drawn (above). Picked on 11 of those flies and scored on the 12th, the same settings won every time. Picked on the 8 training flies, on the held-out flies they found 60% of the manual ROIs plane by plane and made 85% of the scored ones right, and as 3D objects 50% and 71%; the settings before them (−1 and 0.4, grown back by 1 px, joined at 0.3 and 0.5, below) 51%, 94%, 46% and 66%. So they find more glomeruli, but more of their ROIs over drawn glomeruli are wrong, and they make more ROIs where nothing is drawn, which the manual ROIs can't check. Scored strictly, those earlier settings were best (plane by plane, 50% found and 59% right, from Cellpose's 44% and 50%). The defaults are saved with a model, and models saved before they were options take them too. The best settings depend on the model and on how it is scored, so to try others on flies of your own, `alseg-predict` takes them as well.
* **How the planes' masks are joined into 3D objects.** Planes 10 µm apart make a glomerulus' outlines change size and shift from plane to plane, and neighbouring glomeruli overlap between planes. Cellpose's own stitching (each mask to its best match by IoU in the plane before, if ≥ 0.25, so several can join one) joined 93% of the overlapping masks of one glomerulus in adjacent planes, but also 14% of those of two. Instead, masks of adjacent planes are matched one to one, where they overlap by at least a fraction of the smaller one (0.2, `--join-overlap`) and their mean odor tunings correlate (0.7, `--join-corr`): 91% and 10% of those pairs pass. Scored strictly, on 12 flies of the same fit (with the earlier settings above, joined at 0.3 and 0.5) and picked on 11 and scored on the 12th in turn, this found 42% of the manual glomeruli as 3D objects and made 50% right, from Cellpose's 39% and 47%. Scored as partly drawn, on the held-out flies, it found 46%, where joining them perfectly (by the manual ROIs) would find 48%. `--join cellpose` joins as Cellpose does.
* **Where the atlas is on a fly.** Registering images leaves the atlas some µm from a fly's glomeruli. So the atlas on a fly's grid is moved to where the fly's ROIs say its glomeruli are, and the ROIs are named with it there (`alseg/align.py`, `Model.align`; `--align affine`, the default, `shift` or `none`, saved with a model; models saved before it was an option take the default).
  * **How.** One map for the whole fly, fit to ROIs' centres against where the atlas expects their glomeruli. An ROI named wrong lies far from where the others put its glomerulus and gets no weight (reweighted least squares). An affine map (also scale, tilt and shear) has its linear part held to the identity as by 5 more ROIs lying where expected, and one that would stretch the atlas over twofold is not believed.
  * **Which ROIs.** For an affine map, those that odor tuning alone names (each glomerulus the ROI tuned most like it, where their tunings correlate by 0.7 or more), wherever they are. The ROIs named with the atlas as registered are those near where it expects them, so a map fit to them falls short. In a fly with fewer than 18 of the model's odors, or where the map is not believed, it is those after all: a shift, the ROIs named again, then the affine map. A fly with fewer than 5 such ROIs keeps the atlas as registered.
  * **Measured** on the 75 flies of one experimenter that a fit of the database (its fit 9) predicted, was not trained on, and that have ROIs of their own: 2,319 manual glomeruli. A manual ROI's centre is a median 13 µm from where the atlas as registered expects its glomerulus: 7 µm in the plane, 8 µm along z (planes are 10 µm apart). "Named right": the predicted ROI overlapping a manual glomerulus most (IoU over 0.3) has its name. First, what kind of map, each fit once (by the comparison's own code, a little different from `align.fit`), to the ROIs named with the atlas or to the fly's manual ROIs:

    | Atlas moved by | A left-out manual ROI from where a fit to the fly's other manual ROIs puts it (µm: median, rms) | Named right, fit to the ROIs named with the atlas | ROIs named on another manual glomerulus | Named right, fit to the fly's manual ROIs |
    |---|---|---|---|---|
    | nothing | 13.2, 15.6 | 755 | 121 | |
    | a shift | 9.6, 12.4 | 809 | 116 | 832 |
    | an affine map in the plane, a shift along z | 9.6, 11.2 | 819 | 125 | 936 |
    | an affine map | 6.8, 8.8 | 846 | 129 | 1,011 |
    | an affine map, then a smooth warp (the ROIs' residuals, over 30 µm) | 6.6, 8.6 | 845 | 128 | 1,017 |

  * **An affine map, with z in it, and no more.** The terms with z in them (the atlas tilted and scaled along z) are most of what it gains over a shift. A warp on top places a left-out glomerulus 0.2 µm better and names nothing more, even fit to the fly's manual ROIs.
  * **Then, which ROIs to fit it to** (`align.fit`'s maps):

    | Affine map fit to | Named right | ROIs named on another manual glomerulus | ROIs named |
    |---|---|---|---|
    | (the atlas as registered) | 755 | 121 | 1,530 |
    | the ROIs named with the atlas | 841 | 128 | 1,716 |
    | those, after a shift | 851 | 128 | 1,737 |
    | those, after a shift, and once more | 865 | 133 | 1,760 |
    | the ROIs named under a looser `--max-cost` (0.8) | 874 | 129 | 1,765 |
    | the ROIs that tuning alone names; else as two rows up (what is done) | 897 | 127 | 1,763 |
    | the fly's manual ROIs | 970 | 110 | 1,821 |

  * **What it gains.** 19% more glomeruli named right (more in 45 flies, fewer in 6) and 5% more ROIs named on another, as 15% more ROIs are named. At a `--max-cost` of 0.55 it names 796 right and 71 on another: 5% more right than the atlas as registered names at 0.6, with 41% fewer wrong. Of the 75 flies, 62 were moved by an affine map fit to tuning-named ROIs, 7 by one fit to the ROIs named with the atlas, 3 by a shift, and 3 not at all.
  * **Where it gains most.** Most maps are small: the atlas' middle moved by a median 8 µm, turned 5° in the plane and tilted 11°. But in three flies registration had left the atlas turned some 55° in the plane: it named 0, 1 and 6 of their glomeruli right, and moved, 13, 17 and 17. A fit to the ROIs named with the atlas cannot find such a move, as hardly any are named right to begin with.
  * **Tuning alone needs most of the odors.** The same flies named by a random 12, 14 or 16 of their 22 or 23 odors: the map fit to tuning-named ROIs names fewer right than the one fit to the ROIs named with the atlas (742 against 780 at 16). By 18 or 20 odors it names more (818 against 795, and 854 against 829), with fewer wrong. Hence 18.
  * **What it leaves.** Fit to the fly's manual ROIs, the map would name 970 right. The ROIs it is fit to are a median 16 a fly, and a little over half of them (56%) lie on the manual ROI of that name.
  * `<fly>_identity_terms*.json` has how each fly's atlas was moved (`alignment`: the map, how many ROIs it was fit to and which, each one's weight, and their median distance from where the atlas expected them before and after); `<fly>_atlas_maxprob.tif` is the moved atlas.
* **Voxel sizes.** Check the per-fly voxel sizes `run_pipeline.py` prints. Refinement distances (`grow_um`, `min_size_um3`) are in µm, so they carry over between flies with different resolutions.
* **Run-to-run differences.** ANTs registration uses all cores by default, which makes it differ slightly between runs, so close calls (whether a stack gets mirrored, which glomeruli a `--ref` fit leaves out) can change. `--deterministic` makes registration repeatable (a fixed seed and one thread): two runs of `python smoke_test.py --deterministic` wrote identical files, but took 2.4× as long (~230 s vs. 98 s).

### Keeping neighbouring glomeruli out of ROIs

On the 4 held-out flies of the database's fit 7 (in-plane masks, on the diagnostic recordings' trial-mean dF/F; the steps on
the fit's own ROIs, at their defaults; `summary.csv`'s scores, means over the flies):

| ROIs | gap to the nearest ROI (um) | touching | a neighbour's share of the edge facing it | of the whole ROI | area kept |
|---|---|---|---|---|---|
| manual | 2.0 | 0% | 0.14 | 0.03 | |
| Cellpose's | 0.5 | 48% | 0.30 | 0.04 | 100% |
| `trim` | 2.1 | 2% | 0.07 | 0.01 | 89% |
| `margin` | 2.0 | 0% | 0.13 | 0.01 | 93% |
| `core` | 3.0 | 10% | 0.10 | 0.00 | 68% |
| `trim` then `margin` | 2.2 | 0% | 0.07 | 0.01 | 88% |

No ROI is lost by any step. `trim`'s share is measured on data its evidence includes (the diagnostic trials): with evidence of the
other recordings alone it is 0.14. `margin` is purer than `core` at equal area kept (cell probability stays high between touching
masks). `--split` splits 4 of the 11 masks holding at least 40% of two manual ROIs, and 12 of the 266 holding one (mostly cutting
off parts where nothing is drawn); the other 7 are tuned too alike to tell from one glomerulus.

### Glomeruli the ROIs miss

Cellpose outlines what it sees in a fly's average image and its odors' mean responses, and on the database's held-out flies its
ROIs match a little over half of the manual in-plane outlines (IoU ≥ 0.5). `alseg.nmf.find(labels, fly)` looks in the fly's
movies (`mocorr.tif` of every recording: each trial's time points less its baseline) for the rest, and returns candidates to
review, never ROIs to use as they are (the module's docstring says how; `alseg.nmf.check(model, fly, labels, names, prior)` also
cleans and names them). On the 4 held-out flies of the database's fit 7 (36 planes, 325 manual in-plane outlines, of which the
ROIs match 183 and leave 57 at IoU < 0.2), at its defaults:

| candidates | how many | on a manual outline the ROIs did not match (IoU ≥ 0.3 with it) | where none is drawn |
|---|---|---|---|
| glomeruli added (ROIs of their own) | 9 | 2 | 7 |
| more of an ROI, on a plane it had no outline on | 16 | 6 | 10 |
| parts cut off ROIs (`clean_trim.split`'s test) | 27 | 4 of masks over two manual outlines | 23 |

Where a "more of an ROI" candidate and the ROI it is more of (on the plane next to it) both lie on manual outlines (5 of the 16),
those are of the same glomerulus in all 5. Where nothing is drawn a candidate is not known to be wrong (outlines are drawn on some of a glomerulus'
planes, and not of every glomerulus). The tests matter more than the factorization: without them it gives 901 places, 30 on such
an outline, most of them the light round the lobe and between ROIs; and a factorization of each mask's own pixels in two did not
tell masks over two glomeruli from masks over one (parts for 5 of the 11, 2 of them right, and for 159 others), so masks are cut
by the per-trial test `--split` uses. It takes 14 to 22 s a fly on a GPU (RTX 3090), after the movies are read.
