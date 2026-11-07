"""The commands: alseg-fit (fit a model to a database export and score it on held-out flies)
and alseg-predict (run a model on flies). ``run_pipeline.py`` is alseg-fit."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import atlas, olfdb, roi_io
from .fly import load_fly
from .model import Model, Options, fit, write_volume

FIT_DOC = """End-to-end example: ImageJ ROIs + odor responses -> trained model -> named, refined ROIs.

Input: an export from the odor response database (hong-data, its AL Segmentation tab),
unzipped. It has every file where it is on tensor, as al_analysis leaves it, and
roi_training.json, saying which is what; per fly:

    <its folder in the motion-corrected data>/
      <diagnostic recording>/trialmean_dff.tif            (n_odors, Z, Y, X) mean dF/F per odor
      <diagnostic recording>/trial_frames_and_odors.json  each trial's odor
      <one of those>/RoiSet.zip                           ImageJ ROIs, one per glomerulus per plane
      mocorr_concat_avg.tif                               (Z, Y, X) average image: mean of its motion-corrected movie
    <its folder in the raw data>/
      <diagnostic recording>/Experiment.xml               ThorImage metadata: voxel sizes
      anat/                                               optional: its anatomical stack, as recorded

A fly's responses are its diagnostic recordings' one after another, put in one order
for every fly by odor (named alike in every fly by the database), keeping the odors
every fly has. Only glomeruli are trained on, as the database tells them apart: ROIs
named as one (or several joined by /, that could not be told apart), and tentative ones
(DA1_t0) as that glomerulus, unless the fly also has an ROI of it without the mark; not
numbered ROIs, nor names ending in + or ? (see --no-tentative).
Voxel sizes come from the Experiment.xml files. Flies can have different voxel sizes,
and planes can be much further apart than pixels: geometry is in um, the model sees
every fly at one in-plane pixel size (--xy-um), and registration is in physical space,
through each fly's anatomical stack if it has one (see design.md). Stacks are as
recorded, but al_analysis mirrors right ALs' functional data left-right, so each stack
is placed both ways, and kept the way that fits better.

Flies are named <top folder of their data on tensor>_<date>_<fly number>, e.g.
Tom_2023-11-19_1 (as --val takes them).

Steps: ROIs -> labels; response channels; Cellpose fine-tune (on every ROI outlining one
glomerulus, named or not); then for each held-out fly: predict, refine by tuning alone, and
(unless --no-atlas) refine with an atlas prior from a probabilistic atlas of the training
flies, and name ROIs using tuning + position + shape + layout + atlas. The ROIs to use are
Cellpose's own, named (--final refined: the refined ones).

Writes to --out: the model, model/ (everything a new fly is segmented and named with:
run it on your own flies with alseg-predict); per held-out fly, each kind of ROIs
(pred: Cellpose's; refined_tuning; refined_prior) as labels, <fly>_<kind>.tif, and as an
ImageJ RoiSet named by glomerulus, <fly>_<kind>_RoiSet.zip; <fly>_atlas_maxprob.tif;
<fly>_identities_<kind>.csv; and summary.csv (one row per fly and kind of ROIs: matched to
the manual ROIs as 3D objects, and plane by plane, *_2d),
glomeruli.csv (per glomerulus: in how many held-out flies it was found and named right),
name_table.json, and Cellpose's own files under models/.
"""


def model_xy_um(model_path) -> float | None:
    p = Path(f"{model_path}.json")
    return json.loads(p.read_text())["xy_um"] if p.exists() else None


EXAMPLES = """\
examples:
  # train on all flies of an export but three, and evaluate on those three
  alseg-fit olfdb_roi_training_2026-11-04/ --val Tom_2023-11-19_1 Sam_2026-03-31_1 Sam_2026-03-31_2

  # reuse a trained model instead of training again
  alseg-fit data/ --val Tom_2023-11-19_1 --skip-train --model results/models/al_glomeruli

  # without a GPU and with a faster (affine-only) atlas, or without the atlas
  alseg-fit data/ --val Tom_2023-11-19_1 --cpu --reg Affine
  alseg-fit data/ --val Tom_2023-11-19_1 --no-atlas

  # also take glomeruli none of the training flies have labels for from the
  # Grabe et al. 2016 atlas (see --ref)
  alseg-fit data/ --val Tom_2023-11-19_1 --ref ~/src/al_atlas/grabe2016
"""


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keeps the layout of the description and examples, and shows defaults that are set."""

    def _get_help_string(self, action):
        if action.default is False:
            return action.help + " (default: off)"
        if action.default is None or action.default == []:
            return action.help
        return super()._get_help_string(action)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=FIT_DOC, epilog=EXAMPLES, formatter_class=_HelpFormatter)
    ap.add_argument("data", type=Path,
                    help="an export from the odor response database's AL Segmentation tab, unzipped: "
                         "the folder with its roi_training.json")
    ap.add_argument("--val", nargs="+", required=True, metavar="FLY",
                    help="names of the flies to hold out (e.g. Tom_2023-11-19_1). They are left out of training, the "
                         "identity template and the atlas, and scored against their manual ROIs. "
                         "Cellpose also uses them as its validation set while training. Required.")
    ap.add_argument("--out", type=Path, default=Path("results"), metavar="DIR",
                    help="output folder, created if needed")
    ap.add_argument("--no-tentative", action="store_true",
                    help="leave out tentative ROIs (named as a glomerulus but for a _t<n> suffix, "
                         "e.g. DA1_t0), which are otherwise trained on as that glomerulus where the "
                         "fly has no ROI of it without the mark")

    g = ap.add_argument_group("segmentation model")
    g.add_argument("--epochs", type=int, default=200, metavar="N", help="Cellpose fine-tuning epochs")
    g.add_argument("--batch-size", type=int, default=8, metavar="N",
                   help="Cellpose's training batches to start with (slices each), halved while one does not fit in "
                        "the GPU's memory (Cellpose-SAM: 4 at most on 24 GiB)")
    g.add_argument("--xy-um", type=float, metavar="UM",
                   help="in-plane pixel size (um) the model works at. Every fly's planes are "
                        "resampled in-plane to it, and predictions back; planes are never "
                        "resampled in z. Default: the size saved with --model for --skip-train, "
                        "otherwise the median of the flies' pixel sizes.")
    g.add_argument("--erode", type=int, default=1, metavar="PX",
                   help="pixels (at --xy-um) to erode the training ROIs by in-plane, so the model "
                        "learns ROI cores with gaps between neighboring glomeruli")
    g.add_argument("--skip-train", action="store_true", help="don't train; predict with --model")
    g.add_argument("--model", type=Path, metavar="PATH",
                   help="for --skip-train: a model folder (<out>/model of an earlier run: nothing is "
                        "fit again, the held-out flies are only scored), or trained Cellpose weights, "
                        "e.g. results/models/al_glomeruli (the rest is fit again)")
    g.add_argument("--cpu", action="store_true",
                   help="run Cellpose on the CPU instead of a GPU (training gets very slow)")
    g.add_argument("--segmenter", choices=["cellpose", "stand-in"], default="cellpose",
                   help="stand-in: for testing without a GPU or Cellpose's weights; trains nothing, and "
                        "predicts each fly's own ROIs grown by 2 pixels")

    g = ap.add_argument_group("atlas prior and refinement")
    g.add_argument("--no-atlas", action="store_true",
                   help="skip the atlas: no *_refined_prior.tif or *_atlas_maxprob.tif, and "
                        "identities from tuning, position and shape only")
    g.add_argument("--reg", default="SyN", metavar="TRANSFORM",
                   help="ANTs transform for registering flies to each other (training flies into "
                        "the atlas, the atlas into each held-out fly): Affine (pose only, faster) "
                        "or SyN (affine then deformable, for glomerulus layouts that differ between "
                        "flies), or another ANTsPy type_of_transform")
    g.add_argument("--hires-reg", default="Rigid", choices=["Rigid", "Similarity", "Affine"],
                   help="ANTs transform placing a fly's functional planes in its own anatomical stack")
    g.add_argument("--atlas-um", type=float, default=1.0, metavar="UM",
                   help="finest voxel size of the atlas grid, which is one training fly's "
                        "registration image (its anatomical stack if it has one, the most finely "
                        "sampled in z) coarsened to this")
    g.add_argument("--final", choices=["pred", "refined"], default="pred",
                   help="the ROIs to use (written as <fly>_RoiSet.zip by alseg-predict, and scored as the "
                        "final ones): Cellpose's own, named (pred); or refined by odor tuning and the atlas "
                        "(refined: on real flies, this shrank and broke up Cellpose's ROIs)")
    g.add_argument("--w-prior", type=float, default=0.5, metavar="W",
                   help="weight of the atlas prior relative to odor tuning in refinement. Higher "
                        "lets the atlas overrule tuning more; on synthetic data >= 1 shrinks or "
                        "drops ROIs.")
    g.add_argument("--deterministic", action="store_true",
                   help="make the ANTs registrations (atlas, anatomical stacks, --ref) repeatable "
                        "from run to run, with a fixed seed and one thread, which makes registration "
                        "about 4-5x slower. "
                        "Otherwise they use all cores and differ slightly between runs, which can "
                        "change close calls (e.g. whether a stack gets mirrored, or which glomeruli "
                        "a --ref fit leaves out). Doesn't affect Cellpose.")

    g = ap.add_argument_group(
        "fixed reference atlases",
        "Priors (and names) for glomeruli your training flies have no labels for, from another\n"
        "source.")
    g.add_argument("--ref", type=Path, nargs="+", default=[], metavar="FOLDER",
                   help="reference atlas folders to add to the training-fly atlas, each with "
                        "labels.tif (voxel size in its ImageJ metadata) and one .csv with 'label' and "
                        "'glomerulus' columns (and optionally 'side'), as exported by "
                        "~/src/al_atlas. Of those, grabe2016 is the better one: several "
                        "hemibrain meshes in ~/src/al_atlas/hemibrain are fragments. A reference "
                        "is aligned through the glomeruli named in both it and the atlas (needs "
                        ">= 6; Grabe aligned to hemibrain: 5 um median centroid error, median "
                        "Dice 0.63 for shared glomeruli). Glomeruli only in the reference "
                        "get named from position, layout and atlas overlap alone. Needs correct "
                        "voxel sizes, and names that match your ROI names but for upper and lower "
                        "case (DA4L is taken as DA4l). Default: no reference.")
    g.add_argument("--ref-side", metavar="SIDE",
                   help="which AL of a two-sided reference to use, as in its .csv's 'side' column "
                        "(left or right for grabe2016). Default: whichever fits the atlas better, "
                        "mirrored if needed (both are printed).")
    g.add_argument("--ref-reg", default="SyN", choices=["Affine", "SyN"],
                   help="how to align a reference to the atlas: Affine (a linear fit to centroids "
                        "of the shared glomeruli only: full affine or similarity, whichever "
                        "predicts left-out glomeruli better) or SyN (then deformable registration "
                        "between the shared glomeruli's label maps)")
    g.add_argument("--ref-weight", type=float, default=1.0, metavar="N",
                   help="how many training flies a reference counts as when averaging maps of "
                        "glomeruli in both; reference-only glomeruli are used as they are")
    return ap


def fit_main(argv=None):
    ap = _parser()
    a = ap.parse_args(argv)
    if a.ref and a.no_atlas:
        ap.error("--ref needs the atlas (drop --no-atlas)")
    if a.skip_train and a.model is None:
        ap.error("--skip-train needs --model")
    if a.deterministic:
        atlas.set_deterministic()
    if not olfdb.is_export(a.data):
        ap.error(f"{a.data} has no {olfdb.MANIFEST}: give the folder of an unzipped export from the "
                 "database's AL Segmentation tab")
    names = [e["name"] for e in olfdb.fly_entries(a.data)]
    unknown = [v for v in a.val if v not in names]
    if unknown:
        ap.error(f"--val: no fly {' '.join(unknown)} in the export (its flies: {' '.join(names)})")
    a.out.mkdir(parents=True, exist_ok=True)
    data = olfdb.load_export(a.data, tentative=not a.no_tentative)
    flies = data["flies"]
    (a.out / "name_table.json").write_text(json.dumps(data["name_table"], indent=1))
    val_f = [f for f in flies if f["name"] in a.val]
    print(f"{len(flies) - len(val_f)} training flies, {len(val_f)} held out, {len(data['name_table'])} glomerulus names, "
          f"{len(data['odors'])} odors in every fly")
    for f in flies:
        h = f.get("hires")
        print(f"  {f['name']}: voxels (z, y, x) {np.round(f['spacing'], 3).tolist()} um"
              + (f", anatomical stack {np.round(h['spacing'], 3).tolist()} um" if h else ""))

    if a.skip_train and (a.model / "model.json").is_file():
        model = Model.load(a.model)
        print("loaded model", a.model)
    else:
        o = Options(epochs=a.epochs, batch_size=a.batch_size, erode=a.erode, xy_um=a.xy_um or (model_xy_um(a.model) if a.skip_train else None),
                    atlas=not a.no_atlas, reg=a.reg, hires_reg=a.hires_reg, atlas_um=a.atlas_um, w_prior=a.w_prior,
                    refs=[str(r) for r in a.ref], ref_side=a.ref_side, ref_reg=a.ref_reg, ref_weight=a.ref_weight,
                    tentative=not a.no_tentative, gpu=False if a.cpu else None, segmenter=a.segmenter, final=a.final)
        model = fit(flies, a.val, o, out=a.out, odors=data["odors"], aliases=data["aliases"], glomeruli=data["glomeruli"],
                    weights=a.model if a.skip_train else None)
        model.save(a.out / "model")
        print("saved model", a.out / "model")
    res = model.evaluate(val_f, a.out)
    for r in res["identity"]:
        print(f"{r['fly']} identity of {r['rois']}: {r['correct']}/{r['total']}")
    for r in res["summary"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    return res


PREDICT_DOC = """Run a model (the model/ folder alseg-fit writes, or a fit's download from the database)
on flies: segment them, refine and name the ROIs.

A fly is given by its motion-corrected data folder, as al_analysis leaves it (<...>/<date>/<fly>/:
its diagnostic recordings' trialmean_dff.tif and trial_frames_and_odors.json, and
mocorr_concat_avg.tif or mocorr_concat.tif), with its raw data folder (--raw: the
Experiment.xml of its recordings, for its voxel size, and its anatomical stack, anat/, if it
has one); or as a fly of a database export (--export). Its odors are matched to the model's
by their names as recorded; it must have all of them.

Writes to --out, per fly: <fly>_RoiSet.zip, its ROIs named by glomerulus (numbered where none
fits), for ImageJ and al_analysis; each kind of ROIs as labels, <fly>_<kind>.tif (pred:
Cellpose's; refined_tuning; refined_prior, with the atlas) and as a RoiSet,
<fly>_<kind>_RoiSet.zip; <fly>_identities.csv. Flies with ROIs of their own (RoiSet.zip)
are also scored against them (summary.csv, glomeruli.csv, as alseg-fit's).
"""

PREDICT_EXAMPLES = """\
examples:
  # a fly of your own, as al_analysis left it on tensor
  alseg-predict model/ --fly /local/tensor/Ann/analysis/2026-03-31/1 --raw /local/tensor/Ann/raw/2026-03-31/1 --out pred/

  # several, with the same model
  alseg-predict model/ --fly analysis/2026-03-31/1 analysis/2026-03-31/2 --raw raw/2026-03-31/1 raw/2026-03-31/2

  # flies of a database export (their own ROIs scored too)
  alseg-predict model/ --export olfdb_roi_training_2026-11-04/ --flies Tom_2023-11-19_1
"""


def predict_main(argv=None):
    ap = argparse.ArgumentParser(prog="alseg-predict", description=PREDICT_DOC, epilog=PREDICT_EXAMPLES,
                                 formatter_class=_HelpFormatter)
    ap.add_argument("model", type=Path, help="the model's folder (with model.json)")
    ap.add_argument("--fly", type=Path, nargs="+", default=[], metavar="DIR", help="flies' motion-corrected data folders")
    ap.add_argument("--raw", type=Path, nargs="+", default=[], metavar="DIR",
                    help="their raw data folders, in the same order (voxel sizes, anatomical stacks)")
    ap.add_argument("--spacing", type=float, nargs=3, metavar=("Z", "Y", "X"),
                    help="voxel size (um), for flies without --raw")
    ap.add_argument("--export", type=Path, metavar="DIR", help="a database export, instead of --fly")
    ap.add_argument("--flies", nargs="+", default=[], metavar="NAME", help="which flies of --export (default: all)")
    ap.add_argument("--out", type=Path, default=Path("predictions"), metavar="DIR", help="output folder, created if needed")
    ap.add_argument("--cpu", action="store_true", help="run Cellpose on the CPU (slower) even if there is a GPU")
    a = ap.parse_args(argv)
    if not (a.model / "model.json").is_file():
        ap.error(f"{a.model} is not a model's folder (no model.json)")
    if bool(a.fly) == bool(a.export):
        ap.error("give flies with --fly, or an export with --export")
    if a.raw and len(a.raw) != len(a.fly):
        ap.error(f"--raw: {len(a.raw)} folders for {len(a.fly)} flies")
    model = Model.load(a.model)
    if a.cpu:
        model.options.gpu = False
    if a.export:
        data = olfdb.load_export(a.export, tentative=model.options.tentative)
        flies = [f for f in data["flies"] if not a.flies or f["name"] in a.flies]
        for f in flies:
            idx = {o: i for i, o in enumerate(f["odors"])}
            missing = [o for o in model.odors if o not in idx]
            if missing:
                ap.error(f"{f['name']}: the export's flies lack {len(missing)} of the model's odors: {', '.join(missing)}")
            f["responses"], f["odors"] = f["responses"][[idx[o] for o in model.odors]], list(model.odors)
    else:
        flies = [load_fly(d, model, raw_dir=a.raw[i] if a.raw else None, spacing=a.spacing) for i, d in enumerate(a.fly)]
    a.out.mkdir(parents=True, exist_ok=True)
    final = model.final_kind()
    scored = [f for f in flies if f.get("labels") is not None]
    if scored:  # (writes their ROIs too)
        model.evaluate(scored, a.out)
    for f in flies:
        if f.get("labels") is None:
            p = model.predict(f)
            for kind, lab in p["labels"].items():
                write_volume(a.out / f"{f['name']}_{kind}.tif", lab.astype(np.uint16))
                roi_io.write_roiset(lab, {l: v[0] for l, v in p["identities"][kind].items()},
                                    a.out / f"{f['name']}_{kind}_RoiSet.zip")
            with open(a.out / f"{f['name']}_identities.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["label", "assigned", "cost", "confidence"])
                w.writerows([l, name or "", f"{cost:.3f}", f"{conf:.3f}"] for l, (name, cost, conf) in sorted(p["identities"][final].items()))
        shutil.copyfile(a.out / f"{f['name']}_{final}_RoiSet.zip", a.out / f"{f['name']}_RoiSet.zip")
        print(f"{f['name']}: {a.out / (f['name'] + '_RoiSet.zip')}")


if __name__ == "__main__":
    sys.exit(fit_main())
