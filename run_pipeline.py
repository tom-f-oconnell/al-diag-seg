#!/usr/bin/env python3
"""End-to-end example: ImageJ ROIs + odor responses -> trained model -> named, refined ROIs.

Expected layout (one folder per fly; folders without RoiSet.zip are skipped):

    data/
      fly01/
        anatomy.tif       (Z, Y, X)           mean or baseline fluorescence
        responses.tif     (n_odors, Z, Y, X)  mean dF/F per odor, same grid
        RoiSet.zip        ImageJ ROIs, one per glomerulus per z-slice
        Experiment.xml    optional: ThorImage metadata of that recording
        hires/            optional: higher resolution anatomical stack of the same fly
          anatomy.tif     (Z, Y, X)
          Experiment.xml  its ThorImage metadata (required if the fly has one above)
      fly02/ ...

If you have raw movies instead of responses.tif, compute it with
alseg.responses.response_vectors_from_movie(movie, onsets).

Voxel sizes come from the Experiment.xml files if there are any (then every fly needs
one, and --spacing and --hires-spacing are not allowed), otherwise from --spacing and
--hires-spacing. Flies can have different voxel sizes, and planes can be much further
apart than pixels: geometry is in um, the model sees every fly at one in-plane pixel
size (--xy-um), and registration is in physical space, through each fly's hires/ stack
if it has one (see design.md).

Steps: ROIs -> labels; response channels; Cellpose fine-tune; then for each
held-out fly: predict, refine by tuning alone, and (unless --no-atlas) refine
with an anatomical prior from a probabilistic atlas of the training flies, and
name ROIs using tuning + position + shape + layout + atlas.

Writes to --out, per held-out fly: <fly>_pred.tif, <fly>_refined_tuning.tif,
<fly>_refined_prior.tif, <fly>_atlas_maxprob.tif and
<fly>_identities_<refinement>.csv; and summary.csv (one row per fly and ROI
set), name_table.json, and the trained model under models/ (with its --xy-um in
<model>.json).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

from alseg import atlas, identity, metadata, metrics, reference, refine, responses, roi_io, segment


def load_fly(d: Path, name_table=None, spacing=None, hires_spacing=None):
    """One fly folder. Voxel sizes come from its Experiment.xml files if it has
    them, otherwise from ``spacing`` (default 1 um) and ``hires_spacing``."""
    anat = tifffile.imread(d / "anatomy.tif").astype(np.float32)
    resp = tifffile.imread(d / "responses.tif").astype(np.float32)
    assert resp.shape[1:] == anat.shape, f"{d}: responses {resp.shape} vs anatomy {anat.shape}"
    labels, lut, overlap = roi_io.load_roiset(d / "RoiSet.zip", anat.shape, name_table)
    xml = metadata.has_xml(d)
    f = {"name": d.name, "anatomy": anat, "responses": resp, "labels": labels, "lut": lut,
         "overlap": overlap,
         "spacing": (metadata.thorimage_spacing(d / "Experiment.xml", anat.shape) if xml
                     else tuple(float(x) for x in (spacing or (1, 1, 1))))}
    if (d / "hires" / "anatomy.tif").exists():
        h = tifffile.imread(d / "hires" / "anatomy.tif").astype(np.float32)
        hs = (metadata.thorimage_spacing(d / "hires" / "Experiment.xml", h.shape) if xml
              else tuple(float(x) for x in hires_spacing))
        f["hires"] = {"anatomy": h, "spacing": hs}
    return f


def check_voxel_size_sources(dirs, a, error):
    """Experiment.xml for every fly (and hires/ stack) or for none, and no
    --spacing/--hires-spacing with them."""
    with_xml = [d.name for d in dirs if metadata.has_xml(d)]
    hires = [d for d in dirs if (d / "hires" / "anatomy.tif").exists()]
    if with_xml:
        missing = [d.name for d in dirs if not metadata.has_xml(d)]
        if missing:
            error(f"Experiment.xml is missing for {' '.join(missing)}; when any fly has one, all must")
        if a.spacing is not None or a.hires_spacing is not None:
            error("--spacing and --hires-spacing can't be used with Experiment.xml files, "
                  "which give the voxel sizes")
        missing = [d.name for d in hires if not metadata.has_xml(d / "hires")]
        if missing:
            error(f"hires/Experiment.xml is missing for {' '.join(missing)}")
    else:
        if hires and a.hires_spacing is None:
            error(f"{' '.join(d.name for d in hires)} have hires/ stacks: give their voxel size "
                  "with --hires-spacing (or use Experiment.xml files)")
        if a.spacing is None:
            print("no Experiment.xml files or --spacing: assuming 1 um voxels")


def model_xy_um(model_path) -> float | None:
    p = Path(f"{model_path}.json")
    return json.loads(p.read_text())["xy_um"] if p.exists() else None


EXAMPLES = """\
examples:
  # train on all flies but three, and evaluate on those three. With Experiment.xml
  # files in the fly folders, voxel sizes come from those.
  python run_pipeline.py data/ --val fly07 fly13 fly21

  # without Experiment.xml files, give voxel sizes (z y x, in um), including for any
  # hires/ stacks
  python run_pipeline.py data/ --val fly07 --spacing 10 0.6 0.6 --hires-spacing 2 0.4 0.4

  # reuse a trained model instead of training again
  python run_pipeline.py data/ --val fly07 --skip-train --model results/models/al_glomeruli

  # without a GPU and with a faster (affine-only) atlas, or without the atlas
  python run_pipeline.py data/ --val fly07 --cpu --reg Affine
  python run_pipeline.py data/ --val fly07 --no-atlas

  # also take glomeruli none of the training flies have labels for from the
  # Grabe et al. 2016 atlas (see --ref)
  python run_pipeline.py data/ --val fly07 --ref ~/src/al_atlas/grabe2016
"""


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keeps the layout of the description and examples, and shows defaults that are set."""

    def _get_help_string(self, action):
        if action.default is False:
            return action.help + " (default: off)"
        if action.default is None or action.default == []:
            return action.help
        return super()._get_help_string(action)


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=EXAMPLES, formatter_class=_HelpFormatter)
    ap.add_argument("data", type=Path,
                    help="folder with one subfolder per fly, laid out as above")
    ap.add_argument("--val", nargs="+", required=True, metavar="FLY",
                    help="names of the fly folders to hold out. They are left out of training, the "
                         "identity template and the atlas, and scored against their manual ROIs. "
                         "Cellpose also uses them as its validation set while training. Required.")
    ap.add_argument("--out", type=Path, default=Path("results"), metavar="DIR",
                    help="output folder, created if needed")

    g = ap.add_argument_group(
        "voxel sizes",
        "Only without Experiment.xml files; with them, these are errors.")
    g.add_argument("--spacing", type=float, nargs=3, metavar=("Z", "Y", "X"),
                   help="voxel size in um of the functional recordings (anatomy.tif, "
                        "responses.tif), the same for every fly. Default: 1 1 1, with a warning.")
    g.add_argument("--hires-spacing", type=float, nargs=3, metavar=("Z", "Y", "X"),
                   help="voxel size in um of the hires/anatomy.tif stacks, the same for every fly. "
                        "Required if any fly has one.")

    g = ap.add_argument_group("segmentation model")
    g.add_argument("--epochs", type=int, default=200, metavar="N", help="Cellpose fine-tuning epochs")
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
                   help="trained Cellpose model for --skip-train, e.g. results/models/al_glomeruli "
                        "(training saves to <out>/models/al_glomeruli). Required with --skip-train.")
    g.add_argument("--cpu", action="store_true",
                   help="run Cellpose on the CPU instead of a GPU (training gets very slow)")

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
                   help="ANTs transform placing a fly's functional planes in its own hires/ stack")
    g.add_argument("--atlas-um", type=float, default=1.0, metavar="UM",
                   help="finest voxel size of the atlas grid, which is one training fly's "
                        "registration image (its hires/ stack if it has one, the most finely "
                        "sampled in z) coarsened to this")
    g.add_argument("--w-prior", type=float, default=0.5, metavar="W",
                   help="weight of the atlas prior relative to odor tuning in refinement. Higher "
                        "lets the atlas overrule tuning more; on synthetic data >= 1 shrinks or "
                        "drops ROIs.")

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
                        "voxel sizes, and names that match your ROI names after the same cleanup "
                        "(e.g. DA4l -> DA4L). Default: no reference.")
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
    a = ap.parse_args()
    if a.ref and a.no_atlas:
        ap.error("--ref needs the atlas (drop --no-atlas)")
    if a.skip_train and a.model is None:
        ap.error("--skip-train needs --model")

    dirs = sorted(p for p in a.data.iterdir() if (p / "RoiSet.zip").exists())
    check_voxel_size_sources(dirs, a, ap.error)
    a.out.mkdir(parents=True, exist_ok=True)
    shapes = [tifffile.TiffFile(d / "anatomy.tif").series[0].shape for d in dirs]
    name_table = roi_io.build_name_table([d / "RoiSet.zip" for d in dirs], shapes)
    (a.out / "name_table.json").write_text(json.dumps(name_table, indent=1))
    flies = [load_fly(d, name_table, a.spacing, a.hires_spacing) for d in dirs]
    train_f = [f for f in flies if f["name"] not in a.val]
    val_f = [f for f in flies if f["name"] in a.val]
    print(f"{len(train_f)} training flies, {len(val_f)} held out, {len(name_table)} glomerulus names")
    for f in flies:
        h = f.get("hires")
        print(f"  {f['name']}: voxels (z, y, x) {np.round(f['spacing'], 3).tolist()} um"
              + (f", hires/ {np.round(h['spacing'], 3).tolist()} um" if h else ""))

    # Shared PCA basis on training flies only (inside the AL, i.e. labeled voxels)
    pca = responses.SharedPCA(3).fit([f["responses"] for f in train_f],
                                     masks=[f["labels"] > 0 for f in train_f])
    for f in flies:
        f["channels"] = responses.build_channels(f["anatomy"], f["responses"], pca)

    xy_um = a.xy_um or (model_xy_um(a.model) if a.skip_train else None)
    if xy_um is None:
        xy_um = float(np.median([f["spacing"][2] for f in flies]))
    print(f"segmenting at {xy_um:.3f} um pixels")
    if a.skip_train:
        model_path = a.model
    else:
        model_path, tl, vl = segment.train(train_f, val_f, a.out, n_epochs=a.epochs,
                                           erode=a.erode, gpu=not a.cpu, xy_um=xy_um)
        if Path(model_path).exists():
            Path(f"{model_path}.json").write_text(json.dumps({"xy_um": xy_um}))
        print("saved model", model_path)

    template = identity.build_template(train_f)
    atl = None
    if not a.no_atlas:
        print("building atlas ...")
        atl = atlas.build_atlas(train_f, transform=a.reg, hires_transform=a.hires_reg,
                                atlas_um=a.atlas_um)
        print(f"  atlas grid (z, y, x) {atl.template.shape} at {np.round(atl.spacing, 3).tolist()} um")
        for r in a.ref:
            atl, info = reference.add_best_side(atl, r, a.ref_side, transform=a.ref_reg,
                                                weight=a.ref_weight)
            print(f"reference {r} ({info['side'] or 'one side'}{', mirrored' if info['mirrored'] else ''}, "
                  f"{info['fit']} fit): "
                  f"{info['n_shared']} glomeruli shared, centroid residual median "
                  f"{info['residual_median_um']:.1f} um, max {info['residual_max_um']:.1f} um"
                  + (f", left out of fit: {' '.join(info['rejected'])}" if info["rejected"] else "")
                  + f"; added {len(info['added'])}: {' '.join(info['added'])}")

    rows, id_rows = [], []
    for f in val_f:
        resp, sp = f["responses"], f["spacing"]
        pred = segment.predict(model_path, f["channels"], gpu=not a.cpu, spacing=sp, xy_um=xy_um)
        variants = {"manual": f["labels"], "pred": pred,
                    "refined_tuning": refine.refine_by_tuning(resp, pred, sp)}
        ids_plain = identity.assign_identities(resp, variants["refined_tuning"], template, sp,
                                               w_shape=0, w_layout=0)
        ids_by = {"refined_tuning": ids_plain}

        if atl is not None:
            prior = atlas.atlas_to_fly(atl, f, transform=a.reg, hires_transform=a.hires_reg)
            # name the raw prediction first so the prior knows which glomerulus each ROI is
            ids0 = identity.assign_identities(resp, pred, template, sp, prior, atl.names)
            names0 = {l: v[0] for l, v in ids0.items()}
            ref = refine.refine_with_prior(resp, pred, names0, prior, atl.names, sp, w_prior=a.w_prior)
            variants["refined_prior"] = ref
            ids_by["refined_prior"] = identity.assign_identities(resp, ref, template, sp,
                                                                 prior, atl.names)
            tifffile.imwrite(a.out / f"{f['name']}_atlas_maxprob.tif", prior.max(0).astype(np.float32))

        for tag, lab in variants.items():
            if tag != "manual":
                tifffile.imwrite(a.out / f"{f['name']}_{tag}.tif", lab.astype(np.uint16))
            ct = refine.crosstalk_report(resp, lab, sp)
            geo = metrics.match_report(f["labels"], lab) if tag != "manual" else {}
            rows.append({
                "fly": f["name"], "rois": tag,
                "n": len(np.unique(lab)) - 1,
                "recall": geo.get("recall", ""), "precision": geo.get("precision", ""),
                "dice": geo.get("mean_dice_matched", ""),
                "splits": geo.get("splits", ""), "merges": geo.get("merges", ""),
                "coherence": np.mean(list(ct["coherence"].values())) if ct["coherence"] else "",
                "neighbor_corr": np.mean(list(ct["neighbor_corr"].values())) if ct["neighbor_corr"] else "",
                "leak": np.mean(np.abs(list(ct["leak"].values()))) if ct["leak"] else "",
            })

        # identity accuracy: compare to the name of the best-overlapping manual ROI
        for tag, ids in ids_by.items():
            lab = variants[tag]
            iou, g_ids, p_ids = metrics.iou_matrix(f["labels"], lab)
            correct = total = 0
            with open(a.out / f"{f['name']}_identities_{tag}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["label", "assigned", "manual", "cost", "confidence"])
                for j, p in enumerate(p_ids):
                    manual = f["lut"].get(int(g_ids[iou[:, j].argmax()])) if iou[:, j].max() > 0.3 else None
                    name, cost, conf = ids.get(int(p), (None, np.nan, np.nan))
                    w.writerow([int(p), name, manual, f"{cost:.3f}", f"{conf:.3f}"])
                    if manual:
                        total += 1
                        correct += name == manual
            id_rows.append((f["name"], tag, correct, total))
            print(f"{f['name']} identity ({'tuning + position' if tag == 'refined_tuning' else 'with atlas, shape, layout'}): {correct}/{total}")

    with open(a.out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
