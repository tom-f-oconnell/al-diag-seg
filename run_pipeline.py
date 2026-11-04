#!/usr/bin/env python3
"""End-to-end example: ImageJ ROIs + odor responses -> trained model -> named, refined ROIs.

Expected layout (one folder per fly; folders without RoiSet.zip are skipped):

    data/
      fly01/
        anatomy.tif     (Z, Y, X)           mean or baseline fluorescence
        responses.tif   (n_odors, Z, Y, X)  mean dF/F per odor, same grid
        RoiSet.zip      ImageJ ROIs, one per glomerulus per z-slice
      fly02/ ...

If you have raw movies instead of responses.tif, compute it with
alseg.responses.response_vectors_from_movie(movie, onsets).

Steps: ROIs -> labels; response channels; Cellpose fine-tune; then for each
held-out fly: predict, refine by tuning alone, and (unless --no-atlas) refine
with an anatomical prior from a probabilistic atlas of the training flies, and
name ROIs using tuning + position + shape + layout + atlas.

Writes to --out, per held-out fly: <fly>_pred.tif, <fly>_refined_tuning.tif,
<fly>_refined_prior.tif, <fly>_atlas_maxprob.tif and
<fly>_identities_<refinement>.csv; and summary.csv (one row per fly and ROI
set), name_table.json, and the trained model under models/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

from alseg import atlas, identity, metrics, reference, refine, responses, roi_io, segment


def load_fly(d: Path, name_table=None):
    anat = tifffile.imread(d / "anatomy.tif").astype(np.float32)
    resp = tifffile.imread(d / "responses.tif").astype(np.float32)
    assert resp.shape[1:] == anat.shape, f"{d}: responses {resp.shape} vs anatomy {anat.shape}"
    labels, lut, overlap = roi_io.load_roiset(d / "RoiSet.zip", anat.shape, name_table)
    return {"name": d.name, "anatomy": anat, "responses": resp, "labels": labels,
            "lut": lut, "overlap": overlap}


EXAMPLES = """\
examples:
  # train on all flies but three, and evaluate on those three
  python run_pipeline.py data/ --val fly07 fly13 fly21 --spacing 2 0.5 0.5

  # reuse a trained model instead of training again
  python run_pipeline.py data/ --val fly07 --skip-train --model results/models/al_glomeruli

  # without a GPU and with a faster (affine-only) atlas, or without the atlas
  python run_pipeline.py data/ --val fly07 --cpu --reg Affine
  python run_pipeline.py data/ --val fly07 --no-atlas

  # also take glomeruli none of the training flies have labels for from the
  # Grabe et al. 2016 atlas (not preferred, see --ref)
  python run_pipeline.py data/ --val fly07 --spacing 2 0.5 0.5 --ref ~/src/al_atlas/grabe2016
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
    ap.add_argument("--out", type=Path, default=Path("results"), metavar="DIR", help="output folder, created if needed")
    ap.add_argument("--spacing", type=float, nargs=3, default=(1, 1, 1), metavar=("Z", "Y", "X"),
                    help="voxel size in um, the same for every fly. Used to register flies for the "
                         "atlas, for positions and shapes in identity matching, and to align --ref "
                         "atlases.")

    g = ap.add_argument_group("segmentation model")
    g.add_argument("--epochs", type=int, default=200, metavar="N", help="Cellpose fine-tuning epochs")
    g.add_argument("--erode", type=int, default=1, metavar="PX",
                   help="pixels to erode the training ROIs by in-plane, so the model learns ROI "
                        "cores with gaps between neighboring glomeruli")
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
                   help="ANTs transform for registering the training flies into one atlas and the "
                        "atlas into each held-out fly: Affine (pose only, faster) or SyN (affine "
                        "then deformable, for glomerulus layouts that differ between flies), or "
                        "another ANTsPy type_of_transform")
    g.add_argument("--w-prior", type=float, default=0.5, metavar="W",
                   help="weight of the atlas prior relative to odor tuning in refinement. Higher "
                        "lets the atlas overrule tuning more; on synthetic data >= 1 shrinks or "
                        "drops ROIs.")

    g = ap.add_argument_group(
        "fixed reference atlases (not preferred)",
        "Priors for glomeruli your training flies have no labels for, from another source.\n"
        "Labeling those glomeruli in more of your own flies is better: that gives priors\n"
        "from your own preparation, and tuning templates, neither of which a reference has.")
    g.add_argument("--ref", type=Path, nargs="+", default=[], metavar="FOLDER",
                   help="reference atlas folders to add to the training-fly atlas, each with "
                        "labels.tif (voxel size in its ImageJ metadata) and one .csv with 'label' and "
                        "'glomerulus' columns (and optionally 'side'), as exported by "
                        "~/src/al_atlas. Not preferred, and in practice only "
                        "~/src/al_atlas/grabe2016: a reference is one other fly, aligned only "
                        "through the glomeruli named in both (needs >= 6), so its priors are "
                        "rougher than your own flies' (Grabe aligned to hemibrain: 5 um median "
                        "centroid error, median Dice 0.63), and several hemibrain meshes in "
                        "~/src/al_atlas/hemibrain are fragments. Glomeruli only in the reference "
                        "get named from position, layout and atlas overlap alone. Needs a correct "
                        "--spacing, and names that match your ROI names after the same cleanup "
                        "(e.g. DA4l -> DA4L). Default: no reference.")
    g.add_argument("--ref-side", metavar="SIDE",
                   help="which AL of a two-sided reference to use, as in its .csv's 'side' column "
                        "(left or right for grabe2016). Default: whichever fits the atlas better, "
                        "mirrored if needed (both are printed).")
    g.add_argument("--ref-reg", default="SyN", choices=["Affine", "SyN"],
                   help="how to align a reference to the atlas: Affine (fit to centroids of the "
                        "shared glomeruli only) or SyN (then deformable registration between the "
                        "shared glomeruli's label maps)")
    g.add_argument("--ref-weight", type=float, default=1.0, metavar="N",
                   help="how many training flies a reference counts as when averaging maps of "
                        "glomeruli in both; reference-only glomeruli are used as they are")
    a = ap.parse_args()
    if a.ref and a.no_atlas:
        ap.error("--ref needs the atlas (drop --no-atlas)")
    if a.skip_train and a.model is None:
        ap.error("--skip-train needs --model")
    a.out.mkdir(parents=True, exist_ok=True)

    dirs = sorted(p for p in a.data.iterdir() if (p / "RoiSet.zip").exists())
    shapes = [tifffile.TiffFile(d / "anatomy.tif").series[0].shape for d in dirs]
    name_table = roi_io.build_name_table([d / "RoiSet.zip" for d in dirs], shapes)
    (a.out / "name_table.json").write_text(json.dumps(name_table, indent=1))
    flies = [load_fly(d, name_table) for d in dirs]
    train_f = [f for f in flies if f["name"] not in a.val]
    val_f = [f for f in flies if f["name"] in a.val]
    print(f"{len(train_f)} training flies, {len(val_f)} held out, {len(name_table)} glomerulus names")

    # Shared PCA basis on training flies only (inside the AL, i.e. labeled voxels)
    pca = responses.SharedPCA(3).fit([f["responses"] for f in train_f],
                                     masks=[f["labels"] > 0 for f in train_f])
    for f in flies:
        f["channels"] = responses.build_channels(f["anatomy"], f["responses"], pca)

    if a.skip_train:
        model_path = a.model
    else:
        model_path, tl, vl = segment.train(train_f, val_f, a.out, n_epochs=a.epochs,
                                           erode=a.erode, gpu=not a.cpu)
        print("saved model", model_path)

    template = identity.build_template(
        [dict(f, spacing=a.spacing) for f in train_f])
    atl = None
    if not a.no_atlas:
        print("building atlas ...")
        atl = atlas.build_atlas(train_f, a.spacing, transform=a.reg)
        for r in a.ref:
            atl, info = reference.add_best_side(atl, r, a.ref_side, transform=a.ref_reg,
                                                weight=a.ref_weight)
            print(f"reference {r} ({info['side'] or 'one side'}{', mirrored' if info['mirrored'] else ''}): "
                  f"{info['n_shared']} glomeruli shared, centroid residual median "
                  f"{info['residual_median_um']:.1f} um, max {info['residual_max_um']:.1f} um"
                  + (f", left out of fit: {' '.join(info['rejected'])}" if info["rejected"] else "")
                  + f"; added {len(info['added'])}: {' '.join(info['added'])}")

    rows, id_rows = [], []
    for f in val_f:
        resp = f["responses"]
        pred = segment.predict(model_path, f["channels"], gpu=not a.cpu)
        variants = {"manual": f["labels"], "pred": pred,
                    "refined_tuning": refine.refine_by_tuning(resp, pred)}
        ids_plain = identity.assign_identities(resp, variants["refined_tuning"], template, a.spacing,
                                               w_shape=0, w_layout=0)
        ids_by = {"refined_tuning": ids_plain}

        if atl is not None:
            prior = atlas.atlas_to_fly(atl, f["anatomy"], transform=a.reg)
            # name the raw prediction first so the prior knows which glomerulus each ROI is
            ids0 = identity.assign_identities(resp, pred, template, a.spacing, prior, atl.names)
            names0 = {l: v[0] for l, v in ids0.items()}
            ref = refine.refine_with_prior(resp, pred, names0, prior, atl.names, w_prior=a.w_prior)
            variants["refined_prior"] = ref
            ids_by["refined_prior"] = identity.assign_identities(resp, ref, template, a.spacing,
                                                                 prior, atl.names)
            tifffile.imwrite(a.out / f"{f['name']}_atlas_maxprob.tif", prior.max(0).astype(np.float32))

        for tag, lab in variants.items():
            if tag != "manual":
                tifffile.imwrite(a.out / f"{f['name']}_{tag}.tif", lab.astype(np.uint16))
            ct = refine.crosstalk_report(resp, lab)
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
