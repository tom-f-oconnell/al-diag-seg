"""End-to-end example: ImageJ ROIs + odor responses -> trained model -> named, refined ROIs.

Expected layout (one folder per fly):

    data/
      fly01/
        anatomy.tif     (Z, Y, X)       mean or baseline fluorescence
        responses.tif   (n_odors, Z, Y, X)  mean dF/F per odor, same grid
        RoiSet.zip      ImageJ ROIs, one per glomerulus per z-slice
      fly02/ ...

If you have raw movies instead of responses.tif, compute it with
alseg.responses.response_vectors_from_movie(movie, onsets).

Steps: ROIs -> labels; response channels; Cellpose fine-tune; then for each
held-out fly: predict, refine by tuning alone, and (unless --no-atlas)
refine with an anatomical prior from a probabilistic atlas of the training
flies, and name ROIs using tuning + position + shape + layout + atlas.

Usage:
    python run_pipeline.py data/ --val fly07 fly13 --out results/ --epochs 200
    python run_pipeline.py data/ --val fly07 --skip-train --model results/models/al_glomeruli
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

from alseg import atlas, identity, metrics, refine, responses, roi_io, segment


def load_fly(d: Path, name_table=None):
    anat = tifffile.imread(d / "anatomy.tif").astype(np.float32)
    resp = tifffile.imread(d / "responses.tif").astype(np.float32)
    assert resp.shape[1:] == anat.shape, f"{d}: responses {resp.shape} vs anatomy {anat.shape}"
    labels, lut, overlap = roi_io.load_roiset(d / "RoiSet.zip", anat.shape, name_table)
    return {"name": d.name, "anatomy": anat, "responses": resp, "labels": labels,
            "lut": lut, "overlap": overlap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--val", nargs="+", required=True, help="fly folder names held out")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--erode", type=int, default=1, help="px to erode training ROIs")
    ap.add_argument("--spacing", type=float, nargs=3, default=(1, 1, 1), metavar=("Z", "Y", "X"),
                    help="voxel size in um, used for identity matching")
    ap.add_argument("--model", type=Path, help="existing model (with --skip-train)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-atlas", action="store_true", help="skip the anatomical prior")
    ap.add_argument("--reg", default="SyN", help="ANTs transform: Affine or SyN")
    ap.add_argument("--w-prior", type=float, default=0.5, help="weight of the atlas prior in refinement")
    a = ap.parse_args()
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
