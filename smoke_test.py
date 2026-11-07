"""Run run_pipeline.py on synthetic data with Cellpose replaced by a stand-in
(manual labels grown by 2 px, i.e. a model that overshoots borders), to check
everything around the model without GPU or downloaded weights.

syn/ is an export as from the odor response database (make_synthetic.py). Besides the
pipeline's results, checks that the ROIs named are the glomeruli (no numbered ROI;
G16_t0 as G16, but not where the fly has a plain G16 too), that Cellpose trains on the
numbered ROI too (but not on G16_t0 beside a G16), that the odor the fourth fly
lacks is left out of every fly, and that the right ALs' anatomical stacks are mirrored.

Runs three times, into syn_results/:
  base         as is
  dropped      G03 and G11 unlabeled in the training flies (still labeled in the
               held-out flies, to score them)
  dropped_ref  the same plus the synthetic reference atlas (syn/reference), the only
               source of G03 and G11 then

Then fits a model with the stand-in segmenter and no atlas (syn_results/stand_in), and runs
it with alseg-predict on a held-out fly twice: as a fly of the export, and from its own
folders as al_analysis leaves them. Both have to give the same RoiSet.zip, of the
database's glomerulus names (and numbered ROIs).

Options after smoke_test.py are passed on to every run, e.g. --deterministic.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from skimage.segmentation import expand_labels

import make_synthetic
import run_pipeline
from alseg import atlas, olfdb, segment

VAL = ["Ann_2026-01-01_7", "Ann_2026-01-01_8"]
EXTRA = sys.argv[1:]
DROP = ["G03", "G11"]
_state = {"drop": (), "val_labels": [], "mirrored": {}, "odors": None, "rois": {}, "objects": {}}
_orig_load, _orig_orient, _orig_align = olfdb.load, atlas.orient_hires, olfdb.align_odors


def load(root, entry, *args, **kw):
    f = _orig_load(root, entry, *args, **kw)
    _state["rois"][f["name"]] = sorted(f["lut"].values())
    _state["objects"][f["name"]] = len(np.unique(f["seg_labels"])) - 1  # (Cellpose's targets)
    if f["name"] in VAL:
        _state["val_labels"].append(f["labels"])
    else:
        for lab, name in list(f["lut"].items()):
            if name in _state["drop"]:
                f["seg_labels"][f["labels"] == lab] = 0
                f["labels"][f["labels"] == lab] = 0
                del f["lut"][lab]
    return f


def orient(f, *args, **kw):
    mirrored, scores = _orig_orient(f, *args, **kw)
    _state["mirrored"][f["name"]] = mirrored
    return mirrored, scores


def align(flies):
    _state["odors"] = _orig_align(flies)
    return _state["odors"]


def fake_train(train_f, val_f, out, **kw):
    X, Y = segment.make_training_slices(train_f, erode=kw.get("erode", 1), xy_um=kw.get("xy_um"))
    print(f"[fake train] {len(X)} slices, images {sorted({x.shape for x in X})}, "
          f"max objects/slice {max(m.max() for m in Y)}")
    return "fake", [], []


def fake_predict(model, channels, **kw):
    lab = _state["val_labels"].pop(0)  # predictions are requested in VAL order
    return np.stack([expand_labels(s, 2) for s in lab])


def run(out, drop=(), extra=()):
    print(f"\n=== {out}")
    _state["drop"], _state["val_labels"] = drop, []
    sys.argv = ["run_pipeline.py", "syn", "--val", *VAL, "--out", out, "--cpu", *extra, *EXTRA]
    run_pipeline.main()


def dropped_named(out):
    """Correctly named / total held-out ROIs of the dropped glomeruli, per refinement."""
    res = {}
    for tag in ("refined_tuning", "refined_prior"):
        ok = n = 0
        for fly in VAL:
            with open(Path(out) / f"{fly}_identities_{tag}.csv", newline="") as fh:
                for r in csv.DictReader(fh):
                    if r["manual"] in DROP:
                        n += 1
                        ok += r["assigned"] == r["manual"]
        res[tag] = f"{ok}/{n}"
    return res


olfdb.load, atlas.orient_hires, olfdb.align_odors = load, orient, align
segment.train = fake_train
segment.predict = fake_predict
run("syn_results/base")
run("syn_results/dropped", DROP)
run("syn_results/dropped_ref", DROP, ["--ref", "syn/reference"])
names = json.loads(Path("syn_results/base/name_table.json").read_text())
assert sorted(names) == make_synthetic.NAMES, names
assert "G16" in _state["rois"]["Ann_2026-01-01_2"] and _state["rois"]["Ann_2026-01-01_5"].count("G16") == 1
assert sorted(_state["odors"]) == make_synthetic.ODORS[:-1], _state["odors"]
# Cellpose's targets: every glomerulus and the numbered ROI
assert all(_state["objects"][n] == len(r) + 1 for n, r in _state["rois"].items()), (_state["objects"], _state["rois"])
wrong = {n: m for n, m in _state["mirrored"].items() if m != (int(n.rsplit("_", 1)[1]) in make_synthetic.RIGHT_ALS)}
assert not wrong, f"anatomical stacks oriented wrongly: {wrong}"
print(f"\nROIs, odors and the orientation of {len(_state['mirrored'])} anatomical stacks as expected")
print(f"{' '.join(DROP)} named correctly in held-out flies:")
for out in ("syn_results/dropped", "syn_results/dropped_ref"):
    print(f"  {out}: {dropped_named(out)}")

# a saved model, loaded and run on a fly of the export, and on the fly's own folders
import roifile  # noqa: E402

from alseg import cli  # noqa: E402

_state["drop"] = ()
cli.fit_main(["syn", "--val", *VAL, "--out", "syn_results/stand_in", "--segmenter", "stand-in", "--no-atlas", "--cpu"])
fly = next(f for f in json.loads(Path("syn/roi_training.json").read_text())["flies"] if f"{f['date']}_{f['fly_num']}" in VAL[1])
cli.predict_main(["syn_results/stand_in/model", "--export", "syn", "--flies", VAL[1], "--out", "syn_results/predict_export"])
cli.predict_main(["syn_results/stand_in/model", "--fly", f"syn/{fly['dir']}", "--raw", f"syn/{fly['raw_dir']}",
                  "--out", "syn_results/predict_fly"])
a = Path(f"syn_results/predict_export/{VAL[1]}_RoiSet.zip")
b = Path(f"syn_results/predict_fly/2026-01-01_{fly['fly_num']}_RoiSet.zip")
assert a.read_bytes() == b.read_bytes(), "the export's fly and its own folders were segmented differently"
names = {r.name for r in roifile.ImagejRoi.fromfile(a)}
assert names <= set(make_synthetic.NAMES) | {n for n in names if n.isdigit()} and names & set(make_synthetic.NAMES), names
print(f"model saved, loaded and run on {VAL[1]} as a fly of the export and from its own folders: the same "
      f"RoiSet.zip ({len(names)} names)")
