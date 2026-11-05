"""Run run_pipeline.py on synthetic data with Cellpose replaced by a stand-in
(manual labels grown by 2 px, i.e. a model that overshoots borders), to check
everything around the model without GPU or downloaded weights.

syn/ is an export as from the odor response database (make_synthetic.py). Besides the
pipeline's results, checks that the ROIs trained on are the glomeruli (no numbered ROI;
G16_t0 as G16, but not where the fly has a plain G16 too), that the odor the fourth fly
lacks is left out of every fly, and that the right ALs' anatomical stacks are mirrored.

Runs three times, into syn_results/:
  base         as is
  dropped      G03 and G11 unlabeled in the training flies (still labeled in the
               held-out flies, to score them)
  dropped_ref  the same plus the synthetic reference atlas (syn/reference), the only
               source of G03 and G11 then

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
_state = {"drop": (), "val_labels": [], "mirrored": {}, "odors": None, "rois": {}}
_orig_load, _orig_orient, _orig_align = olfdb.load, atlas.orient_hires, olfdb.align_odors


def load(root, entry, *args, **kw):
    f = _orig_load(root, entry, *args, **kw)
    _state["rois"][f["name"]] = sorted(f["lut"].values())
    if f["name"] in VAL:
        _state["val_labels"].append(f["labels"])
    else:
        for lab, name in list(f["lut"].items()):
            if name in _state["drop"]:
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
wrong = {n: m for n, m in _state["mirrored"].items() if m != (int(n.rsplit("_", 1)[1]) in make_synthetic.RIGHT_ALS)}
assert not wrong, f"anatomical stacks oriented wrongly: {wrong}"
print(f"\nROIs, odors and the orientation of {len(_state['mirrored'])} anatomical stacks as expected")
print(f"{' '.join(DROP)} named correctly in held-out flies:")
for out in ("syn_results/dropped", "syn_results/dropped_ref"):
    print(f"  {out}: {dropped_named(out)}")
