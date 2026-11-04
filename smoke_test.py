"""Run run_pipeline.py on synthetic data with Cellpose replaced by a stand-in
(manual labels grown by 2 px, i.e. a model that overshoots borders), to check
everything around the model without GPU or downloaded weights.

Runs three times, into syn_results/:
  base         as is
  dropped      G03 and G11 unlabeled in the training flies (still labeled in the
               held-out flies, to score them)
  dropped_ref  the same plus the synthetic reference atlas (syn/reference), the only
               source of G03 and G11 then
"""
import csv
import sys
from pathlib import Path

import numpy as np
from skimage.segmentation import expand_labels

import run_pipeline
from alseg import segment

VAL = ["fly07", "fly08"]
DROP = ["G03", "G11"]
_state = {"drop": (), "val_labels": []}
_orig_load = run_pipeline.load_fly


def load(d, *args, **kw):
    f = _orig_load(d, *args, **kw)
    if f["name"] in VAL:
        _state["val_labels"].append(f["labels"])
    else:
        for lab, name in list(f["lut"].items()):
            if name in _state["drop"]:
                f["labels"][f["labels"] == lab] = 0
                del f["lut"][lab]
    return f


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
    sys.argv = ["run_pipeline.py", "syn", "--val", *VAL, "--out", out, "--cpu", *extra]
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


run_pipeline.load_fly = load
segment.train = fake_train
segment.predict = fake_predict
run("syn_results/base")
run("syn_results/dropped", DROP)
run("syn_results/dropped_ref", DROP, ["--ref", "syn/reference"])
print(f"\n{' '.join(DROP)} named correctly in held-out flies:")
for out in ("syn_results/dropped", "syn_results/dropped_ref"):
    print(f"  {out}: {dropped_named(out)}")
