"""Run run_pipeline.py on synthetic data with Cellpose replaced by a stand-in
(manual labels grown by 2 px, i.e. a model that overshoots borders), to check
everything around the model without GPU or downloaded weights."""
import sys

import numpy as np
from skimage.segmentation import expand_labels

import run_pipeline
from alseg import segment

VAL = ["fly07", "fly08"]
_val_labels = []
_orig_load = run_pipeline.load_fly


def load(d, nt=None):
    f = _orig_load(d, nt)
    if f["name"] in VAL:
        _val_labels.append(f["labels"])
    return f


def fake_train(train_f, val_f, out, **kw):
    X, Y = segment.make_training_slices(train_f, erode=kw.get("erode", 1))
    print(f"[fake train] {len(X)} slices, image {X[0].shape}, max objects/slice {max(m.max() for m in Y)}")
    return "fake", [], []


def fake_predict(model, channels, **kw):
    lab = _val_labels.pop(0)  # predictions are requested in VAL order
    return np.stack([expand_labels(s, 2) for s in lab])


run_pipeline.load_fly = load
segment.train = fake_train
segment.predict = fake_predict
sys.argv = ["run_pipeline.py", "syn", "--val", *VAL, "--out", "syn_results", "--cpu"]
run_pipeline.main()
