"""Flies that lack some of a model's odors (`alseg.fly.load_fly`, `alseg.model.Model.predict`): named by a model fit with more
odors than they have exactly as by one fit, to the same flies, with the odors they have, and with Cellpose's images made from
the same odors (`Options.image_odors`), given the same images; makeshift images where they lack some of those."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from alseg import identity
from alseg import model as am
from alseg import responses as R
from alseg.fly import load_fly

SPACING = (10.0, 0.5, 0.5)
ODORS = [f"odor{i} @ -3" for i in range(7)]
GLOMS = [f"G{i + 1}" for i in range(5)]


def fly(seed, odors=ODORS, name=None):
    """A fly of 5 glomeruli (boxes on 3 planes, shifted from fly to fly), each tuned its own way to 7 odors (the same in
    every fly, and noise), with its responses to ``odors``."""
    rng = np.random.default_rng(seed)
    tuning = np.random.default_rng(99).uniform(0, 1, (5, len(ODORS))) + np.eye(5, len(ODORS))
    lab = np.zeros((3, 40, 48), np.int32)
    for i, (y, x) in enumerate([(4, 4), (4, 26), (22, 4), (22, 26), (13, 15)]):
        dy, dx = rng.integers(0, 3, 2)
        lab[:, y + dy:y + dy + 12, x + dx:x + dx + 14] = i + 1
    resp = rng.normal(0, 0.05, (len(ODORS),) + lab.shape).astype(np.float32)
    for i in range(5):
        resp[:, lab == i + 1] += (tuning[i] * rng.uniform(0.8, 1.2))[:, None].astype(np.float32)
    rows = [ODORS.index(o) for o in odors]
    return {"name": name or f"fly{seed}", "labels": lab, "lut": {i + 1: g for i, g in enumerate(GLOMS)}, "spacing": SPACING,
            "average": (1 + (lab > 0) + rng.normal(0, 0.05, lab.shape)).astype(np.float32), "odors": list(odors), "responses": resp[rows]}


def fit(odors, image_odors=None):
    o = am.Options(segmenter="stand-in", atlas=False, gpu=False, image_odors=image_odors)
    return am.fit([fly(s, odors) for s in range(4)], options=o, odors=odors)


def test_template_for():
    flies = [fly(s) for s in range(4)]
    full = identity.build_template(flies)
    some = [0, 2, 3, 6]
    part = identity.build_template([{**f, "responses": f["responses"][some]} for f in flies])
    made = identity.template_for(full, some)
    for g in GLOMS:
        assert full[g]["tunings"].shape == (4, 7) and np.array_equal(made[g]["tuning"], part[g]["tuning"])
        assert np.array_equal(made[g]["tunings"], part[g]["tunings"]) and np.array_equal(made[g]["shape"], full[g]["shape"])
        # (not the full template's tuning at those odors: each fly's was z-scored over all of them)
        assert not np.allclose(made[g]["tuning"], full[g]["tuning"][some])
    # a template saved before each fly's tuning was kept: its mean tuning at those odors
    old = {g: {k: v for k, v in t.items() if k != "tunings"} for g, t in full.items()}
    assert np.array_equal(identity.template_for(old, some)["G1"]["tuning"], full["G1"]["tuning"][some])


def test_more_odors_as_fewer(tmp_path):
    """A model fit with 7 odors, its images from 5 of them, on a fly with those 5: as the model fit with those 5 alone."""
    few = ODORS[:5]
    a, b = fit(ODORS, image_odors=few), fit(few)
    assert a.odors == ODORS and a.image_odors() == few and b.odors == few == b.image_odors()
    assert np.array_equal(a.pca.mean_, b.pca.mean_) and np.array_equal(a.pca.components_, b.pca.components_)
    new = fly(10, few)
    assert a.fly_odors(new) == [0, 1, 2, 3, 4] and np.array_equal(a.channels(new), b.channels(new))
    pa, pb = a.predict(new), b.predict(new)
    assert all(np.array_equal(pa["labels"][k], pb["labels"][k]) for k in pb["labels"]) and set(pa["labels"]) == set(pb["labels"])
    assert pa["identities"] == pb["identities"] and {v[0] for v in pa["identities"]["pred"].values()} == set(GLOMS)
    # saved and loaded, the same (each training fly's tuning is kept with the model)
    a.save(tmp_path / "a")
    again = am.Model.load(tmp_path / "a")
    assert again.options.image_odors == few and again.predict(new)["identities"] == pb["identities"]
    assert np.array_equal(again.template["G1"]["tunings"], a.template["G1"]["tunings"])
    # with all 7, and with the 5 in another model's order of them: named as well
    assert {v[0] for v in a.predict(fly(10))["identities"]["pred"].values()} == set(GLOMS)
    # the images of a fly with all 7 are of the 5 too: nothing of the other odors is in them
    assert np.array_equal(a.channels(fly(10)), a.channels(new))
    # (images of two odors only: fewer components than a model keeps otherwise)
    two = fit(ODORS, image_odors=ODORS[:2])
    assert two.pca.components_.shape == (2, 2) and np.isfinite(two.channels(fly(10))).all()
    # a model fit with images of all its odors (as before there was a choice): those are its image odors
    assert fit(ODORS).image_odors() == ODORS and am.Options().image_odors is None
    with pytest.raises(ValueError, match="image_odors"):
        fit(ODORS, image_odors=["odor9 @ -3", ODORS[0]])


def test_lacking_image_odors():
    """A fly that lacks some of the odors the images are made from: the correlation of those it has, and the component with the
    others at the training flies' mean; joined and named by those it has."""
    a = fit(ODORS, image_odors=ODORS[:5])
    have = [ODORS[0], ODORS[2], ODORS[3], ODORS[6]]  # (three of the image odors, and one of the others)
    f = fly(10, have)
    ch = a.channels(f)
    r = f["responses"][:3]
    v = np.tile(a.pca.mean_, (r[0].size, 1))
    v[:, [0, 2, 3]] = R._zscore_vectors(r)
    pc = ((v - a.pca.mean_) @ a.pca.components_[0]).reshape(r.shape[1:])
    assert np.array_equal(ch[1], R.normalize01(R.local_correlation(r))) and np.array_equal(ch[2], R.normalize01(pc))
    assert np.array_equal(ch[0], R.normalize01(f["average"])) and np.isfinite(ch).all()
    p = a.predict(f)
    assert {v[0] for v in p["identities"]["pred"].values()} == set(GLOMS)
    assert np.array_equal(a.template_of(f)["G2"]["tunings"], a.template["G2"]["tunings"][:, [0, 2, 3, 6]])
    with pytest.raises(ValueError, match="1 of the 5 odors the model's images are made from"):
        a.channels(fly(10, [ODORS[1], ODORS[5], ODORS[6]]))


def test_load_fly(tmp_path):
    rng = np.random.default_rng(0)
    rec = tmp_path / "diag1"
    rec.mkdir()
    had = ODORS[:4] + ["something else @ -2"]
    resp = rng.normal(0, 1, (len(had), 2, 6, 8)).astype(np.float32)
    tifffile.imwrite(rec / "trialmean_dff.tif", resp, imagej=True, metadata={"axes": "TZYX"})
    (rec / "trial_frames_and_odors.json").write_text(json.dumps([{"odors": o} for o in had for _ in range(2)]))
    tifffile.imwrite(tmp_path / "mocorr_concat_avg.tif", rng.normal(0, 1, (2, 6, 8)).astype(np.float32), imagej=True, metadata={"axes": "ZYX"})
    model = SimpleNamespace(odors=ODORS, aliases={}, atlas=None, info={}, glomeruli=GLOMS)
    f = load_fly(tmp_path, model, spacing=SPACING)
    assert f["odors"] == ODORS[:4] and f["missing"] == ODORS[4:] and np.array_equal(f["responses"], resp[:4])
    # under 3 of the model's odors: no tuning to go by
    tifffile.imwrite(rec / "trialmean_dff.tif", resp[:3], imagej=True, metadata={"axes": "TZYX"})
    (rec / "trial_frames_and_odors.json").write_text(json.dumps([{"odors": o} for o in (ODORS[0], ODORS[1], "something else @ -2")]))
    with pytest.raises(ValueError, match="have 2 of the model's 7 odors"):
        load_fly(tmp_path, model, spacing=SPACING)
