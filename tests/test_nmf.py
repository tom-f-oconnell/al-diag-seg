"""Looking for glomeruli ROIs miss (`alseg.nmf`), the movies read as its evidence
(`alseg.evidence.frames`) and names kept while others are given (`alseg.identity.assign_identities`'s
``fixed``), on small synthetic flies."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from scipy import ndimage as ndi

from alseg import evidence, identity, nmf

pytest.importorskip("torch")

SPACING = (10.0, 0.5, 0.5)
CENTERS = [(18, 18), (18, 46), (46, 18), (46, 46), (32, 32)]
ODORS, REPEATS, BASE, ON = 6, 3, 4, 6


def truth():
    """5 glomeruli (disks of 4 um) on each of 3 planes: four at the corners, the fifth among them."""
    yy, xx = np.mgrid[:64, :64]
    lab = np.zeros((3, 64, 64), np.int32)
    for i, (cy, cx) in enumerate(CENTERS):
        lab[:, (yy - cy) ** 2 + (xx - cx) ** 2 <= 8 ** 2] = i + 1
    return lab


def movies(lab, seed=0, signal=8.0):
    """Trials of 6 odors, each glomerulus answering one of its own most, its light blurred beyond its
    outline, with noise of 1: as `evidence.frames` gives them, and the tuning each glomerulus has."""
    rng = np.random.default_rng(seed)
    tuning = np.eye(5, ODORS) * signal + rng.uniform(0, 1, (5, ODORS)) * (signal > 0)
    foot = np.stack([ndi.gaussian_filter((lab == i + 1).astype(np.float32), (0, 1.5, 1.5)) for i in range(5)])
    dff, trial, on, trials = [], [], [], []
    for t in range(ODORS * REPEATS):
        amp = tuning[:, t % ODORS] * rng.uniform(0.8, 1.2)
        v = np.zeros((BASE + ON,) + lab.shape, np.float32)
        v[BASE:] = np.tensordot(amp, foot, 1)
        dff.append(v + rng.normal(0, 1, v.shape).astype(np.float32))
        trial += [t] * (BASE + ON)
        on += [False] * BASE + [True] * ON
        trials.append(("rec", f"odor{t % ODORS}"))
    return {"dff": np.concatenate(dff).astype(np.float16), "trial": np.array(trial), "on": np.array(on), "trials": trials,
            "recordings": ["rec"]}, tuning


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def missing(lab):
    """The ROIs a model would have left of ``lab``: none of glomerulus 5, and none of 4 on the middle plane."""
    out = lab.copy()
    out[out == 5] = 0
    out[1][out[1] == 4] = 0
    return out


def test_find():
    lab = truth()
    labels, (mv, _) = missing(lab), movies(lab)
    stages = []
    res = nmf.find(labels, {"spacing": SPACING}, mv, gpu=False, split=False, progress=lambda *a: stages.append(a))
    cands = {c["kind"]: c for c in res["candidates"]}
    assert sorted(c["kind"] for c in res["candidates"]) == ["added", "extended"], res["candidates"]
    assert res["planes"] == [0, 1, 2] and stages[-1] == ("factorize", 3, 3)
    # the glomerulus with no ROI: one new ROI over its three planes, numbered after the others
    a = cands["added"]
    assert a["planes"] == [0, 1, 2] and a["label"] == 5 and a["of"] is None and a["eta2"] > 0.8 and a["novelty"] > 0.8
    assert all(iou(res["labels"][z] == 5, lab[z] == 5) > 0.6 for z in range(3))
    # the one with no outline on the middle plane: more of its ROI there
    e = cands["extended"]
    assert e["planes"] == [1] and e["label"] == 4 and e["of"] == 4 and iou(res["labels"][1] == 4, lab[1] == 4) > 0.6
    # the ROIs themselves are as they were, and each voxel changed says which candidate it is of
    assert np.array_equal(res["labels"][labels > 0], labels[labels > 0]) and res["labels"].dtype == labels.dtype
    assert np.array_equal(res["candidate"] > 0, res["labels"] != labels)
    assert set(np.unique(res["candidate"][res["labels"] == 5])) == {a["id"]}
    # with some of them only: the others' voxels as in the ROIs they were found for
    only = nmf.subset(labels, res["labels"], res["candidate"], [e["id"]])
    assert (only == 5).sum() == 0 and np.array_equal(only == 4, res["labels"] == 4)
    assert np.array_equal(nmf.subset(labels, res["labels"], res["candidate"], []), labels)


def test_find_nothing():
    """Nothing is found where there is nothing: in noise alone, in ROIs that leave nothing out (the light
    round them is theirs), nor on a plane with too few ROIs to say where the lobe is."""
    lab = truth()
    mv, _ = movies(lab, signal=0.0)
    assert nmf.find(missing(lab), {"spacing": SPACING}, mv, gpu=False, split=False)["candidates"] == []
    mv, _ = movies(lab)
    assert nmf.find(lab, {"spacing": SPACING}, mv, gpu=False, split=False)["candidates"] == []
    # without the factorization: the ROIs' own masks by the same tests, and no candidate
    res = nmf.find(missing(lab), {"spacing": SPACING}, mv, gpu=False, split=False, n_free=0)
    assert res["candidates"] == [] and len(res["rois"]) == 4 * 3 - 1 and all(r["eta2"] > 0.8 and r["novelty"] > 0.8 and r["radius_um"] >= 3.5 for r in res["rois"])
    two = np.where(lab <= 2, lab, 0)
    res = nmf.find(two, {"spacing": SPACING}, mv, gpu=False, split=False)
    assert res["candidates"] == [] and res["planes"] == []
    with pytest.raises(ValueError, match="no parameter min_size"):
        nmf.find(lab, {"spacing": SPACING}, mv, gpu=False, split=False, min_size=3)


def responses(lab, tuning):
    """(odors, Z, Y, X): each glomerulus' tuning over its voxels."""
    out = np.zeros((ODORS,) + lab.shape, np.float32)
    for i in range(5):
        out[:, lab == i + 1] = tuning[i][:, None]
    return out


def test_check():
    """Candidates cleaned as the ROIs were, and named: with the ROIs' names kept, and all named again."""
    lab = truth()
    labels, (mv, tuning) = missing(lab), movies(lab)
    lut = {i + 1: f"G{i + 1}" for i in range(5)}
    template = identity.build_template([{"responses": responses(lab, tuning), "labels": lab, "lut": lut, "spacing": SPACING}])
    model = SimpleNamespace(options=SimpleNamespace(gpu=False, max_cost=0.6), template=template, atlas=None)
    fly = {"spacing": SPACING, "responses": responses(lab, tuning) + np.random.default_rng(1).normal(0, 0.2, (ODORS,) + lab.shape).astype(np.float32)}
    # (the ROIs' names as a model gave them: one of them wrong, one left without)
    names = {1: "G1", 2: "G5", 3: None, 4: "G4"}
    stages = []
    res = nmf.check(model, fly, labels, names, steps=["margin"], clean_params={"margin": {"gap_um": 1.0}}, movies=mv, split=False,
                    progress=lambda *a: stages.append(a))
    assert res["cleaned"] == ["margin"] and stages[-1] == ("name", 2, 2)
    added = next(c for c in res["candidates"] if c["kind"] == "added")
    # kept from the ROIs: 1 um from their outlines' pixels at least (the ROIs as they were)
    far = ndi.distance_transform_edt(labels[0] == 0, sampling=SPACING[1:])
    assert far[res["labels"][0] == added["label"]].min() >= 1.0 and np.array_equal(res["labels"][labels > 0], labels[labels > 0])
    kept, joint = res["names"]["kept"], res["names"]["joint"]
    assert {l: kept[l] for l in names} == names and kept[added["label"]] not in ("G1", "G5", "G4")
    assert joint == {1: "G1", 2: "G2", 3: "G3", 4: "G4", added["label"]: "G5"}
    # some of them, named again together (as a caller with the look's volumes does)
    sub = nmf.subset(labels, res["labels"], res["candidate"], [added["id"]])
    ids, terms = nmf.name(model, fly, sub, keep={l: names[l] for l in names})
    assert ids[2][0] == "G5" and ids[3][0] is None and ids[added["label"]][0] in ("G2", "G3", None) and set(terms["rois"]) == set(ids)
    # ROIs cleaned since they were named: named by what they were then, so that with nothing found they get the names they had
    shrunk = np.where(ndi.binary_erosion(lab > 0, np.ones((1, 5, 5))), lab, 0)
    res = nmf.check(model, fly, shrunk, lut, movies=mv, split=False, named=lab)
    assert res["candidates"] == [] and res["names"]["joint"] == {l: v[0] for l, v in nmf.name(model, fly, lab)[0].items()} == lut
    # a candidate that cleaning leaves too small is dropped, its voxels as they were
    res = nmf.check(model, fly, labels, names, steps=["margin"], clean_params={"margin": {"erode_um": 3.5, "min_size_um2": 1.0}}, movies=mv, split=False)
    assert res["candidates"] == [] and np.array_equal(res["labels"], labels) and not res["candidate"].any()


def test_fixed_names():
    lab = truth()
    _, tuning = movies(lab)
    lut = {i + 1: f"G{i + 1}" for i in range(5)}
    resp = responses(lab, tuning)
    template = identity.build_template([{"responses": resp, "labels": lab, "lut": lut, "spacing": SPACING}])
    free = identity.assign_identities(resp, lab, template, SPACING, w_shape=0, w_layout=0)
    assert {l: v[0] for l, v in free.items()} == lut
    # ROI 1 keeps a name that is not its own, ROI 2 stays unnamed: no other gets G3, and G2 is left over
    ids, terms = identity.assign_identities(resp, lab, template, SPACING, w_shape=0, w_layout=0, fixed={1: "G3", 2: None}, return_terms=True)
    assert {l: v[0] for l, v in ids.items()} == {1: "G3", 2: None, 3: None, 4: "G4", 5: "G5"}
    assert terms["rois"][1]["assigned"] == "G3" and terms["rois"][2]["assigned"] is None
    # a name no glomerulus of the template has is kept as it is
    ids = identity.assign_identities(resp, lab, template, SPACING, w_shape=0, w_layout=0, fixed={1: "mine", 2: "G2", 3: "G2"})
    assert {l: v[0] for l, v in ids.items()} == {1: "mine", 2: "G2", 3: "G2", 4: "G4", 5: "G5"}


def test_frames(tmp_path):
    rng = np.random.default_rng(0)
    movie = rng.integers(100, 200, (30, 2, 6, 8)).astype(np.uint16)
    trials = [{"start_frame": 0, "first_odor_frame": 3, "end_frame": 9, "odors": "a @ -3"},
              {"start_frame": 10, "first_odor_frame": 14, "end_frame": 19, "odors": "b @ -3"},
              {"start_frame": 20, "first_odor_frame": 20, "end_frame": 25, "odors": "a @ -3"},  # (no time point before its odor: left out)
              {"start_frame": 26, "first_odor_frame": 28, "end_frame": 40, "odors": "a @ -3"}]  # (past the movie's end: as far as it goes)
    for name, m in (("one", movie), ("two", movie[:, :, ::-1]), ("other_grid", movie[:, :1]), ("no_trials", movie)):
        (tmp_path / name).mkdir()
        tifffile.imwrite(tmp_path / name / evidence.MOVIE, m, imagej=True, metadata={"axes": "TZYX"})
        if name != "no_trials":
            (tmp_path / name / evidence.TRIALS).write_text(json.dumps(trials))
    seen = []
    mv = evidence.frames(tmp_path, (2, 6, 8), progress=lambda *a: seen.append(a))
    assert mv["recordings"] == ["one", "two"] and seen[-1][0] == seen[-1][1]
    assert mv["dff"].shape == (2 * (10 + 10 + 4), 2, 6, 8) and mv["dff"].dtype == np.float16
    assert mv["trials"] == [("one", "a @ -3"), ("one", "b @ -3"), ("one", "a @ -3"), ("two", "a @ -3"), ("two", "b @ -3"), ("two", "a @ -3")]
    assert mv["trial"].tolist() == [0] * 10 + [1] * 10 + [2] * 4 + [3] * 10 + [4] * 10 + [5] * 4
    assert mv["on"][:10].tolist() == [False] * 3 + [True] * 7 and mv["on"][20:24].tolist() == [False, False, True, True]
    first = movie[:10].astype(np.float32)
    assert np.allclose(mv["dff"][:10], first - first[:3].mean(0), atol=0.1)
    assert np.allclose(mv["dff"][24:34], mv["dff"][:10][:, :, ::-1], atol=0.1)
    assert evidence.frames(tmp_path, (2, 6, 8), names=["two"])["recordings"] == ["two"]
    with pytest.raises(ValueError, match="no recording with mocorr.tif"):
        evidence.frames(tmp_path, (3, 6, 8))
