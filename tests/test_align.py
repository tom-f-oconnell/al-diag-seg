"""A fly's atlas moved to where its ROIs say its glomeruli are (`alseg.align`, `Model.align`): the fit to named ROIs (a
shift or an affine map, whatever a few wrong names say), the atlas' maps resampled by it, and `Model.predict` naming a
fly's ROIs with the atlas moved: as fit to the ROIs that tuning alone names, or, for a fly with few odors, to those
named with the atlas. On small synthetic flies, without registration (the atlas on a fly's grid is made here)."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from alseg import align
from alseg import atlas as atlas_mod
from alseg import model as am

SPACING = (10.0, 0.5, 0.5)
SHAPE = (6, 96, 96)
# 9 glomeruli: 3 on each of planes 1, 3 and 5, in triangles turned from plane to plane (um; z, y, x)
CENTERS = np.array([[10 + 20 * k, 24 + 11 * np.cos(a + k), 24 + 11 * np.sin(a + k)] for k in range(3) for a in (0.3, 2.4, 4.5)])
GLOMS = [f"G{i}" for i in range(len(CENTERS))]
ODORS = [f"odor{i} @ -3" for i in range(20)]
TUNING = np.random.default_rng(0).normal(0, 1, (len(GLOMS), len(ODORS))).astype(np.float32)


def blobs(centers, sigma=(8.0, 4.0, 4.0)):
    """An atlas on the grid: each glomerulus' probability a Gaussian blob at its centre (um)."""
    grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(SHAPE, SPACING)], indexing="ij"), -1)
    return np.stack([np.exp(-(((grid - c) / sigma) ** 2).sum(-1) / 2) for c in centers]).astype(np.float32)


def rois(centers, radius=5.0):
    """ROIs on the grid: each a disk on the plane nearest its centre (um)."""
    lab = np.zeros(SHAPE, np.int32)
    yy, xx = np.meshgrid(np.arange(SHAPE[1]) * SPACING[1], np.arange(SHAPE[2]) * SPACING[2], indexing="ij")
    for i, c in enumerate(centers):
        lab[int(round(c[0] / SPACING[0]))][(yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= radius ** 2] = i + 1
    return lab


def moved(centers, A, b):
    return centers @ np.asarray(A, float).T + np.asarray(b, float)


def test_fit_shift_and_affine():
    rng = np.random.default_rng(1)
    e = rng.uniform([0, 0, 0], [90, 100, 100], (30, 3))
    A = np.array([[1.25, 0.0, 0.0], [0.02, 1.05, -0.08], [-0.03, 0.08, 0.95]])
    b = np.array([-6.0, 3.0, -4.0])
    r = moved(e, A, b) + rng.normal(0, 0.5, e.shape)
    r[:4] += rng.choice([-1, 1], (4, 3)) * 25  # (ROIs named wrong: far from where the others put their glomerulus)
    f = align.fit(e, r, "affine", ridge=0)
    assert np.allclose(f["matrix"], A, atol=0.02) and np.allclose(f["offset_um"], b, atol=1.5)
    assert f["rois"] == 30 and np.allclose(f["weights"][:4], 0) and min(f["weights"][4:]) > 0.5
    assert f["after_um"] < 1.5 < f["before_um"]
    assert np.allclose(align.apply(f, e[4:]), r[4:], atol=2.5)
    # held to the identity (the default), it is the same map but for a little less of it
    held = align.fit(e, r, "affine")
    assert np.allclose(held["matrix"], A, atol=0.06) and np.abs(np.array(held["matrix"]) - np.eye(3)).sum() < np.abs(A - np.eye(3)).sum()
    # a shift: the median offset, whatever the wrong names say
    s = align.fit(e, e + [2.0, -3.0, 5.0] + np.r_[np.full((4, 3), 30.0), np.zeros((26, 3))], "shift")
    assert np.allclose(s["matrix"], np.eye(3)) and np.allclose(s["offset_um"], [2, -3, 5]) and np.allclose(s["shift_um"], [2, -3, 5])
    assert np.allclose(s["weights"][:4], 0) and np.allclose(s["weights"][4:], 1) and s["after_um"] == 0
    # few named ROIs: an affine map of them is little more than their shift; fewer than MIN_ROIS: no fit
    few = align.fit(e[4:10], r[4:10], "affine")
    assert np.abs(np.array(few["matrix"]) - np.eye(3)).max() < np.abs(A - np.eye(3)).max()
    assert align.fit(e[:align.MIN_ROIS - 1], r[:align.MIN_ROIS - 1]) is None and align.fit(np.zeros((0, 3)), np.zeros((0, 3))) is None
    with pytest.raises(ValueError, match="one of shift, affine"):
        align.fit(e, r, "warp")
    # ROIs all on one plane say nothing of z: the atlas is only shifted along it, whatever holds it to the identity
    flat = e[4:] * [0, 1, 1] + [30.0, 0, 0]
    for ridge in (0, align.RIDGE):
        f = align.fit(flat, moved(flat, A, b), "affine", ridge=ridge)
        assert f["kind"] == "affine" and np.allclose(np.array(f["matrix"])[:, 0], [1, 0, 0]) and np.allclose(np.array(f["matrix"])[1:, 1:], A[1:, 1:], atol=0.03)
        assert f["offset_um"][0] == pytest.approx(1.5, abs=1e-6)  # (their z: 30 um in the atlas, 0.25 * 30 - 6 more in the fly)
    # a map that would stretch the atlas over twofold is not believed: the shift instead
    wild = align.fit(e, moved(e, np.diag([1.0, 2.5, 1.0]), b), "affine", ridge=0)
    assert wild["kind"] == "shift" and np.allclose(wild["matrix"], np.eye(3))
    assert np.allclose(wild["offset_um"], np.median(moved(e, np.diag([1.0, 2.5, 1.0]), b) - e, 0))
    assert align.fit(e, moved(e, np.diag([1.0, 1.9, 1.0]), b), "affine", ridge=0)["kind"] == "affine"


def test_move_and_align():
    A, b = np.diag([1.2, 1.0, 1.0]), np.array([-8.0, 3.0, -2.5])
    atlas_at = moved(CENTERS, np.linalg.inv(A), -np.linalg.inv(A) @ b)  # (where the atlas has them: moved by (A, b), on the ROIs)
    prior, lab = blobs(atlas_at), rois(CENTERS)
    names = {i + 1: g for i, g in enumerate(GLOMS)}
    labs, expected, found = align.pairs(lab, {**names, 3: None, 4: "not in the atlas"}, prior, GLOMS, SPACING)
    assert labs == [1, 2, 5, 6, 7, 8, 9] and expected.shape == found.shape == (7, 3)
    assert np.allclose(found, CENTERS[[0, 1, 4, 5, 6, 7, 8]], atol=0.3)
    out, f = align.align(lab, names, prior, GLOMS, SPACING, "affine", ridge=0)
    assert out.shape == prior.shape and out.dtype == np.float32 and f["rois"] == 9
    assert f["after_um"] < 1 < f["before_um"]
    # each map's peak is at its ROI now (to a voxel in the plane, on its plane)
    sp = np.array(SPACING)
    for m, c in zip(out, CENTERS):
        at = np.array(np.unravel_index(int(m.argmax()), m.shape)) * sp
        assert at[0] == c[0] and np.abs(at[1:] - c[1:]).max() <= 1.0, (at, c)
    # a shift along z does not empty the plane it moves away from
    up = align.move(prior, {"matrix": np.eye(3).tolist(), "offset_um": [5.0, 0, 0]}, SPACING)
    assert np.allclose(up[:, 0], prior[:, 0]) and np.allclose(up[:, 1], (prior[:, 0] + prior[:, 1]) / 2, atol=1e-6)
    # too few named ROIs: the atlas as it is
    same, none = align.align(lab, {1: "G0", 2: "G1"}, prior, GLOMS, SPACING)
    assert same is prior and none is None


def fly(seed, shift=(0.0, 0.0, 0.0), jitter=0.0):
    """A fly of the 9 glomeruli (each tuned its own way), all moved by ``shift`` (um), and each on its own by about
    ``jitter`` in the plane."""
    rng = np.random.default_rng(seed)
    lab = rois(CENTERS + np.asarray(shift, float) + rng.normal(0, jitter, CENTERS.shape) * [0, 1, 1])
    resp = rng.normal(0, 0.05, (len(ODORS),) + lab.shape).astype(np.float32)
    for i in range(len(GLOMS)):
        resp[:, lab == i + 1] += TUNING[i][:, None]
    return {"name": f"fly{seed}", "labels": lab, "lut": {i + 1: g for i, g in enumerate(GLOMS)}, "spacing": SPACING,
            "average": (1 + (lab > 0) + rng.normal(0, 0.05, lab.shape)).astype(np.float32), "odors": list(ODORS), "responses": resp}


def model(**kw):
    """A model of such flies (their glomeruli each 2 um or so from the others') with an atlas that has the glomeruli
    where they are on average (no registration)."""
    m = am.fit([fly(s, jitter=2.0) for s in range(4)], options=am.Options(segmenter="stand-in", atlas=False, gpu=False, **kw), odors=ODORS)
    m.atlas = atlas_mod.Atlas(np.zeros((1, 1, 1), np.float32), np.zeros((len(GLOMS), 1, 1, 1), np.float16), list(GLOMS), (1.0, 1.0, 1.0))
    return m


def test_predict_moves_the_atlas(monkeypatch, tmp_path):
    registered = blobs(CENTERS)  # (the atlas as registration leaves it: where the training flies' glomeruli are)
    monkeypatch.setattr(am.Model, "prior_of", lambda self, f: registered.copy())
    shift = np.array([0.0, 7.0, -6.0])
    f = fly(10, shift)  # (a fly whose glomeruli are all 9 um from there)
    m = model()
    assert m.options.align == "affine"
    # the affine map is fit to the ROIs that odor tuning alone names (here all of them, each as its glomerulus)
    assert m.named_by_tuning(f, am._stand_in(f)) == {i + 1: g for i, g in enumerate(GLOMS)}
    steps, fit_and_move = [], align.align
    monkeypatch.setattr(align, "align", lambda labels, names, prior, *a, **kw: steps.append((a[-1], prior is not None and np.array_equal(prior, registered)))
                        or fit_and_move(labels, names, prior, *a, **kw))
    p = m.predict(f)
    assert steps == [("affine", True)] and p["alignment"]["rois_named_by"] == "tuning"
    a = p["alignment"]
    # (along z less exactly: a map's centre of mass on the fly's few planes is drawn inwards from the first and last)
    assert a["kind"] == "affine" and a["rois"] == 9 and np.allclose(a["shift_um"][1:], shift[1:], atol=1.0) and abs(a["shift_um"][0]) < 2
    assert a["after_um"] < 1.5 and 8 < a["before_um"] < 10.5
    # the atlas it gives is the moved one: each map's centre of mass at the fly's glomerulus
    for k, c in enumerate(CENTERS + shift):
        at = np.array(ndi.center_of_mass(p["prior"][k])) * SPACING
        assert np.abs(at[1:] - c[1:]).max() < 1.5, (k, at, c)
    assert np.array_equal(p["atlas_maxprob"], p["prior"].max(0))
    # and what its ROIs are named by: every name right, at a lower cost than with the atlas as registered
    names = {l: v[0] for l, v in p["identities"]["pred"].items()}
    assert names == {i + 1: g for i, g in enumerate(GLOMS)}
    assert p["identity_terms"]["pred"]["alignment"] == a and "alignment" not in p["identity_terms"]["refined_tuning"]
    off = model(align=None).predict(f)
    assert off["alignment"] is None and np.array_equal(off["prior"], registered)
    assert off["identity_terms"]["pred"]["alignment"] is None
    cost = lambda q: np.mean([v[1] for v in q["identities"]["pred"].values()])  # noqa: E731
    assert cost(p) < cost(off) - 0.05
    pos = lambda q: np.mean([c["position"] for r in q["identity_terms"]["pred"]["rois"].values() for c in r["candidates"][:1]])  # noqa: E731
    assert pos(p) < pos(off) - 0.3
    # a shift alone: fit to the ROIs as named with the atlas
    del steps[:]
    s = model(align="shift").predict(f)
    assert s["alignment"]["kind"] == "shift" and np.allclose(s["alignment"]["matrix"], np.eye(3))
    assert np.allclose(s["alignment"]["offset_um"], shift, atol=0.5) and s["alignment"]["rois_named_by"] == "atlas" and steps == [("shift", True)]
    # a fly with few of the model's odors, of which tuning alone says too little: the affine map too is fit to the ROIs
    # as named with the atlas, after a shift (each a fit of the atlas as registered)
    few = {**f, "odors": ODORS[:align.TUNING_ODORS - 1], "responses": f["responses"][:align.TUNING_ODORS - 1]}
    del steps[:]
    q = m.predict(few)
    assert steps == [("shift", True), ("affine", True)] == [(k, True) for k in align.STEPS["affine"]]
    assert q["alignment"]["kind"] == "affine" and q["alignment"]["rois_named_by"] == "atlas" and q["alignment"]["after_um"] < 1.5
    # so is it where the map fit to the ROIs named by tuning is not one to believe (or there are too few of them)
    monkeypatch.setattr(am.Model, "named_by_tuning", lambda self, fly, labels: {1: "G0", 2: "G1"})
    del steps[:]
    assert m.predict(f)["alignment"]["rois_named_by"] == "atlas" and steps == [("affine", True), ("shift", True), ("affine", True)]
    monkeypatch.undo()
    monkeypatch.setattr(am.Model, "prior_of", lambda self, f: registered.copy())
    # scored (evaluate): the moved atlas and how it was moved are what is written
    res = m.evaluate([f], tmp_path, priors=True)
    assert res["alignment"] == {f["name"]: a}
    z = np.load(tmp_path / f"{f['name']}_atlas_prior.npz")
    assert np.allclose(z["prior"].astype(np.float32), p["prior"], atol=1e-3) and am.read_alignment(z) == a
    assert am.read_alignment(np.load(_plain(tmp_path, registered))) is None
    # saved and loaded, the option is kept; a model saved before it was one takes the default
    m.atlas = None
    m.options.align = "shift"
    m.save(tmp_path / "model")
    assert am.Model.load(tmp_path / "model").options.align == "shift"


def _plain(folder, prior):
    am.write_prior(folder / "plain.npz", prior, GLOMS)
    return folder / "plain.npz"


def test_too_few_names_leave_the_atlas(monkeypatch):
    registered = blobs(CENTERS)
    monkeypatch.setattr(am.Model, "prior_of", lambda self, f: registered.copy())
    m = model(max_cost=0.0)  # (no ROI is named with the atlas)
    f = fly(11, (0.0, 7.0, -6.0))
    # the ROIs that tuning alone names are enough to move the atlas by, whatever a name may cost
    p = m.predict(f)
    assert p["alignment"]["rois_named_by"] == "tuning" and not any(v[0] for v in p["identities"]["pred"].values())
    # none of those either (a fly with few odors is not named by tuning alone): nothing to fit the atlas to
    few = {**f, "odors": ODORS[:4], "responses": f["responses"][:4]}
    p = m.predict(few)
    assert p["alignment"] is None and np.array_equal(p["prior"], registered)
    assert not any(v[0] for v in p["identities"]["pred"].values())
