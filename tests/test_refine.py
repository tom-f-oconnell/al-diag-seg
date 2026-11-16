"""Cellpose's ROIs cleaned of their neighbours (`alseg.model.Model.refine`), as `predict` does when asked and as can be
done after, from what a prediction keeps: masks over two glomeruli split and joined again, named again; then trimmed,
with what each step did to each ROI. With the stand-in segmenter, on small synthetic flies (no GPU)."""
import numpy as np

from alseg import model as am

SPACING = (10.0, 0.5, 0.5)
ODORS = [f"odor{i} @ -3" for i in range(6)]
GLOMS = ["A", "B", "C"]
# each glomerulus' tuning: A and B unlike each other (C, the one other mask on a plane, which a mask is centered by, weakly tuned)
TUNING = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0.4, 0.2, 0.3, 0.1, 0.3, 0.5]], np.float32)


def fly(seed, name=None):
    """A fly of 3 glomeruli on 3 planes: A and B side by side (touching), C apart."""
    rng = np.random.default_rng(seed)
    lab = np.zeros((3, 40, 60), np.int32)
    dy, dx = rng.integers(0, 3, 2)
    lab[:, 6 + dy:22 + dy, 4 + dx:20 + dx] = 1
    lab[:, 6 + dy:22 + dy, 20 + dx:36 + dx] = 2
    lab[:, 24:36, 42:56] = 3
    resp = rng.normal(0, 0.05, (len(ODORS),) + lab.shape).astype(np.float32)
    for i in range(3):
        resp[:, lab == i + 1] += TUNING[i][:, None]
    return {"name": name or f"fly{seed}", "labels": lab, "lut": {i + 1: g for i, g in enumerate(GLOMS)}, "spacing": SPACING,
            "average": (1 + (lab > 0) + rng.normal(0, 0.05, lab.shape)).astype(np.float32), "odors": list(ODORS), "responses": resp}


def merged(f):
    """The stand-in's ROIs of a fly with A and B as one mask on every plane (as Cellpose's may be)."""
    lab = f["labels"].copy()
    lab[lab == 2] = 1
    return lab


def model(**kw):
    o = am.Options(segmenter="stand-in", atlas=False, gpu=False, **kw)
    return am.fit([fly(s) for s in range(4)], options=o, odors=ODORS)


def test_refine_split_and_trim(monkeypatch):
    m = model()
    monkeypatch.setattr(am, "_stand_in", merged)
    f = fly(10)
    p = m.predict(f)
    assert p["final"] == "pred" and p["report"] is None and len(np.unique(p["labels"]["pred"])) - 1 == 2
    r = m.refine(f, p["labels"]["pred"], p["identities"]["pred"], p["identity_terms"]["pred"], split=True, steps=["trim"])
    assert r["steps"] == ["split", "trim"] and r["final"] == "clean" and set(r["labels"]) == {"split", "clean"}
    split = r["labels"]["split"]
    # A and B each an ROI again, on every plane, and named so; C as it was
    names = {v[0] for v in r["identities"]["split"].values()}
    assert len(np.unique(split)) - 1 == 3 and names == set(GLOMS)
    by = {v[0]: l for l, v in r["identities"]["split"].items()}
    assert {z for z in range(3) if (split[z] == by["A"]).any() and (split[z] == by["B"]).any()} == {0, 1, 2}
    a_ab = p["labels"]["pred"][0][f["labels"][0] == 1][0]  # (the merged mask's label in Cellpose's)
    assert r["report"]["split"] == {by["A"]: {0: a_ab, 1: a_ab, 2: a_ab}, by["B"]: {0: a_ab, 1: a_ab, 2: a_ab}}
    # trim only removes pixels, keeping labels and names, and says where it cut
    clean = r["labels"]["clean"]
    assert np.all((clean == 0) | (clean == split)) and r["identities"]["clean"] == {l: v for l, v in r["identities"]["split"].items()
                                                                                     if l in set(np.unique(clean).tolist())}
    for lab, planes in r["report"]["trimmed"].items():
        for z, kept in planes.items():
            assert 0 < kept < 1 and (clean[z] == lab).sum() < (split[z] == lab).sum()
    # as predict does it, when the options ask
    m.options.split, m.options.clean, m.options.final = True, ["trim"], "clean"
    q = m.predict(fly(10))
    assert q["final"] == "clean" and np.array_equal(q["labels"]["pred"], p["labels"]["pred"])
    assert np.array_equal(q["labels"]["split"], split) and np.array_equal(q["labels"]["clean"], clean) and q["report"] == r["report"]
    assert np.array_equal(q["labels"]["clean_trim"], clean)  # (each step alone: on the split ROIs)


def test_refine_nothing_to_split(monkeypatch):
    """No mask over two glomeruli: the ROIs and their names as they were; steps that cannot run (core, without Cellpose's
    probabilities) left out, and their kind as the ROIs before them."""
    m = model()
    f = fly(11)
    p = m.predict(f)
    r = m.refine(f, p["labels"]["pred"], p["identities"]["pred"], split=True, steps=["core"])
    assert r["steps"] == ["split"] and r["report"] == {"split": {}, "trimmed": {}}
    assert np.array_equal(r["labels"]["split"], p["labels"]["pred"]) and r["identities"]["split"] == p["identities"]["pred"]
    assert np.array_equal(r["labels"]["clean"], p["labels"]["pred"]) and r["final"] == "clean"


def test_score_without_own_rois():
    """A fly with no ROIs of its own: only how its predicted ROIs keep from each other is scored."""
    m = model()
    f = fly(12)
    p = m.predict(f)
    f.pop("labels")
    s = am.score(f, p["labels"]["pred"], p["identities"]["pred"])
    assert s["identity"] is None and s["row"]["recall"] == "" and s["row"]["n"] == 3 and s["row"]["n_2d"] == 9
    assert isinstance(s["row"]["leak"], float) and s["row"]["touching"] == 2 / 3  # (A and B touch, C does not)
