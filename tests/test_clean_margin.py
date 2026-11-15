"""The margin and core cleaning steps (`alseg.clean_margin`), and `segment.predict`'s cell
probability map (with Cellpose's model replaced by a stand-in), on small synthetic arrays."""
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels

from alseg import clean_margin, segment, synthetic


def fly(spacing=(10.0, 0.5, 0.5), **kw):
    return {"name": "test", "spacing": spacing, **kw}


def cols(L, z, lab):
    """The columns and rows label ``lab`` spans on plane z."""
    y, x = np.nonzero(L[z] == lab)
    return (x.min(), x.max() + 1), (y.min(), y.max() + 1)


def test_margin_touching():
    L = np.zeros((1, 30, 40), np.uint16)
    L[0, 5:15, 5:15], L[0, 5:15, 15:25] = 1, 2
    out = clean_margin.margin(L, fly(), erode_um=0.5, gap_um=0, min_size_um2=1)
    assert out.dtype == L.dtype
    assert cols(out, 0, 1) == ((6, 14), (6, 14)) and cols(out, 0, 2) == ((16, 24), (6, 14))  # 1 px all round
    # the side facing the other ROI 2 px (1 um) from its outline, so they end 2 um apart
    out = clean_margin.margin(L, fly(), erode_um=0.5, gap_um=1.0, min_size_um2=1)
    assert cols(out, 0, 1) == ((6, 13), (6, 14)) and cols(out, 0, 2) == ((17, 24), (6, 14))
    # 0.553 um pixels: 0.5 um is still one
    out = clean_margin.margin(L, fly((10.0, 0.553, 0.553)), erode_um=0.5, gap_um=0, min_size_um2=1)
    assert cols(out, 0, 1) == ((6, 14), (6, 14))
    # no margin: unchanged; the defaults: only the facing sides, 2 um each
    assert np.array_equal(clean_margin.margin(L, fly(), erode_um=0, gap_um=0), L)
    out = clean_margin.margin(L, fly())
    assert cols(out, 0, 1) == ((5, 11), (5, 15)) and cols(out, 0, 2) == ((19, 25), (5, 15))


def test_margin_image_edge_planes_and_pieces():
    L = np.zeros((2, 40, 40), np.int32)
    L[0, :8, 32:] = 3  # in the image's corner
    L[0, 10:20, 5:15], L[1, 10:20, 15:25] = 1, 2  # side by side, but on different planes
    L[1, 30:38, 20:28], L[1, 33, 28:31], L[1, 31:37, 31:37] = 4, 4, 4  # two pieces joined by a thin neck
    out = clean_margin.margin(L, fly(), erode_um=0.5, gap_um=2.0, min_size_um2=1)
    assert cols(out, 0, 3) == ((33, 40), (0, 7))  # not eroded along the image's edge
    assert cols(out, 0, 1) == ((6, 14), (11, 19)) and cols(out, 1, 2) == ((16, 24), (11, 19))  # in-plane only
    assert cols(out, 1, 4) == ((21, 27), (31, 37))  # the neck gone, and with it the smaller piece
    assert np.all((out == 0) | (out == L))


def test_margin_min_size():
    L = np.zeros((1, 20, 30), np.int32)
    L[0, 2:8, 2:8] = 1  # 9 um2, of which 1 um2 is over 1 um inside
    L[0, 12:15, 12:15] = 2  # 2.25 um2
    out = clean_margin.margin(L, fly(), erode_um=1.0, gap_um=0, min_size_um2=4)
    assert cols(out, 0, 1) == ((3, 7), (3, 7))  # its 16 px (4 um2) farthest inside
    assert np.array_equal(out == 2, L == 2)  # no bigger than the least size: left whole
    out = clean_margin.margin(L, fly(), erode_um=1.0, gap_um=0, min_size_um2=0.5)
    assert cols(out, 0, 1) == ((4, 6), (4, 6))


@pytest.mark.parametrize("erode,gap", [(1.0, 0.0), (1.0, 2.0), (0.5, 3.0)])
def test_margin_synthetic_fly(erode, gap):
    sp = (9.0, 1.1, 1.1)
    L = synthetic.sample(synthetic.fly_model(101), (5, 96, 96), sp)[0]
    L = np.stack([expand_labels(z, 1) for z in L])  # touching, as Cellpose's masks often are
    out = clean_margin.margin(L, fly(sp), erode_um=erode, gap_um=gap, min_size_um2=10)
    assert np.all((out == 0) | (out == L))
    half, min_px = sp[1] / 2, int(np.ceil(10 / (sp[1] * sp[2])))
    n_full = 0
    for z in range(L.shape[0]):
        for lab in np.unique(L[z])[1:]:
            m, kept = L[z] == lab, out[z] == lab
            assert kept.any(), "no ROI may vanish"
            assert ndi.label(kept)[1] == 1
            # (on the whole plane, not the box around the mask the step works in)
            ok = m & (ndi.distance_transform_edt(m, sampling=sp[1:]) - half - erode >= 0)
            others = (L[z] > 0) & ~m
            if others.any():
                ok &= ndi.distance_transform_edt(~others, sampling=sp[1:]) - half - gap >= 0
            cc, n = ndi.label(ok)
            biggest = np.bincount(cc.ravel())[1:].max() if n else 0
            if biggest >= min_px:
                assert kept.sum() == biggest and ok[kept].all()
                n_full += 1
            else:
                assert kept.sum() >= min(min_px, m.sum())
    assert n_full > 20


def test_core():
    L = np.zeros((2, 30, 30), np.uint16)
    L[0, 5:25, 5:25], L[1, 5:25, 5:25] = 7, 7
    yy, xx = np.mgrid[:30, :30]
    r = 0.5 * np.hypot(yy - 14.5, xx - 14.5)  # um from the mask's centre
    prob = np.stack([3 - r, np.maximum(3 - r, 3 - 0.5 * np.hypot(yy - 21, xx - 21) - 0.5)]).astype(np.float32)
    out = clean_margin.core(L, fly(cellprob=prob), core_cellprob=0, min_size_um2=1)
    assert out.dtype == L.dtype
    assert np.array_equal(out[0] == 7, (L[0] == 7) & (prob[0] >= 0))
    # two pieces over the threshold: the larger kept
    two = (L[1] == 7) & (prob[1] >= 2)
    assert ndi.label(two)[1] == 2
    out = clean_margin.core(L, fly(cellprob=prob), core_cellprob=2, min_size_um2=0.25)
    assert np.array_equal(out[1] == 7, (L[1] == 7) & (prob[0] >= 2))
    # too small: the most probable 10 um2 (40 px) instead
    out = clean_margin.core(L, fly(cellprob=prob), core_cellprob=2.8, min_size_um2=10)
    kept = out[0] == 7
    assert 40 <= kept.sum() <= 48 and kept[14:16, 14:16].all() and prob[0][kept].min() >= prob[0][(L[0] == 7) & ~kept].max()
    with pytest.raises(ValueError, match="cell probability"):
        clean_margin.core(L, fly())
    with pytest.raises(ValueError, match="cell probability"):
        clean_margin.core(L, fly(cellprob=prob[:1]))


class StandIn:
    """Cellpose's model, as `segment.predict` calls it: cell probability (logits) 4 * the first
    channel - 2, masks where it is over the threshold (squeezed, as Cellpose's are)."""

    def __init__(self, gpu=False, pretrained_model=None):
        pass

    def eval(self, x, z_axis=None, cellprob_threshold=0.0, **kw):
        prob = x[..., 0] * 4 - 2
        masks = np.stack([ndi.label(p > cellprob_threshold)[0] for p in prob])
        return masks.squeeze(), [None, np.zeros((2,) + prob.shape).squeeze(), prob.squeeze()], None


@pytest.mark.parametrize("nz", [1, 3])
@pytest.mark.parametrize("tuning", [False, True])
def test_predict_cellprob(monkeypatch, nz, tuning):
    cp = pytest.importorskip("cellpose.models")
    monkeypatch.setattr(cp, "CellposeModel", StandIn)
    yy, xx = np.mgrid[:20, :24]
    blob = np.exp(-((yy - 9) ** 2 + (xx - 11) ** 2) / 30.0)
    ch = np.stack([np.stack([blob] * nz)] * 3).astype(np.float32)
    resp = np.stack([ch[0], 1 - ch[0]]) if tuning else None
    kw = dict(spacing=(10.0, 1.0, 1.0), xy_um=0.5, gpu=False, cellprob_threshold=-1.0, responses=resp)
    masks = segment.predict("m", ch, **kw)
    m2, prob = segment.predict("m", ch, return_cellprob=True, **kw)
    assert np.array_equal(masks, m2) and masks.shape == (nz, 20, 24)
    assert prob.dtype == np.float32 and prob.shape == (nz, 20, 24)
    # resampled to the model's 0.5 um pixels and back: close to the stand-in's map on the fly's grid
    assert np.abs(prob - (4 * ch[0] - 2)).max() < 0.15
    assert (prob[masks > 0] > -1.2).all() and masks.any()
