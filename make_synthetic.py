"""Write a few fake flies in the layout run_pipeline.py expects, for testing.

Each fly has ~16 glomeruli (Voronoi cells inside an ellipsoid), each with a
fixed 24-odor tuning shared across flies (two pairs nearly identical),
jittered positions, a random in-plane rotation and shift per fly, blur that
mixes signal across borders, and per-slice ImageJ ROIs named by glomerulus.

Also writes <out>/reference/, a fixed reference atlas (labels.tif + glomeruli.csv,
the format of ~/src/al_atlas exports) from one more fake fly, mirrored and on a finer
in-plane grid (2/3 um vs. the flies' 1 um), so aligning it needs a reflection and a
scale.
"""
import sys
from pathlib import Path

import numpy as np
import roifile
import tifffile
from scipy import ndimage as ndi
from skimage.measure import find_contours

N_GLOM, N_ODORS, SHAPE = 16, 24, (10, 128, 128)
rng0 = np.random.default_rng(0)
TUNING = np.clip(rng0.normal(0, 1, (N_GLOM, N_ODORS)), 0, None) ** 1.5  # sparse, positive
# G15/G16 and G13/G14 are tuned almost alike, so tuning alone confuses them
TUNING[15] = TUNING[14] * 0.9 + 0.1 * TUNING[15]
TUNING[13] = TUNING[12] * 0.85 + 0.15 * TUNING[13]
BASE_POS = rng0.uniform(-0.6, 0.6, (N_GLOM, 2))
BASE_Z = rng0.uniform(-0.7, 0.7, N_GLOM)
NAMES = [f"G{i + 1:02d}" for i in range(N_GLOM)]


def make_fly(seed):
    rng = np.random.default_rng(seed)
    Z, Y, X = SHAPE
    zz, yy, xx = np.meshgrid(np.linspace(-1, 1, Z), np.linspace(-1, 1, Y),
                             np.linspace(-1, 1, X), indexing="ij")
    # each fly sits at a slightly different pose: in-plane rotation + shift
    th = np.deg2rad(rng.uniform(-12, 12))
    sy, sx = rng.uniform(-0.08, 0.08, 2)
    yy, xx = (np.cos(th) * (yy - sy) - np.sin(th) * (xx - sx),
              np.sin(th) * (yy - sy) + np.cos(th) * (xx - sx))
    inside = (zz / 1.1) ** 2 + yy**2 + xx**2 < 0.8
    pos = BASE_POS + rng.normal(0, 0.05, BASE_POS.shape)
    zpos = BASE_Z + rng.normal(0, 0.1, N_GLOM)
    d = np.stack([(yy - p[0]) ** 2 + (xx - p[1]) ** 2 + 0.5 * (zz - zp) ** 2
                  for p, zp in zip(pos, zpos)])
    lab = d.argmin(0) + 1
    lab[~inside] = 0
    # thin gaps between glomeruli
    edges = ndi.morphological_gradient(lab, size=(1, 3, 3)) > 0
    lab[edges] = 0

    anat = np.where(lab > 0, 1.0, 0.2) + 0.3 * edges * inside
    anat = ndi.gaussian_filter(anat, (0.7, 1.5, 1.5)) + rng.normal(0, 0.05, SHAPE)
    resp = np.zeros((N_ODORS,) + SHAPE, np.float32)
    for g in range(N_GLOM):
        resp += TUNING[g][:, None, None, None] * (lab == g + 1)
    resp = ndi.gaussian_filter(resp, (0, 1.2, 2.0, 2.0))  # optical crosstalk, worse in z
    resp += rng.normal(0, 0.15, resp.shape)

    rois = []
    for z in range(Z):
        for g in range(N_GLOM):
            m = lab[z] == g + 1
            if m.sum() < 15:
                continue
            c = max(find_contours(m.astype(float), 0.5), key=len)
            r = roifile.ImagejRoi.frompoints(c[:, ::-1])  # (x, y)
            r.roitype = roifile.ROI_TYPE.POLYGON
            r.position = z + 1
            r.name = f"{NAMES[g]}_z{z + 1:02d}"  # zip entry names must be unique
            rois.append(r)
    return anat.astype(np.float32), resp, rois, lab


def write_reference(d: Path, seed: int = 999, zoom: float = 1.5):
    lab = make_fly(seed)[3][:, :, ::-1]
    lab = ndi.zoom(lab, (1, zoom, zoom), order=0)
    d.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(d / "labels.tif", lab.astype(np.uint8), imagej=True,
                     resolution=(zoom, zoom), metadata={"spacing": 1.0, "unit": "um"})
    with open(d / "glomeruli.csv", "w") as fh:
        fh.write("label,glomerulus\n")
        fh.writelines(f"{g + 1},{n}\n" for g, n in enumerate(NAMES))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    for i in range(n):
        d = out / f"fly{i + 1:02d}"
        d.mkdir(parents=True, exist_ok=True)
        anat, resp, rois, _ = make_fly(100 + i)
        tifffile.imwrite(d / "anatomy.tif", anat)
        tifffile.imwrite(d / "responses.tif", resp)
        zp = d / "RoiSet.zip"
        zp.unlink(missing_ok=True)
        roifile.roiwrite(zp, rois)
        print(d, len(rois), "ROIs")
    write_reference(out / "reference")
    print(out / "reference", "(reference atlas)")
