"""Write fake flies in the layout run_pipeline.py expects, for testing.

Each fly is defined in um: ~16 glomeruli (Voronoi cells inside an ellipsoidal AL about
100 x 85 x 75 um), each with a fixed 24-odor tuning shared across flies (two pairs
nearly identical), jittered positions, and a random pose per fly (in-plane rotation,
and shifts, including of the planes in depth). Functional recordings are sampled like
ours: 8-12 planes 8-12 um apart and 0.9-1.2 um pixels, with blur that mixes signal
across borders, and per-plane ImageJ ROIs named by glomerulus. Odd-numbered flies also
get a hires/ anatomical stack (3 um planes, 1.2 um pixels) with its own field of view.
Every recording gets a minimal ThorImage Experiment.xml with its voxel size.

Also writes <out>/reference/, a fixed reference atlas (labels.tif + glomeruli.csv,
the format of ~/src/al_atlas exports) from one more fake fly, mirrored and on another
grid, so aligning it needs a reflection and a scale.

Usage: python make_synthetic.py OUT [N_FLIES]
"""
import sys
from pathlib import Path

import numpy as np
import roifile
import tifffile
from scipy import ndimage as ndi
from skimage.measure import find_contours

N_GLOM, N_ODORS = 16, 24
AL_RADII = np.array([38.0, 42.0, 50.0])  # (z, y, x) um
HIRES_SHAPE, HIRES_SPACING = (36, 112, 112), (3.0, 1.2, 1.2)
rng0 = np.random.default_rng(0)
TUNING = np.clip(rng0.normal(0, 1, (N_GLOM, N_ODORS)), 0, None) ** 1.5  # sparse, positive
# G15/G16 and G13/G14 are tuned almost alike, so tuning alone confuses them
TUNING[15] = TUNING[14] * 0.9 + 0.1 * TUNING[15]
TUNING[13] = TUNING[12] * 0.85 + 0.15 * TUNING[13]
BASE = rng0.uniform(-0.6, 0.6, (N_GLOM, 3)) * AL_RADII  # glomerulus centers, (z, y, x) um
NAMES = [f"G{i + 1:02d}" for i in range(N_GLOM)]


def fly_model(seed):
    """Glomerulus centers and pose (in-plane rotation, (z, y, x) shift, um) of one fly."""
    rng = np.random.default_rng(seed)
    return {"centers": BASE + rng.normal(0, 3.0, BASE.shape),
            "theta": np.deg2rad(rng.uniform(-12, 12)), "shift": rng.uniform(-6, 6, 3)}


def sample(fly, shape, spacing, offset=(0.0, 0.0, 0.0)):
    """Labels and inter-glomerular gaps of ``fly`` at the voxel centers of a grid of
    ``shape`` and ``spacing`` (z, y, x), centered on the AL plus ``offset`` (um)."""
    zz, yy, xx = np.meshgrid(*[(np.arange(n) - (n - 1) / 2) * d + o
                               for n, d, o in zip(shape, spacing, offset)], indexing="ij")
    zz, yy, xx = zz - fly["shift"][0], yy - fly["shift"][1], xx - fly["shift"][2]
    c, s = np.cos(fly["theta"]), np.sin(fly["theta"])
    yy, xx = c * yy - s * xx, s * yy + c * xx
    inside = (zz / AL_RADII[0]) ** 2 + (yy / AL_RADII[1]) ** 2 + (xx / AL_RADII[2]) ** 2 < 1
    d = np.stack([(zz - p[0]) ** 2 + (yy - p[1]) ** 2 + (xx - p[2]) ** 2 for p in fly["centers"]])
    lab = d.argmin(0) + 1
    lab[~inside] = 0
    edges = ndi.morphological_gradient(lab, size=(1, 3, 3)) > 0  # thin gaps between glomeruli
    lab[edges] = 0
    return lab, edges & inside


def anatomy(lab, gaps, spacing, rng, noise=0.05):
    a = np.where(lab > 0, 1.0, 0.2) + 0.3 * gaps
    return (ndi.gaussian_filter(a, 1.5 / np.asarray(spacing)) + rng.normal(0, noise, lab.shape)).astype(np.float32)


def write_xml(path: Path, shape, spacing):
    """Minimal ThorImage Experiment.xml, with what hong2p reads voxel sizes from."""
    nz, ny, nx = shape
    path.write_text(f"""<?xml version="1.0"?>
<ThorImageExperiment>
  <LSM pixelX="{nx}" pixelY="{ny}" widthUM="{nx * spacing[2]:.5f}" heightUM="{ny * spacing[1]:.5f}" pixelSizeUM="{spacing[2]:.3f}" />
  <ZStage name="ThorZPiezo" steps="{nz}" stepSizeUM="{spacing[0]:.3f}" zStreamFrames="1" />
  <PMT enableA="1" enableB="0" />
</ThorImageExperiment>
""")


def write_fly(d: Path, seed: int, hires: bool):
    rng = np.random.default_rng(seed)
    fly = fly_model(seed)
    nz, dz, dxy = int(rng.integers(8, 13)), round(rng.uniform(8, 12), 3), round(rng.uniform(0.9, 1.2), 3)
    shape, spacing = (nz, 128, 128), (dz, dxy, dxy)
    lab, gaps = sample(fly, shape, spacing, (rng.uniform(-6, 6), 0, 0))
    resp = np.zeros((N_ODORS,) + shape, np.float32)
    for g in range(N_GLOM):
        resp += TUNING[g][:, None, None, None] * (lab == g + 1)
    # optical crosstalk: 3 um in z (a fraction of a plane), 2 um in-plane
    resp = ndi.gaussian_filter(resp, (0, 3 / dz, 2 / dxy, 2 / dxy)) + rng.normal(0, 0.15, resp.shape)

    rois = []
    for z in range(nz):
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

    d.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(d / "anatomy.tif", anatomy(lab, gaps, spacing, rng))
    tifffile.imwrite(d / "responses.tif", resp.astype(np.float32))
    (d / "RoiSet.zip").unlink(missing_ok=True)
    roifile.roiwrite(d / "RoiSet.zip", rois)
    write_xml(d / "Experiment.xml", shape, spacing)
    if hires:
        hl, hg = sample(fly, HIRES_SHAPE, HIRES_SPACING, rng.uniform(-5, 5, 3))
        (d / "hires").mkdir(exist_ok=True)
        tifffile.imwrite(d / "hires" / "anatomy.tif", anatomy(hl, hg, HIRES_SPACING, rng, 0.03))
        write_xml(d / "hires" / "Experiment.xml", HIRES_SHAPE, HIRES_SPACING)
    return len(rois), spacing


def write_reference(d: Path, seed: int = 999, shape=(40, 150, 150), spacing=(2.0, 0.75, 0.75)):
    lab = sample(fly_model(seed), shape, spacing)[0][:, :, ::-1]
    d.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(d / "labels.tif", lab.astype(np.uint8), imagej=True,
                     resolution=(1 / spacing[2], 1 / spacing[1]),
                     metadata={"spacing": spacing[0], "unit": "um"})
    with open(d / "glomeruli.csv", "w") as fh:
        fh.write("label,glomerulus\n")
        fh.writelines(f"{g + 1},{n}\n" for g, n in enumerate(NAMES))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    for i in range(n):
        d = out / f"fly{i + 1:02d}"
        n_rois, sp = write_fly(d, 100 + i, hires=i % 2 == 0)
        print(d, n_rois, "ROIs, voxels (z, y, x)", sp, "um", "+ hires/" if i % 2 == 0 else "")
    write_reference(out / "reference")
    print(out / "reference", "(reference atlas)")
