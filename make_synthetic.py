"""Write fake flies as an export from the odor response database (hong-data's ROI training
tab), which run_pipeline.py reads, for testing.

Each fly is defined in um: ~16 glomeruli (Voronoi cells inside an ellipsoidal AL about
100 x 85 x 75 um), each with a fixed 24-odor tuning shared across flies (two pairs
nearly identical), jittered positions, and a random pose per fly (in-plane rotation,
and shifts, including of the planes in depth). Functional recordings are sampled like
ours: 8-12 planes 8-12 um apart and 0.9-1.2 um pixels, with blur that mixes signal
across borders, and per-plane ImageJ ROIs named by glomerulus. Odd-numbered flies also
get an anatomical stack (3 um planes, 1.2 um pixels) with its own field of view, as
ThorImage writes it (ChanA_*.tif, two frames per plane), and as recorded: flies 3 and 7 are
right ALs, whose functional data al_analysis mirrored left-right, so their stacks are
mirrored relative to it. Each fly's 24 odors are split between two diagnostic recordings,
in an order of its own; fly 4 has no odor24. Each RoiSet also has a numbered ROI (not to
be trained on) and its G16 is tentative (G16_t0) in fly 2, and in fly 5 next to a G16 too. Every recording gets a minimal ThorImage Experiment.xml with its
voxel size, in the fly's raw data folder.

Also writes <out>/reference/, a fixed reference atlas (labels.tif + glomeruli.csv,
the format of ~/src/al_atlas exports) from one more fake fly, mirrored and on another
grid, so aligning it needs a reflection and a scale.

Usage: python make_synthetic.py OUT [N_FLIES]
"""
import json
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
USER, DATE = "Ann", "2026-01-01"  # whose data the flies are (the top folder of their paths), and when
RECORDINGS = (("diagnostics1", range(0, 12)), ("diagnostics2", range(12, 24)))
ODORS = [f"odor{i + 1:02d} @ -3" for i in range(N_ODORS)]
RIGHT_ALS = (3, 7)  # flies whose functional data al_analysis mirrored (their stacks are not)
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


def fluorescence(lab, gaps, spacing, rng, noise=0.05):
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


def write_stack(d: Path, stack, rng):
    """An anatomical stack as ThorImage writes it: one TIFF per frame, two frames per plane."""
    d.mkdir(parents=True, exist_ok=True)
    frame = 0
    for z, plane in enumerate(stack):
        for _ in range(2):
            frame += 1
            tifffile.imwrite(d / f"ChanA_0001_0001_{z + 1:04d}_{frame:04d}.tif",
                             np.clip((plane + rng.normal(0, 0.02, plane.shape)) * 1000, 0, None).astype(np.uint16))
    write_xml(d / "Experiment.xml", stack.shape, HIRES_SPACING)


def write_fly(out: Path, num: int, seed: int, hires: bool) -> dict:
    """One fly, where al_analysis leaves its files (in the folders of its motion-corrected
    and raw data); its entry in the export's roi_training.json."""
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
            r.name = NAMES[g] + ("_t0" if g == N_GLOM - 1 and num in (2, 5) else "")
            rois.append(r)
            if g == N_GLOM - 1 and num == 5:  # a plain G16 too, so the tentative one is left out
                plain = roifile.ImagejRoi.frompoints(c[:, ::-1])
                plain.roitype, plain.position, plain.name = roifile.ROI_TYPE.POLYGON, z + 1, NAMES[g]
                rois.append(plain)
    numbered = roifile.ImagejRoi.frompoints(np.array([[10, 10], [30, 10], [30, 30], [10, 30]]))
    numbered.roitype, numbered.position, numbered.name = roifile.ROI_TYPE.POLYGON, 1, "12"
    rois.append(numbered)

    rel = lambda p: p.relative_to(out).as_posix()  # noqa: E731
    adir, rdir = out / USER / "analysis" / DATE / str(num), out / USER / "raw" / DATE / str(num)
    adir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(adir / "mocorr_concat_avg.tif", fluorescence(lab, gaps, spacing, rng), imagej=True, metadata={"axes": "ZYX"})
    recordings = []
    for name, odors in RECORDINGS:
        odors = [i for i in rng.permutation(list(odors)) if not (num == 4 and i == N_ODORS - 1)]
        (adir / name).mkdir(exist_ok=True)
        tifffile.imwrite(adir / name / "trialmean_dff.tif", resp[odors].astype(np.float32), imagej=True,
                         metadata={"axes": "TZYX"})
        trials = [{"start_frame": 0, "first_odor_frame": 1, "end_frame": 2, "odors": ODORS[i], "name": ODORS[i].split()[0],
                   "log10_conc": -3} for i in odors for _ in range(2)]
        (adir / name / "trial_frames_and_odors.json").write_text(json.dumps(trials))
        (rdir / name).mkdir(parents=True, exist_ok=True)
        write_xml(rdir / name / "Experiment.xml", shape, spacing)
        recordings.append({"name": name, "responses": rel(adir / name / "trialmean_dff.tif"),
                           "trials": rel(adir / name / "trial_frames_and_odors.json"),
                           "experiment_xml": rel(rdir / name / "Experiment.xml"),
                           "odors": [ODORS[i] for i in odors], "as_recorded": [ODORS[i] for i in odors]})
    roiset = adir / RECORDINGS[0][0] / "RoiSet.zip"
    roiset.unlink(missing_ok=True)
    # ImageJ's zip entry names (unique), as the ROIs' own names repeat on each plane
    roifile.roiwrite(roiset, rois, name=[f"{r.position:04d}-{i:04d}" for i, r in enumerate(rois)])
    stacks = []
    if hires:
        hl, hg = sample(fly, HIRES_SHAPE, HIRES_SPACING, rng.uniform(-5, 5, 3))
        stack = fluorescence(hl, hg, HIRES_SPACING, rng, 0.03)
        write_stack(rdir / "anat", stack[..., ::-1] if num in RIGHT_ALS else stack, rng)
        stacks.append({"name": "anat", "dir": rel(rdir / "anat"), "experiment_xml": rel(rdir / "anat" / "Experiment.xml"),
                       "orientation": "as recorded"})
    return {"fly": f"{DATE}/{num}", "date": DATE, "fly_num": num, "datasets": [{"id": 1, "title": "synthetic", "uploader": USER}],
            "dir": rel(adir), "raw_dir": rel(rdir), "shape": list(shape), "average": rel(adir / "mocorr_concat_avg.tif"),
            "average_from": "al_analysis", "roiset": rel(roiset), "recordings": recordings, "stacks": stacks, "failed": {},
            "_spacing": spacing, "_n_rois": len(rois) - 1}


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
    flies = []
    for i in range(n):
        f = write_fly(out, i + 1, 100 + i, hires=i % 2 == 0)
        print(f"{USER}_{DATE}_{i + 1}", f.pop("_n_rois"), "ROIs, voxels (z, y, x)", f.pop("_spacing"), "um",
              "+ anatomical stack" + (" (right AL)" if i + 1 in RIGHT_ALS else "") if f["stacks"] else "")
        flies.append(f)
    (out / "roi_training.json").write_text(json.dumps({"format": "olfdb ROI training export", "version": 1, "stacks": True,
                                                       "glomeruli": NAMES, "flies": flies, "skipped": []}, indent=1))
    write_reference(out / "reference")
    print(out / "reference", "(reference atlas)")
