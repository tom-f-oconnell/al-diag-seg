"""A fly from its folders as al_analysis leaves them (not a database export), to run a model on.

From its motion-corrected data folder (``<...>/<date>/<fly>/``):

* its diagnostic recordings: the subfolders with ``trialmean_dff.tif`` (odor, Z, Y, X) and
  ``trial_frames_and_odors.json``, of which more than half of the odors are the model's;
* its average image, ``mocorr_concat_avg.tif``, else the mean of ``mocorr_concat.tif``;
* its ROIs, if it has them (``RoiSet.zip``, in the folder or a recording's), to score the
  model against;

and from its raw data folder (optional), the ``Experiment.xml`` of a diagnostic recording
(its voxel size, unless given) and its anatomical stack (``anat``), as ThorImage wrote them.

Odors are matched to the model's by their names as recorded (``odors`` in
``trial_frames_and_odors.json``), as the training flies had them (`Model.aliases`), else
by the names themselves.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import tifffile

from . import metadata, olfdb, roi_io

TRIALMEAN = "trialmean_dff.tif"
TRIALS = "trial_frames_and_odors.json"
ROISET = "RoiSet.zip"
CONCAT, CONCAT_AVG = "mocorr_concat.tif", "mocorr_concat_avg.tif"
ANATOMICAL = re.compile(r"(?:anat|anatomical)(?:_.*)?")  # as the database (hong-data) finds them


def recorded_odors(path: Path) -> list[str]:
    """A recording's odors as recorded, in the order al_analysis' ``trialmean_dff.tif`` has
    them (each one's first trial)."""
    trials = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(dict.fromkeys(t["odors"] for t in trials))


def _average(d: Path) -> np.ndarray:
    if (d / CONCAT_AVG).is_file():
        return tifffile.imread(d / CONCAT_AVG).astype(np.float32)
    if (d / CONCAT).is_file():
        movie = tifffile.memmap(d / CONCAT, mode="r") if tifffile.TiffFile(d / CONCAT).series[0].dataoffset else tifffile.imread(d / CONCAT)
        total = np.zeros(movie.shape[1:], np.float64)
        for i in range(0, movie.shape[0], 50):
            total += movie[i:i + 50].sum(0, dtype=np.float64)
        return (total / movie.shape[0]).astype(np.float32)
    raise FileNotFoundError(f"{d}: no {CONCAT_AVG} or {CONCAT} (its average image)")


def load_fly(mocorr_dir, model, raw_dir=None, spacing=None, name=None, glomeruli=None, tentative: bool = True) -> dict:
    """A fly (as `olfdb.load` gives them) with its responses in ``model``'s odors and order.

    Args:
        mocorr_dir: its motion-corrected data folder.
        model: the `alseg.model.Model` to run on it (its odors, and their names as recorded).
        raw_dir: its raw data folder (voxel size and anatomical stack), if any.
        spacing: (z, y, x) voxel size in um, if not from an ``Experiment.xml``.
        glomeruli: names the database accepts (its ROIs are scored as glomeruli by them);
            default the model's.

    Raises:
        ValueError: if it lacks some of the model's odors (listed), or has no voxel size.
    """
    d = Path(mocorr_dir)
    raw = Path(raw_dir) if raw_dir else None
    want = set(model.odors)
    canon = lambda s: model.aliases.get(s, s)  # noqa: E731
    resp, odors, recs = [], [], []
    for r in sorted(p for p in d.iterdir() if p.is_dir()):
        if not ((r / TRIALMEAN).is_file() and (r / TRIALS).is_file()):
            continue
        names = [canon(s) for s in recorded_odors(r / TRIALS)]
        if sum(n in want for n in names) <= len(names) / 2:
            continue
        x = olfdb.finite(tifffile.imread(r / TRIALMEAN).astype(np.float32), r / TRIALMEAN)
        x = x[None] if x.ndim == 3 else x
        if x.shape[0] != len(names):
            raise ValueError(f"{r}: {TRIALMEAN} has {x.shape[0]} odors, but {TRIALS} {len(names)}")
        resp.append(x)
        odors += names
        recs.append(r)
    missing = [o for o in model.odors if o not in odors]
    if missing:
        raise ValueError(f"{d}: none of its recordings has {len(missing)} of the model's odors: {', '.join(missing)}"
                         + ("" if recs else " (no recording with trialmean_dff.tif and mostly the model's odors)"))
    avg = olfdb.finite(_average(d), d / "average image")
    allr = np.concatenate(resp)
    if allr.shape[1:] != avg.shape:
        raise ValueError(f"{d}: responses {allr.shape[1:]}, but average image {avg.shape}")
    idx = {}
    for i, o in enumerate(odors):
        idx.setdefault(o, []).append(i)
    f = {"name": name or f"{d.parent.name}_{d.name}", "average": avg, "odors": list(model.odors),
         "responses": np.stack([allr[idx[o]].mean(0) for o in model.odors]).astype(np.float32)}
    xml = next((raw / r.name / "Experiment.xml" for r in recs if raw and (raw / r.name / "Experiment.xml").is_file()), None)
    if spacing is not None:
        f["spacing"] = tuple(float(s) for s in spacing)
    elif xml is not None:
        f["spacing"] = metadata.thorimage_spacing(xml, avg.shape)
    else:
        raise ValueError(f"{d}: no voxel size: give the raw data folder (an Experiment.xml of its recordings) or the spacing")
    stack = next((s for s in sorted(raw.iterdir()) if s.is_dir() and ANATOMICAL.fullmatch(s.name)
                  and (s / "Experiment.xml").is_file()), None) if raw else None
    if stack is not None and model.atlas is not None:
        h = olfdb.read_stack(stack)
        f["hires"] = {"anatomy": h, "spacing": metadata.thorimage_spacing(stack / "Experiment.xml", h.shape), "mirror": None}
    roiset = d / ROISET if (d / ROISET).is_file() else next((r / ROISET for r in recs if (r / ROISET).is_file()), None)
    if roiset is not None:
        names = set(glomeruli or model.info.get("accepted_glomeruli") or model.glomeruli)
        keep = olfdb.roi_keeper(names, tentative=tentative)
        labels, lut, overlap = roi_io.load_roiset(roiset, avg.shape, keep=keep)
        f.update(labels=labels, lut=lut, overlap=overlap)
    return f
