"""Flies from an olfdb (hong-data) ROI training export, as ``run_pipeline.load_fly`` gives them.

The export (the database's AL Segmentation tab) is a folder with ``roi_training.json`` and every
file where it is on tensor, as al_analysis leaves it there. Per fly:

* each diagnostic recording's ``trialmean_dff.tif`` (odor, Z, Y, X: mean dF/F of each odor)
  and ``trial_frames_and_odors.json``,
* its ``RoiSet.zip`` (drawn on its recordings registered together, so on the same grid),
  of which only glomeruli are trained on, as the database tells them apart (`roi_kind`),
* its average image, ``mocorr_concat_avg.tif`` (Z, Y, X): the mean of its motion-corrected movie,
* the ``Experiment.xml`` of its recordings in its raw data (voxel sizes),
* optionally its anatomical stacks (``anat/``), as ThorImage wrote them (``ChanA_*.tif``,
  one per frame of each plane, or a ``.raw``), not mirrored as al_analysis mirrors right ALs'
  functional data left-right (so `alseg.atlas.orient_hires` finds out which way they go).

The manifest names each recording's odors as in the database's odor table, so the same odor
has one name in every fly. A fly's responses are its recordings' ``trialmean_dff.tif`` one
after another, and `align_odors` puts every fly's in one order.

Paths in the manifest are from the folder it is in, or absolute (as the database writes
them for a fit it runs itself, reading the files where they are on tensor), and
`load_export` takes the folder or the manifest itself.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

from . import metadata, roi_io

MANIFEST = "roi_training.json"
# ThorImage's TIFF output: ChanA_<region>_<?>_<plane>_<frame>.tif, numbered from 1
_CHAN_A = re.compile(r"ChanA_(\d+)_(\d+)_(\d+)_(\d+)\.tiff?")
# a mark of a glomerulus name not certain (DA1_t0), as the database (hong-data) tells them apart
TENTATIVE = re.compile(r"_t\d+$")


def manifest_path(path) -> Path:
    """The manifest of an export: ``path`` itself if it is a file, else ``path``/`MANIFEST`."""
    path = Path(path)
    return path if path.is_file() else path / MANIFEST


def is_export(d: Path) -> bool:
    return manifest_path(d).is_file()


def fly_name(entry: dict) -> str:
    """A fly's name (for --val and output files): the top folder of its data on tensor
    (whose it is), its date and number, e.g. Tom_2023-11-19_1."""
    return f"{entry['dir'].split('/')[0]}_{entry['date']}_{entry['fly_num']}"


def manifest(root: Path) -> dict:
    return json.loads(manifest_path(root).read_text())


def fly_entries(root: Path) -> list[dict]:
    """The manifest's flies whose files all came (others are listed, and left out), each
    with its ``name`` (`fly_name`, made unique with -2, -3, ... where two would share one)."""
    m = manifest(root)
    out, seen = [], {}
    for f in m["flies"]:
        if f.get("failed"):
            print(f"  {fly_name(f)} left out: {len(f['failed'])} of its files could not be exported "
                  f"({'; '.join(f'{k}: {v}' for k, v in list(f['failed'].items())[:2])})")
            continue
        name = fly_name(f)
        seen[name] = seen.get(name, 0) + 1
        out.append({**f, "name": name if seen[name] == 1 else f"{name}-{seen[name]}"})
    return out


def roi_kind(name: str, glomeruli: set[str]) -> str:
    """An ROI's name as the database (hong-data's ``movies.roi_kind``) takes it: 'glomerulus'
    if it is one of the ``glomeruli`` it accepts (the export's) or several of them joined by
    / (that could not be told apart); 'tentative' if it is but for a ``_t<n>`` suffix (a name
    not certain); 'other' if not (numbered ROIs, names ending in + or ?, among others)."""
    def named(n: str) -> bool:
        return n in glomeruli or all(p in glomeruli for p in n.split("/"))

    if named(name):
        return "glomerulus"
    bare = TENTATIVE.sub("", name)
    return "tentative" if bare != name and named(bare) else "other"


def roi_keeper(glomeruli: set[str], tentative: bool = True):
    """Which ROIs of a RoiSet to train on (`roi_io.load_roiset`'s ``keep``): glomeruli, and
    (unless not ``tentative``) tentative ones, as that glomerulus, but for those of a
    glomerulus the RoiSet also has an ROI of without the mark (that one is it). Numbered ROIs,
    and others, are left out."""
    def keep(name: str, names: set[str]) -> bool:
        kind = roi_kind(name, glomeruli)
        if kind == "tentative":
            return tentative and TENTATIVE.sub("", name) not in names
        return kind == "glomerulus"

    return keep


def finite(x: np.ndarray, what) -> np.ndarray:
    """``x`` with values that are not finite (NaN, inf) set to 0: al_analysis' dF/F has them at
    the odd voxel whose baseline is 0, and one breaks the PCA, correlation and refinement.
    Says how many there were."""
    bad = ~np.isfinite(x)
    if bad.any():
        print(f"  {what}: {int(bad.sum())} values not finite (NaN or inf), taken as 0")
        x = np.where(bad, 0, x).astype(x.dtype)
    return x


def read_stack(d: Path) -> np.ndarray:
    """A ThorImage anatomical stack: (Z, Y, X) float32, the mean of the frames of each plane.
    From its ``ChanA_*.tif`` files, else its ``.raw`` (through hong2p)."""
    planes = defaultdict(list)
    for p in Path(d).iterdir():
        m = _CHAN_A.fullmatch(p.name)
        if m:
            planes[int(m.group(3))].append(p)
    if planes:
        # each file's own image: the first one's OME metadata describes the whole series
        return np.stack([np.mean([tifffile.imread(p, is_ome=False).astype(np.float32) for p in sorted(planes[z])], 0)
                         for z in sorted(planes)]).astype(np.float32)
    from hong2p import thor

    movie = np.asarray(thor.read_movie(d), np.float32)
    return movie.mean(0) if movie.ndim == 4 else movie


def load(root: Path, entry: dict, name_table=None, keep=None) -> dict:
    """One fly of the export (see the module's docstring), with ``odors``: the name of each
    of its response channels. ``keep``: which ROIs to train on (`roi_keeper`)."""
    root = Path(root)
    avg = finite(tifffile.imread(root / entry["average"]).astype(np.float32), entry["average"])
    resp, odors = [], []
    for r in entry["recordings"]:
        x = finite(tifffile.imread(root / r["responses"]).astype(np.float32), r["responses"])
        x = x[None] if x.ndim == 3 else x
        assert x.shape[0] == len(r["odors"]) and x.shape[1:] == avg.shape, (
            f"{r['responses']}: {x.shape}, but {len(r['odors'])} odors and average image {avg.shape}")
        resp.append(x)
        odors += r["odors"]
    labels, lut, overlap = roi_io.load_roiset(root / entry["roiset"], avg.shape, name_table, keep=keep)
    xml = next(root / r["experiment_xml"] for r in entry["recordings"] if r.get("experiment_xml"))
    f = {"name": entry.get("name") or fly_name(entry), "average": avg, "responses": np.concatenate(resp), "odors": odors,
         "labels": labels, "lut": lut, "overlap": overlap, "spacing": metadata.thorimage_spacing(xml, avg.shape)}
    if entry.get("stacks"):
        s = entry["stacks"][0]
        h = read_stack(root / s["dir"])
        # mirror: None until known (`alseg.atlas.orient_hires`), as the stack is as recorded
        f["hires"] = {"anatomy": h, "spacing": metadata.thorimage_spacing(root / s["experiment_xml"], h.shape),
                      "mirror": None}
    return f


def align_odors(flies: list[dict]) -> list[str]:
    """Gives every fly's ``responses`` the same odors in the same order: those of every fly
    (in the first fly's order; a fly's channels of one odor averaged). Returns them."""
    common = [o for o in dict.fromkeys(flies[0]["odors"]) if all(o in f["odors"] for f in flies[1:])]
    for f in flies:
        idx = defaultdict(list)
        for i, o in enumerate(f["odors"]):
            idx[o].append(i)
        dropped = [o for o in dict.fromkeys(f["odors"]) if o not in common]
        if dropped:
            print(f"  {f['name']}: {len(dropped)} odors not in every fly left out ({', '.join(dropped)})")
        f["responses"] = np.stack([f["responses"][idx[o]].mean(0) for o in common]).astype(np.float32)
        f["odors"] = common
    return common


def load_export(path, tentative: bool = True, progress=None) -> dict:
    """Every fly of an export (`fly_entries`, `load`), their odors in one order
    (`align_odors`), and what a model fit to them keeps of the export.

    Args:
        path: the export's folder, or its manifest.
        tentative: train on tentative ROIs (``DA1_t0``) as that glomerulus (`roi_keeper`).
        progress: called as ``progress("load", flies loaded, flies)``.

    Returns:
        dict with ``flies``; ``odors`` (the ones of every fly, in their order);
        ``aliases``: each odor's name as recorded (in ``trial_frames_and_odors.json``) ->
        as the database names it, so a new fly's odors can be named alike; ``name_table``
        (glomerulus -> label, shared by every fly); ``glomeruli``: the names the database
        accepts (the export's).
    """
    path = Path(path)
    root = manifest_path(path).parent
    m = manifest(path)
    entries = fly_entries(path)
    keep = roi_keeper(set(m["glomeruli"]), tentative=tentative)
    name_table = roi_io.build_name_table([root / e["roiset"] for e in entries], [tuple(e["shape"]) for e in entries],
                                         keep=keep)
    flies = []
    for i, e in enumerate(entries):
        flies.append(load(root, e, name_table, keep))
        if progress:
            progress("load", i + 1, len(entries))
    odors = align_odors(flies)
    aliases = {}
    for e in entries:
        for r in e["recordings"]:
            aliases.update(zip(r.get("as_recorded") or r["odors"], r["odors"]))
    return {"flies": flies, "odors": odors, "aliases": aliases, "name_table": name_table, "glomeruli": m["glomeruli"]}
