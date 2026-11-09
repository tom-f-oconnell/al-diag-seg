"""A fitted model, saved to and loaded from a folder, run on flies and scored against their ROIs.

A model is everything a new fly is segmented and named with: the fine-tuned Cellpose
weights (and the pixel size they are at), the PCA basis of the input channels, the
identity template (each glomerulus's tuning, position and shape), the atlas (if built),
the odors (in the order a fly's responses have to be in), and how each odor was named as
recorded, so a new fly's odors can be matched by name.

The folder (`Model.save`; `FORMAT` says its layout)::

    model.json      what it is: options, odors, glomeruli, training flies, versions
    segmenter       Cellpose weights (absent for the stand-in segmenter, see `Options`)
    pca.npz         mean and components
    template.json   the identity template
    atlas.npz       template image, probability maps, names, voxel size (if built)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from . import atlas as atlas_mod
from . import identity, metrics, refine, reference, responses, roi_io, segment

FORMAT = 1
STAGES = ("load", "train", "atlas", "evaluate")
# a predicted ROI is that manual ROI if they overlap this much (IoU), for naming it
NAME_IOU = 0.3
# and detects it if this much
DETECT_IOU = 0.5


@dataclass
class Options:
    """How a model is fit (`fit`). ``segmenter``: "cellpose", or "stand-in" (for tests, without a
    GPU or Cellpose's weights: trains nothing, and "predicts" a fly's own ROIs grown by 2
    pixels, so only works on flies with ROIs). ``final``: the ROIs a fly's are (`Model.final_kind`):
    "pred", Cellpose's own, named (with the atlas, if there is one); or "refined", refined by odor
    tuning and the atlas (on real flies, refinement shrank and broke up Cellpose's ROIs: see the
    README)."""
    epochs: int = 200
    erode: int = 1
    xy_um: float | None = None
    atlas: bool = True
    reg: str = "SyN"
    hires_reg: str = "Rigid"
    atlas_um: float = 1.0
    w_prior: float = 0.5
    refs: list[str] = field(default_factory=list)
    ref_side: str | None = None
    ref_reg: str = "SyN"
    ref_weight: float = 1.0
    tentative: bool = True
    gpu: bool | None = None  # None: if PyTorch sees one
    segmenter: str = "cellpose"
    batch_size: int = 8  # Cellpose's training batches to start with, halved while one does not fit in the GPU
    final: str = "pred"
    # how Cellpose's masks are made (`segment.predict`), also for models saved before these were
    # options (they then take these defaults): picked on the database's flies (see the README)
    cellprob_threshold: float = -1.0
    flow_threshold: float = 0.4
    dilate: int = 1  # (grown back by `erode`)


def use_gpu(gpu: bool | None) -> bool:
    if gpu is not None:
        return gpu
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def version() -> dict:
    """This package's version, and the commit it is of (from a pip install from git, or a checkout)."""
    from importlib import metadata

    out = {"version": None, "commit": None}
    try:
        dist = metadata.distribution("al-diag-seg")
        out["version"] = dist.version
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        out["commit"] = (direct.get("vcs_info") or {}).get("commit_id")
    except metadata.PackageNotFoundError:
        pass
    if out["commit"] is None:
        import subprocess

        try:
            out["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                                           text=True, timeout=10, check=True).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def write_volume(path, a: np.ndarray) -> None:
    """A (Z, Y, X) volume as an ImageJ stack of planes (as al_analysis' TIFFs are), so that
    one of 3 or 4 planes is not taken for a colour image."""
    tifffile.imwrite(path, a, imagej=True, metadata={"axes": "ZYX"})


def _stand_in(fly) -> np.ndarray:
    from skimage.segmentation import expand_labels

    if fly.get("labels") is None:
        raise ValueError(f"{fly.get('name')}: the stand-in segmenter needs the fly's own ROIs")
    return np.stack([expand_labels(s, 2) for s in fly["labels"]])


@dataclass
class Model:
    segmenter: str  # path of the Cellpose weights, or "stand-in"
    xy_um: float
    pca: responses.SharedPCA
    template: dict
    atlas: atlas_mod.Atlas | None
    odors: list[str]
    aliases: dict[str, str]
    glomeruli: list[str]  # those it was trained on
    options: Options
    info: dict = field(default_factory=dict)

    # --------------------------------------------------------------- folder
    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        seg = self.segmenter
        if seg != "stand-in" and Path(seg).is_file():
            if Path(seg).resolve() != (folder / "segmenter").resolve():
                shutil.copyfile(seg, folder / "segmenter")
            seg = "segmenter"
        np.savez(folder / "pca.npz", mean=self.pca.mean_, components=self.pca.components_)
        tmpl = {n: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in g.items()} for n, g in self.template.items()}
        (folder / "template.json").write_text(json.dumps(tmpl))
        if self.atlas is not None:
            a = self.atlas
            np.savez_compressed(folder / "atlas.npz", template=a.template, prob=a.prob, names=np.array(a.names),
                                spacing=np.array(a.spacing),
                                counts=a.counts if a.counts is not None else np.ones(len(a.names)))
        (folder / "model.json").write_text(json.dumps({
            "format": FORMAT, "segmenter": seg, "xy_um": self.xy_um, "atlas": self.atlas is not None,
            "odors": self.odors, "aliases": self.aliases, "glomeruli": self.glomeruli,
            "options": asdict(self.options), "info": self.info,
        }, indent=1))
        if seg == "segmenter":
            self.segmenter = str(folder / "segmenter")
        return folder

    @classmethod
    def load(cls, folder) -> "Model":
        folder = Path(folder)
        m = json.loads((folder / "model.json").read_text())
        if m.get("format") != FORMAT:
            raise ValueError(f"{folder}: a model of format {m.get('format')}; this al-diag-seg reads format {FORMAT}")
        p = np.load(folder / "pca.npz")
        pca = responses.SharedPCA(len(p["components"]))
        pca.mean_, pca.components_ = p["mean"], p["components"]
        tmpl = {n: {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in g.items()}
                for n, g in json.loads((folder / "template.json").read_text()).items()}
        atl = None
        if m["atlas"]:
            a = np.load(folder / "atlas.npz")
            atl = atlas_mod.Atlas(a["template"], a["prob"], [str(n) for n in a["names"]], tuple(float(x) for x in a["spacing"]),
                                  a["counts"])
        o = Options(**{k: v for k, v in m["options"].items() if k in Options.__dataclass_fields__})
        seg = "stand-in" if m["segmenter"] == "stand-in" else str(folder / m["segmenter"])
        return cls(seg, m["xy_um"], pca, tmpl, atl, m["odors"], m["aliases"], m["glomeruli"], o, m.get("info", {}))

    # ------------------------------------------------------------- predict
    def predict(self, fly) -> dict:
        """Segments and names a fly: ``fly`` has ``average``, ``responses`` (in `odors`' order),
        ``spacing`` and optionally ``hires`` (its anatomical stack, for the atlas).

        Returns:
            dict with ``labels``: each kind of ROIs ('pred': Cellpose's; 'refined_tuning';
            'refined_prior', with the atlas), ``identities``: for each, {label: (name or
            None, cost, confidence)}, ``final``: the kind to use, ``atlas_maxprob``: the
            atlas on the fly's grid (its largest probability), or None.
        """
        o, resp, sp = self.options, fly["responses"], fly["spacing"]
        if self.segmenter == "stand-in":
            pred = _stand_in(fly)
        else:
            channels = responses.build_channels(fly["average"], resp, self.pca)
            pred = segment.predict(self.segmenter, channels, gpu=use_gpu(o.gpu), spacing=sp, xy_um=self.xy_um,
                                   cellprob_threshold=o.cellprob_threshold, flow_threshold=o.flow_threshold,
                                   dilate=o.dilate)
        labels = {"pred": pred, "refined_tuning": refine.refine_by_tuning(resp, pred, sp)}
        ids = {"refined_tuning": identity.assign_identities(resp, labels["refined_tuning"], self.template, sp,
                                                            w_shape=0, w_layout=0)}
        maxprob = None
        if self.atlas is not None:
            if fly.get("hires") and fly["hires"].get("mirror") is None:
                atlas_mod.orient_hires(fly, o.hires_reg)
            prior = atlas_mod.atlas_to_fly(self.atlas, fly, transform=o.reg, hires_transform=o.hires_reg)
            # name the raw prediction first so the prior knows which glomerulus each ROI is
            ids["pred"] = identity.assign_identities(resp, pred, self.template, sp, prior, self.atlas.names)
            names0 = {l: v[0] for l, v in ids["pred"].items()}
            labels["refined_prior"] = refine.refine_with_prior(resp, pred, names0, prior, self.atlas.names, sp,
                                                               w_prior=o.w_prior)
            ids["refined_prior"] = identity.assign_identities(resp, labels["refined_prior"], self.template, sp, prior,
                                                              self.atlas.names)
            maxprob = prior.max(0).astype(np.float32)
        else:
            ids["pred"] = identity.assign_identities(resp, pred, self.template, sp, w_shape=0, w_layout=0)
        return {"labels": labels, "identities": ids, "atlas_maxprob": maxprob,
                "final": self.final_kind()}

    def final_kind(self) -> str:
        """The kind of ROIs a fly's are (`Options.final`): 'pred' (Cellpose's), else the refined ones:
        'refined_prior' with the atlas, 'refined_tuning' without."""
        if self.options.final == "pred":
            return "pred"
        return "refined_prior" if self.atlas is not None else "refined_tuning"

    def evaluate(self, flies, out=None, progress=None) -> dict:
        """Predicts each fly (with ROIs) and scores each kind of predicted ROIs against its own.

        Writes to ``out`` (if given), per fly: each kind's labels (``<fly>_<kind>.tif``) and
        RoiSet.zip, named (``<fly>_<kind>_RoiSet.zip``), its identities
        (``<fly>_identities_<kind>.csv``), the atlas on its grid; and ``summary.csv``,
        ``glomeruli.csv``.

        Returns:
            dict with ``summary`` (a row per fly and kind of ROIs: recall, precision and
            mean Dice of matched ROIs, as 3D objects and (``*_2d``) plane by plane
            (`metrics.match_report_2d`), splits, merges, crosstalk; and of its manual ROIs,
            their crosstalk, to compare with), ``identity`` (per fly and
            kind: ROIs named as the manual one they overlap, of those that overlap one),
            ``glomeruli`` (per glomerulus, over the flies: how many have it, in how many
            the final ROIs detect it, mean Dice, named right) and ``final``.
        """
        out = Path(out) if out is not None else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
        rows, id_rows, per_glom = [], [], {}
        final = None
        for i, f in enumerate(flies):
            p = self.predict(f)
            final = p["final"]
            lut = f["lut"]
            if out:
                if p["atlas_maxprob"] is not None:
                    write_volume(out / f"{f['name']}_atlas_maxprob.tif", p["atlas_maxprob"])
            ct = refine.crosstalk_report(f["responses"], f["labels"], f["spacing"])
            rows.append({"fly": f["name"], "rois": "manual", "n": len(np.unique(f["labels"])) - 1, "recall": "", "precision": "",
                         "dice": "", "splits": "", "merges": "", "n_2d": int(sum(len(np.unique(z)) - 1 for z in f["labels"])),
                         "recall_2d": "", "precision_2d": "", "dice_2d": "",
                         "coherence": np.mean(list(ct["coherence"].values())) if ct["coherence"] else "",
                         "neighbor_corr": np.mean(list(ct["neighbor_corr"].values())) if ct["neighbor_corr"] else "",
                         "leak": np.mean(np.abs(list(ct["leak"].values()))) if ct["leak"] else ""})
            for kind, lab in p["labels"].items():
                names = {l: v[0] for l, v in p["identities"][kind].items()}
                if out:
                    write_volume(out / f"{f['name']}_{kind}.tif", lab.astype(np.uint16))
                    roi_io.write_roiset(lab, names, out / f"{f['name']}_{kind}_RoiSet.zip")
                geo = metrics.match_report(f["labels"], lab)
                g2 = metrics.match_report_2d(f["labels"], lab)
                ct = refine.crosstalk_report(f["responses"], lab, f["spacing"])
                rows.append({
                    "fly": f["name"], "rois": kind, "n": len(np.unique(lab)) - 1,
                    "recall": geo.get("recall", ""), "precision": geo.get("precision", ""),
                    "dice": geo.get("mean_dice_matched", ""), "splits": geo.get("splits", ""), "merges": geo.get("merges", ""),
                    "n_2d": g2["n_pred"], "recall_2d": g2["recall"], "precision_2d": g2["precision"],
                    "dice_2d": g2["mean_dice_matched"],
                    "coherence": np.mean(list(ct["coherence"].values())) if ct["coherence"] else "",
                    "neighbor_corr": np.mean(list(ct["neighbor_corr"].values())) if ct["neighbor_corr"] else "",
                    "leak": np.mean(np.abs(list(ct["leak"].values()))) if ct["leak"] else "",
                })
                iou, g_ids, p_ids = metrics.iou_matrix(f["labels"], lab)
                correct = total = 0
                id_lines = []
                for j, pl in enumerate(p_ids):
                    manual = lut.get(int(g_ids[iou[:, j].argmax()])) if iou.size and iou[:, j].max() > NAME_IOU else None
                    name, cost, conf = p["identities"][kind].get(int(pl), (None, np.nan, np.nan))
                    id_lines.append([int(pl), name, manual, f"{cost:.3f}", f"{conf:.3f}"])
                    if manual:
                        total += 1
                        correct += name == manual
                id_rows.append({"fly": f["name"], "rois": kind, "correct": correct, "total": total})
                if out:
                    with open(out / f"{f['name']}_identities_{kind}.csv", "w", newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow(["label", "assigned", "manual", "cost", "confidence"])
                        w.writerows(id_lines)
                if kind == final:
                    named = {int(pl): p["identities"][kind].get(int(pl), (None,))[0] for pl in p_ids}
                    for gi, g in enumerate(g_ids):
                        name = lut.get(int(g))
                        if name is None:
                            continue
                        best = int(iou[gi].argmax()) if iou.shape[1] else -1
                        b = float(iou[gi, best]) if best >= 0 else 0.0
                        r = per_glom.setdefault(name, {"glomerulus": name, "flies": 0, "detected": 0, "dice": [], "named": 0})
                        r["flies"] += 1
                        if b >= DETECT_IOU:
                            r["detected"] += 1
                            r["dice"].append(2 * b / (1 + b))
                        if b > NAME_IOU and named.get(int(p_ids[best])) == name:
                            r["named"] += 1
            if progress:
                progress("evaluate", i + 1, len(flies))
        glom = [{**r, "dice": float(np.mean(r["dice"])) if r["dice"] else None} for r in sorted(per_glom.values(), key=lambda r: r["glomerulus"])]
        if out and rows:
            with open(out / "summary.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            with open(out / "glomeruli.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["glomerulus", "flies", "detected", "dice", "named"])
                w.writeheader()
                w.writerows(glom)
        clean = lambda v: float(v) if isinstance(v, (float, np.floating)) else v  # noqa: E731
        return {"summary": [{k: clean(v) for k, v in r.items()} for r in rows], "identity": id_rows, "glomeruli": glom,
                "final": final}


def fit(flies, val=(), options: Options | None = None, out=None, progress=None, odors=None, aliases=None,
        glomeruli=None, weights=None) -> Model:
    """Fits a model to the flies not in ``val`` (names); ``val`` flies are Cellpose's
    validation set only (their losses), and are never in the PCA basis, template or atlas.

    Args:
        flies: as `olfdb.load_export` gives them (responses in one order of odors).
        out: where Cellpose writes its weights (``out``/models).
        progress: called as ``progress(stage, done, total)``, stage one of `STAGES`.
        odors, aliases, glomeruli: of the export (`olfdb.load_export`), kept with the model.
        weights: Cellpose weights already trained, to use instead of training.
    """
    o = options or Options()
    t0 = time.time()
    train_f = [f for f in flies if f["name"] not in set(val)]
    val_f = [f for f in flies if f["name"] in set(val)]
    if not train_f:
        raise ValueError("no flies to train on")
    durations = {}
    # shared PCA basis on training flies only (inside the AL, i.e. labeled voxels)
    pca = responses.SharedPCA(3).fit([f["responses"] for f in train_f], masks=[f["labels"] > 0 for f in train_f])
    for f in flies:
        f["channels"] = responses.build_channels(f["average"], f["responses"], pca)
    xy_um = o.xy_um or float(np.median([f["spacing"][2] for f in flies]))
    print(f"segmenting at {xy_um:.3f} um pixels")
    t = time.time()
    if progress:
        progress("train", 0, o.epochs)
    if o.segmenter == "stand-in":
        seg = "stand-in"
    elif weights is not None:
        seg = str(weights)
    else:
        seg, _, _ = segment.train(train_f, val_f, out or ".", n_epochs=o.epochs, erode=o.erode, gpu=use_gpu(o.gpu),
                                  xy_um=xy_um, progress=progress, batch_size=o.batch_size)
    durations["train"] = time.time() - t
    template = identity.build_template(train_f)
    atl = None
    t = time.time()
    if o.atlas:
        stacks = [f for f in flies if f.get("hires") and f["hires"]["mirror"] is None]
        total = len(stacks) + len(train_f) + len(o.refs)
        if progress:
            progress("atlas", 0, total)
        for i, f in enumerate(stacks):
            mirrored, scores = atlas_mod.orient_hires(f, o.hires_reg)
            print(f"  {f['name']}: anatomical stack {'mirrored left-right' if mirrored else 'as recorded'} "
                  f"(fit as recorded {scores[0]:.3f}, mirrored {scores[1]:.3f}; lower is better)")
            if progress:
                progress("atlas", i + 1, total)
        print("building atlas ...")
        atl = atlas_mod.build_atlas(train_f, transform=o.reg, hires_transform=o.hires_reg, atlas_um=o.atlas_um,
                                    progress=(lambda n: progress("atlas", len(stacks) + n, total)) if progress else None)
        print(f"  atlas grid (z, y, x) {atl.template.shape} at {np.round(atl.spacing, 3).tolist()} um")
        for k, r in enumerate(o.refs):
            atl, info = reference.add_best_side(atl, r, o.ref_side, transform=o.ref_reg, weight=o.ref_weight)
            print(f"reference {r} ({info['side'] or 'one side'}{', mirrored' if info['mirrored'] else ''}, "
                  f"{info['fit']} fit): {info['n_shared']} glomeruli shared, centroid residual median "
                  f"{info['residual_median_um']:.1f} um, max {info['residual_max_um']:.1f} um"
                  + (f", left out of fit: {' '.join(info['rejected'])}" if info["rejected"] else "")
                  + f"; added {len(info['added'])}: {' '.join(info['added'])}")
            if glomeruli:
                foreign = [n for n in info["added"] if not (n in glomeruli or all(p in glomeruli for p in n.split("/")))]
                if foreign:
                    print(f"  of those, not glomerulus names the database accepts (ROIs named so would not be glomeruli "
                          f"to it): {' '.join(foreign)}")
            if progress:
                progress("atlas", len(stacks) + len(train_f) + k + 1, total)
    durations["atlas"] = time.time() - t
    names = sorted({n for f in train_f for n in f["lut"].values()})
    info = {"created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "alseg": version(),
            "accepted_glomeruli": list(glomeruli) if glomeruli else None,
            "train_flies": [f["name"] for f in train_f], "val_flies": [f["name"] for f in val_f],
            "seconds": {**durations, "fit": time.time() - t0}}
    return Model(seg, xy_um, pca, template, atl, list(odors or train_f[0]["odors"]), dict(aliases or {}), names, o, info)
