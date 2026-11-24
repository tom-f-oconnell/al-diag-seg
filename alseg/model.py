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

from . import align as align_mod
from . import atlas as atlas_mod
from . import clean, identity, metrics, refine, reference, responses, roi_io, segment

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
    README); or "clean", Cellpose's own after the steps of ``clean`` (`alseg.clean.STEPS`, each with
    its parameters, ``clean_params``: {step: {name: value}}; none: as "pred"). Each step alone is a
    kind of ROIs of its own too (``clean_<step>``), to compare them by."""
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
    # the odors Cellpose's two images of responses are made from (`responses.build_channels`), of the model's
    # (None: all of them). The model's other odors are then only in what ROIs are joined and named by, so a fly
    # with these odors alone is segmented, joined and named exactly as by a model fit, to the same flies, with
    # the odors that fly has: a model can take in an odor some flies lack without changing what it does for them
    image_odors: list[str] | None = None
    # how Cellpose's masks are made (`segment.predict`), also for models saved before these were
    # options (they then take these defaults): picked on the database's flies, scored as partly
    # drawn (see the README)
    cellprob_threshold: float = -2.0
    flow_threshold: float = 0.8
    dilate: int = 0
    # how the planes' masks are joined into 3D objects: "tuning" (`segment.join_planes`, with
    # join_overlap and join_corr) or "cellpose" (Cellpose's stitching, by IoU >= 0.25)
    join: str = "tuning"
    join_overlap: float = 0.2
    join_corr: float = 0.7
    # the most naming an ROI as a glomerulus may cost (`identity.assign_identities`): an ROI none
    # costs less for is left unnamed (numbered); also for models saved before it was an option
    max_cost: float = 0.6
    # the atlas on a fly's grid moved to where the fly's ROIs say its glomeruli are, and the ROIs named with it there
    # (`Model.align`): by an "affine" map (fit to the ROIs that odor tuning alone names; else to those named with the
    # atlas, after a shift), or by a "shift" (of those named with the atlas); None: left where registration put it.
    # Also for models saved before it was an option
    align: str | None = "affine"
    # masks over two glomeruli split by their tuning (`alseg.clean_trim.split`, with clean_params["split"]), and the
    # planes' masks joined again: the "split" kind, which the steps of ``clean`` then clean (`Model.refine`)
    split: bool = False
    clean: list[str] = field(default_factory=list)
    clean_params: dict = field(default_factory=dict)


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


def write_identity_terms(path, terms: dict) -> None:
    """What the costs of naming a fly's ROIs are made of (`identity.assign_identities`'s
    ``return_terms``), as JSON: each ROI by its label (in the fly's label .tif of that kind)."""
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return float(v) if np.isfinite(v) else None
        return v

    Path(path).write_text(json.dumps(clean(terms), indent=1))


def write_prior(path, prior: np.ndarray, names, alignment: dict | None = None) -> None:
    """A model's atlas on a fly's grid (`Model.predict`'s ``prior``: (glomeruli, Z, Y, X), each one's
    probability), with the glomeruli's ``names``, as a compressed .npz (``prior``, float16, and
    ``names``): what a fly's ROIs are named by, kept so that ROIs added later (`alseg.nmf`) are named
    without registering the atlas again. For an atlas moved to where the fly's ROIs say its glomeruli are, how
    (``alignment``, `Model.predict`'s, as JSON: `read_alignment` reads it)."""
    more = {} if alignment is None else {"alignment": np.array(json.dumps(alignment))}
    np.savez_compressed(path, prior=np.asarray(prior, np.float16), names=np.array(list(names)), **more)


def read_alignment(npz) -> dict | None:
    """How the atlas of a `write_prior` file (opened with ``np.load``) was moved to where its fly's ROIs say its
    glomeruli are (`Model.align`'s dict); None for one left where registration put it (or written before atlases
    were moved)."""
    return json.loads(str(npz["alignment"])) if "alignment" in npz.files else None


def _purity(responses, labels, spacing, ref=None) -> dict:
    """`refine.purity_report` as a row's columns: medians of ``gap_um``, ``edge_share`` and
    ``whole_share``, and ``touching``, the fraction of in-plane masks touching another ROI ('' where
    there is none to take them of)."""
    r = refine.purity_report(responses, labels, spacing, ref=ref)
    med = lambda v: float(np.median(v)) if len(v) else ""  # noqa: E731
    return {"gap_um": med(r["gap_um"]), "touching": float(np.mean(np.array(r["gap_um"]) <= 0)) if r["gap_um"] else "",
            "edge_share": med(r["edge_share"]), "whole_share": med(r["whole_share"])}


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
    def load(cls, folder, atlas: bool = True) -> "Model":
        """The model saved in ``folder``; without its atlas (which is most of what there is to read) if
        not ``atlas``: enough to name ROIs with an atlas already on a fly's grid (`alseg.nmf.name`)."""
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
        if m["atlas"] and atlas:
            a = np.load(folder / "atlas.npz")
            atl = atlas_mod.Atlas(a["template"], a["prob"], [str(n) for n in a["names"]], tuple(float(x) for x in a["spacing"]),
                                  a["counts"])
        o = Options(**{k: v for k, v in m["options"].items() if k in Options.__dataclass_fields__})
        seg = "stand-in" if m["segmenter"] == "stand-in" else str(folder / m["segmenter"])
        return cls(seg, m["xy_um"], pca, tmpl, atl, m["odors"], m["aliases"], m["glomeruli"], o, m.get("info", {}))

    # ------------------------------------------------------------- predict
    def image_odors(self) -> list[str]:
        """The odors Cellpose's images of responses are made from (`Options.image_odors`), in the model's order."""
        chosen = self.options.image_odors
        return list(self.odors) if chosen is None else [x for x in self.odors if x in set(chosen)]

    def fly_odors(self, fly) -> list[int]:
        """The places, among the model's odors, of those a fly's ``responses`` are of (its ``odors``; all of
        them for a fly that does not say)."""
        have = fly.get("odors")
        return list(range(len(self.odors))) if have is None else [self.odors.index(x) for x in have]

    def template_of(self, fly) -> dict:
        """The template a fly is named by: the model's, over the odors the fly has (`identity.template_for`)."""
        idx = self.fly_odors(fly)
        return self.template if len(idx) == len(self.odors) else identity.template_for(self.template, idx)

    def channels(self, fly) -> np.ndarray:
        """Cellpose's input images of a fly (`responses.build_channels`), from its responses to the model's
        image odors: those of them it has, if it lacks some."""
        idx, img = self.fly_odors(fly), [self.odors.index(x) for x in self.image_odors()]
        rows = [idx.index(i) for i in img if i in idx]
        if len(rows) < 2:
            raise ValueError(f"{fly.get('name')}: it has {len(rows)} of the {len(img)} odors the model's images are made from: at least 2 are needed")
        present = None if len(rows) == len(img) else [k for k, i in enumerate(img) if i in idx]
        return responses.build_channels(fly["average"], fly["responses"][rows], self.pca, present)

    def prior_of(self, fly):
        """The model's atlas on ``fly``'s grid (`atlas.atlas_to_fly`: (glomeruli, Z, Y, X), each one's
        probability, in the order of the atlas' names), registered through its anatomical stack
        (``hires``) if it has one; None for a model without an atlas."""
        if self.atlas is None:
            return None
        o = self.options
        if fly.get("hires") and fly["hires"].get("mirror") is None:
            atlas_mod.orient_hires(fly, o.hires_reg)
        return atlas_mod.atlas_to_fly(self.atlas, fly, transform=o.reg, hires_transform=o.hires_reg)

    def name(self, fly, labels, prior=None, prior_names=None, **kw):
        """Names a fly's ROIs (``labels``) as the model names those it predicts: with its atlas on the
        fly's grid (``prior``, of the glomeruli ``prior_names``: the atlas' by default), if given; else
        by tuning and position alone (no shape or layout terms). ``kw``: more of
        `identity.assign_identities`' arguments.

        Returns:
            ({label: (name or None, cost, confidence)}, what the costs are made of), as
            `identity.assign_identities` with ``return_terms``.
        """
        if prior is None:
            kw, args = {"w_shape": 0, "w_layout": 0, **kw}, ()
        else:
            args = (prior, list(prior_names) if prior_names is not None else self.atlas.names)
        return identity.assign_identities(fly["responses"], labels, self.template_of(fly), fly["spacing"], *args,
                                          max_cost=self.options.max_cost, return_terms=True, **kw)

    def named_by_tuning(self, fly, labels) -> dict:
        """{label: name or None}: each of a fly's ROIs as the glomerulus tuned most like it (one to one), where
        that costs `alseg.align.TUNING_COST` at most (1 - the correlation of their odor tunings), wherever the ROI
        is. Not what the ROIs are named: what an affine map of the atlas is fit to (`align`)."""
        ids = identity.assign_identities(fly["responses"], labels, self.template_of(fly), fly["spacing"], w_position=0,
                                         w_shape=0, w_layout=0, max_cost=align_mod.TUNING_COST)
        return {l: v[0] for l, v in ids.items()}

    def align(self, fly, labels, prior, identities, terms=None, prior_names=None):
        """The atlas on a fly's grid as registered (``prior``, of the glomeruli ``prior_names``: the atlas' by
        default) moved to where the fly's ROIs (``labels``) say its glomeruli are, and the ROIs named again with it
        there, as `Options.align` asks:

        - "affine": by an affine map fit to the ROIs that odor tuning alone names (`named_by_tuning`: wherever they
          are, so the map is not held back by where the atlas expected them), in a fly with
          `alseg.align.TUNING_ODORS` of the model's odors or more. With fewer odors, with too few such ROIs, or
          where that map is not believed (`alseg.align.fit`), as below.
        - else by a fit to the ROIs as named with the atlas (``identities``, at the costs ``terms``: `name`'s), in
          the steps of `alseg.align.STEPS`: a shift, and for "affine" then an affine map, each a fit of the atlas
          as registered to the ROIs as the step before named them.

        Returns:
            (the atlas moved, how, the ROIs' names with it, what they cost): how, `alseg.align.fit`'s dict of the
            last fit, with ``rois_named_by`` ("tuning", or "atlas": which ROIs it was fit to); or (``prior``, None,
            ``identities``, ``terms``) as they are if `Options.align` is None, or too few of the ROIs are named to
            fit the atlas to.
        """
        kind, names = self.options.align, list(prior_names) if prior_names is not None else self.atlas.names
        if not kind:
            return prior, None, identities, terms
        if kind == "affine" and len(self.fly_odors(fly)) >= align_mod.TUNING_ODORS:
            m, a = align_mod.align(labels, self.named_by_tuning(fly, labels), prior, names, fly["spacing"], kind)
            if a is not None and a["kind"] == kind:
                return (m, {**a, "rois_named_by": "tuning"}, *self.name(fly, labels, m, names))
        moved, how = prior, None
        for step in align_mod.STEPS[kind]:
            m, a = align_mod.align(labels, {l: v[0] for l, v in identities.items()}, prior, names, fly["spacing"], step)
            if a is None:
                break
            moved, how = m, {**a, "rois_named_by": "atlas"}
            identities, terms = self.name(fly, labels, moved, names)
        return moved, how, identities, terms

    def predict(self, fly) -> dict:
        """Segments and names a fly: ``fly`` has ``average``, ``responses`` (in `odors`' order; or of
        those of them it has, ``odors``, in that order), ``spacing`` and optionally ``hires`` (its
        anatomical stack, for the atlas).

        A fly that lacks some of the model's odors is joined, refined and named by those it has (its
        template: `template_of`); Cellpose's images are of the image odors (`Options.image_odors`),
        makeshift where it lacks some of those (`responses.build_channels`).

        Returns:
            dict with ``labels``: each kind of ROIs ('pred': Cellpose's; 'refined_tuning';
            'refined_prior', with the atlas; 'split' and 'clean', Cellpose's cleaned as
            `Options.split` and `Options.clean` ask (`refine`); and 'clean_<step>', each cleaning
            step alone on Cellpose's, split if that was asked for), ``identities``: for each,
            {label: (name or None, cost, confidence)}, ``identity_terms``: for each, what those
            costs are made of (`identity.assign_identities`; with ``alignment``, below, for the
            kinds named with the atlas), ``final``: the kind to use,
            ``atlas_maxprob``: the atlas on the fly's grid (its largest probability), or None,
            ``prior``: the atlas there, each glomerulus' probability ((glomeruli, Z, Y, X), in
            the order of the atlas' names), or None: as registration put it, then moved to where
            Cellpose's ROIs say the fly's glomeruli are (`align`), which they are named again
            by; ``alignment``: how it was moved (`align`'s dict; None if it was not:
            `Options.align` None, or too few ROIs to fit it to), ``cellprob``: Cellpose's cell probabilities
            ((Z, Y, X) on the fly's grid; None for the stand-in segmenter), and ``report``: what
            cleaning did to each ROI (`refine`'s; None if none was asked for).
        """
        o, resp, sp = self.options, fly["responses"], fly["spacing"]
        if self.segmenter == "stand-in":
            pred = _stand_in(fly)
        else:
            channels = self.channels(fly)
            out = segment.predict(self.segmenter, channels, gpu=use_gpu(o.gpu), spacing=sp, xy_um=self.xy_um,
                                  cellprob_threshold=o.cellprob_threshold, flow_threshold=o.flow_threshold,
                                  dilate=o.dilate, responses=resp if o.join == "tuning" else None,
                                  join_overlap=o.join_overlap, join_corr=o.join_corr, return_cellprob=True)
            # (Cellpose's cell probabilities, for `clean`'s core)
            pred, fly["cellprob"] = out if isinstance(out, tuple) else (out, None)
        labels = {"pred": pred, "refined_tuning": refine.refine_by_tuning(resp, pred, sp)}
        ids, terms = {}, {}

        def name(kind, *args):
            ids[kind], terms[kind] = self.name(fly, labels[kind], *args)

        name("refined_tuning")
        maxprob = prior = alignment = None
        if self.atlas is not None:
            prior = self.prior_of(fly)
            # name the raw prediction first so the prior knows which glomerulus each ROI is
            name("pred", prior)
            # and where its glomeruli are in this fly: the atlas moved there, and the ROIs named again
            prior, alignment, ids["pred"], terms["pred"] = self.align(fly, pred, prior, ids["pred"], terms["pred"])
            names0 = {l: v[0] for l, v in ids["pred"].items()}
            labels["refined_prior"] = refine.refine_with_prior(resp, pred, names0, prior, self.atlas.names, sp,
                                                               w_prior=o.w_prior)
            name("refined_prior", prior)
            maxprob = prior.max(0).astype(np.float32)
        else:
            name("pred")
        report, base = None, "pred"
        if o.split or o.clean:
            r = self.refine(fly, pred, ids["pred"], terms["pred"], prior)
            for kind in r["labels"]:
                labels[kind], ids[kind], terms[kind] = r["labels"][kind], r["identities"][kind], r["identity_terms"][kind]
            report, base = r["report"], "split" if o.split else "pred"
        # each cleaning step alone, on Cellpose's ROIs (split, if asked for); named as those are
        for st in clean.STEPS:
            if not clean.usable(st, fly):
                continue
            kind = f"{clean.KIND}_{st}"
            labels[kind] = clean.apply(labels[base], fly, [st], o.clean_params)
            left = set(np.unique(labels[kind]).tolist())
            ids[kind] = {l: v for l, v in ids[base].items() if l in left}
            terms[kind] = terms[base]
        if prior is not None:
            for kind in terms:
                if kind != "refined_tuning":  # (named without the atlas)
                    terms[kind]["alignment"] = alignment
        return {"labels": labels, "identities": ids, "identity_terms": terms, "atlas_maxprob": maxprob,
                "prior": prior, "alignment": alignment, "final": self.final_kind(), "cellprob": fly.get("cellprob"),
                "report": report}

    def refine(self, fly, pred, identities, terms=None, prior=None, prior_names=None, split=None, steps=None,
               params=None) -> dict:
        """Cellpose's ROIs of a fly (``pred``, named ``identities``, at the costs ``terms``: `predict`'s
        "pred") cleaned of their neighbours, as `Options.split`, `Options.clean` and `Options.clean_params`
        say, or ``split``, ``steps`` and ``params`` where given (``params`` over the options'):

        - ``split``: each in-plane mask over two glomeruli tuned unlike is cut in two
          (`alseg.clean_trim.split`, with ``params["split"]``), and the planes' masks are joined again,
          as the model joins them (`segment.rejoin`), and all named again (`name`: with the atlas on the
          fly's grid, ``prior``, of the glomeruli ``prior_names``, if given: as `predict` gave it, so
          moved onto the fly's glomeruli already, and not moved again). If none is cut, they are
          left as they are.
        - then ``steps`` (of `alseg.clean.STEPS`, applied in that order, each with ``params[step]``),
          which only remove pixels, so names are kept. Those that cannot run on the fly (core, without
          its ``cellprob``) are left out.

        `predict` does this as the options ask, and it can be done after, from what a prediction kept:
        its "pred" ROIs and their names, the atlas on the fly's grid (to name ROIs split) and
        Cellpose's cell probabilities (for core). The fly needs ``responses``, ``spacing`` and, for
        split and trim, its folder (``dir``), where its per-trial evidence is read from, once
        (`clean.ensure_evidence`).

        Returns:
            dict with ``labels``, ``identities`` and ``identity_terms`` of each kind made: "split"
            (if asked for) and "clean" (if steps were, even if none could run); ``final``: the last
            of them ("pred" if none); ``steps``: those done, in order ("split" first, if it was);
            and ``report``, by the final kind's labels: ``split`` ({label: {z: the label in ``pred``
            of the mask it is a part of}}, on each plane where it is a part of a mask cut in two)
            and ``trimmed`` ({label: {z: the share of its area there that trim kept}}, where it cut
            some).
        """
        from . import clean_trim

        o = self.options
        split = o.split if split is None else split
        steps = clean.ordered(o.clean if steps is None else steps)
        params = {**o.clean_params, **(params or {})}
        labels, ids, terms_of = {}, {}, {}
        base, base_ids, base_terms = pred, identities, terms
        done, cut_from, trimmed = [], {}, {}
        if split:
            clean.ensure_evidence(fly)
            cut, rep = clean_trim.split(pred, fly, return_report=True, **(params.get("split") or {}))
            if rep["split"]:
                base = segment.rejoin(cut, fly["responses"], o.join, o.join_overlap, o.join_corr)
                base_ids, base_terms = self.name(fly, base, prior, prior_names)
                if terms is not None and "alignment" in terms:  # (named with the atlas the model's own ROIs were)
                    base_terms["alignment"] = terms["alignment"]
                for lab, z in rep["split"]:
                    for part in np.unique(base[z][pred[z] == lab]):
                        if part:
                            cut_from.setdefault(int(part), {})[int(z)] = int(lab)
            labels["split"], ids["split"], terms_of["split"] = base, base_ids, base_terms
            done.append("split")
        final = "split" if split else "pred"
        if steps:
            run = [s for s in steps if clean.usable(s, fly)]
            reports = {}
            lab = clean.apply(base, fly, run, params, reports=reports)
            left = set(np.unique(lab).tolist())
            labels[clean.KIND], ids[clean.KIND], terms_of[clean.KIND] = lab, {l: v for l, v in base_ids.items() if l in left}, base_terms
            done += run
            for (l, z), frac in ((reports.get("trim") or {}).get("kept") or {}).items():
                if frac < 1 and l in left:
                    trimmed.setdefault(int(l), {})[int(z)] = float(frac)
            cut_from = {l: v for l, v in cut_from.items() if l in left}
            final = clean.KIND
        return {"labels": labels, "identities": ids, "identity_terms": terms_of, "final": final, "steps": done,
                "report": {"split": cut_from, "trimmed": trimmed}}

    def final_kind(self) -> str:
        """The kind of ROIs a fly's are (`Options.final`): 'pred' (Cellpose's; 'split', those with
        masks over two glomeruli split, if `Options.split`), 'clean' (those after the steps of
        `Options.clean`), else the refined ones: 'refined_prior' with the atlas, 'refined_tuning'
        without."""
        o = self.options
        if o.final == "pred" or (o.final == clean.KIND and not o.clean):
            return "split" if o.split else "pred"
        if o.final == clean.KIND:
            return clean.KIND
        return "refined_prior" if self.atlas is not None else "refined_tuning"

    def evaluate(self, flies, out=None, progress=None, partly_drawn: bool = True, priors: bool = False) -> dict:
        """Predicts each fly (with ROIs) and scores each kind of predicted ROIs against its own.

        Writes to ``out`` (if given), per fly: each kind's labels (``<fly>_<kind>.tif``) and
        RoiSet.zip, named (``<fly>_<kind>_RoiSet.zip``), its identities
        (``<fly>_identities_<kind>.csv``) and what their costs are made of
        (``<fly>_identity_terms_<kind>.json``), the atlas on its grid, as moved to where its ROIs
        say its glomeruli are (`predict`'s ``prior``: its largest probability; with ``priors``, each glomerulus' too, and
        how it was moved: ``<fly>_atlas_prior.npz``, `write_prior`); and ``summary.csv``,
        ``glomeruli.csv``.

        Returns:
            dict with ``summary`` (a row per fly and kind of ROIs: recall, precision and
            mean Dice of matched ROIs, as 3D objects and (``*_2d``) plane by plane
            (`metrics.match_report_2d`), splits, merges, crosstalk, how far its in-plane masks
            keep from each other and how much of a neighbour they carry (`_purity`), and how
            broadly those matching a manual one respond against it (``breadth``,
            ``breadth_manual``: `refine.breadth`); and of its manual ROIs, their crosstalk and
            purity, to compare with), ``identity`` (per fly and
            kind: ROIs named as the manual one they overlap, of those that overlap one),
            ``glomeruli`` (per glomerulus, over the flies: how many have it, in how many
            the final ROIs detect it, mean Dice, named right), ``alignment`` (per fly, by name:
            how its atlas was moved, `predict`'s), ``final`` and ``scoring``.

        ``partly_drawn`` (``scoring``: "partly drawn", else "strict"): as glomeruli are often
        drawn on some of their planes only (their clearest), a predicted ROI's voxels on planes
        where the manual one has no outline count against it only where they lie in another manual
        ROI (`metrics.iou_matrix_partly_drawn`), here and in naming, and precision is of the
        predicted ROIs matched or lying mostly in manual ROIs (``n_scored``, ``n_scored_2d``).
        """
        out = Path(out) if out is not None else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
        rows, id_rows, per_glom, moved = [], [], {}, {}
        final = None
        for i, f in enumerate(flies):
            p = self.predict(f)
            final = p["final"]
            lut = f["lut"]
            moved[f["name"]] = p["alignment"]
            if out:
                if p["atlas_maxprob"] is not None:
                    write_volume(out / f"{f['name']}_atlas_maxprob.tif", p["atlas_maxprob"])
                    if priors:
                        write_prior(out / f"{f['name']}_atlas_prior.npz", p["prior"], self.atlas.names, p["alignment"])
            rows.append(manual_row(f))
            # (cleaned ROIs keep the labels of those they were cleaned from: Cellpose's, split if that was asked for)
            base = p["labels"].get("split", p["labels"]["pred"])
            for kind, lab in p["labels"].items():
                names = {l: v[0] for l, v in p["identities"][kind].items()}
                if out:
                    write_volume(out / f"{f['name']}_{kind}.tif", lab.astype(np.uint16))
                    roi_io.write_roiset(lab, names, out / f"{f['name']}_{kind}_RoiSet.zip")
                    write_identity_terms(out / f"{f['name']}_identity_terms_{kind}.json", p["identity_terms"][kind])
                s = score(f, lab, p["identities"][kind], kind, base if kind.startswith(clean.KIND) else None, partly_drawn)
                rows.append(s["row"])
                id_rows.append({"fly": f["name"], "rois": kind, **s["identity"]})
                if out:
                    with open(out / f"{f['name']}_identities_{kind}.csv", "w", newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow(["label", "assigned", "manual", "cost", "confidence"])
                        w.writerows(s["lines"])
                if kind == final:
                    iou, g_ids, p_ids = s["match"]
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
        plain = lambda v: float(v) if isinstance(v, (float, np.floating)) else v  # noqa: E731
        return {"summary": [{k: plain(v) for k, v in r.items()} for r in rows], "identity": id_rows, "glomeruli": glom,
                "alignment": moved, "final": final, "scoring": "partly drawn" if partly_drawn else "strict"}


def manual_row(fly) -> dict:
    """A fly's own ROIs (``labels``) as a row of `Model.evaluate`'s summary ("manual"): how many, as 3D
    objects and in-plane masks, their crosstalk and how they keep from each other (`_purity`), to
    compare predicted ROIs with ('' where there is nothing to say)."""
    lab, ct = fly["labels"], refine.crosstalk_report(fly["responses"], fly["labels"], fly["spacing"])
    return {"fly": fly["name"], "rois": "manual", "n": len(np.unique(lab)) - 1, "recall": "", "precision": "",
            "dice": "", "splits": "", "merges": "", "n_scored": "",
            "n_2d": int(sum(len(np.unique(z)) - 1 for z in lab)), "n_scored_2d": "",
            "recall_2d": "", "precision_2d": "", "dice_2d": "",
            "coherence": np.mean(list(ct["coherence"].values())) if ct["coherence"] else "",
            "neighbor_corr": np.mean(list(ct["neighbor_corr"].values())) if ct["neighbor_corr"] else "",
            "leak": np.mean(np.abs(list(ct["leak"].values()))) if ct["leak"] else "",
            **_purity(fly["responses"], lab, fly["spacing"]), "breadth": "", "breadth_manual": ""}


def score(fly, labels, identities=None, kind: str = "pred", ref=None, partly_drawn: bool = True) -> dict:
    """Scores one kind of a fly's predicted ROIs (``labels``, named ``identities``: {label: (name, cost,
    confidence)}) as `Model.evaluate` does: how they keep from each other (crosstalk, `_purity`: against
    the neighbours ``ref`` had, the ROIs they were cleaned from, if given) and, if the fly has ROIs of its
    own (``labels``, named by its ``lut``), how they match them, as 3D objects and plane by plane (as
    ``partly_drawn`` says: see `Model.evaluate`), and whether they are named as the one they overlap.

    Returns:
        dict with ``row`` (`Model.evaluate`'s summary row of the kind, ``kind``: '' for what cannot be
        said, as all that needs the fly's own ROIs, for a fly without them) and, for a fly with ROIs of
        its own (else None each): ``identity`` ({"correct", "total"}: of the predicted ROIs overlapping
        one of its own by IoU over `NAME_IOU`, those named as it), ``lines`` (per predicted ROI: its
        label, name, the manual one's name, cost and confidence) and ``match`` ((IoU matrix, the manual
        ROIs' labels, the predicted ones'), as `metrics.iou_matrix` gives them).
    """
    resp, sp, gt = fly["responses"], fly["spacing"], fly.get("labels")
    ct = refine.crosstalk_report(resp, labels, sp)
    row = {"fly": fly.get("name"), "rois": kind, "n": len(np.unique(labels)) - 1, "recall": "", "precision": "", "dice": "",
           "splits": "", "merges": "", "n_scored": "", "n_2d": int(sum(len(np.unique(z)) - 1 for z in labels)), "n_scored_2d": "",
           "recall_2d": "", "precision_2d": "", "dice_2d": "",
           "coherence": np.mean(list(ct["coherence"].values())) if ct["coherence"] else "",
           "neighbor_corr": np.mean(list(ct["neighbor_corr"].values())) if ct["neighbor_corr"] else "",
           "leak": np.mean(np.abs(list(ct["leak"].values()))) if ct["leak"] else "",
           **_purity(resp, labels, sp, ref), "breadth": "", "breadth_manual": ""}
    if gt is None:
        return {"row": row, "identity": None, "lines": None, "match": None}
    geo = metrics.match_report(gt, labels, partly_drawn=partly_drawn)
    g2 = metrics.match_report_2d(gt, labels, partly_drawn=partly_drawn)
    br = refine.breadth(resp, gt, labels)
    row.update(recall=geo.get("recall", ""), precision=geo.get("precision", ""), dice=geo.get("mean_dice_matched", ""),
               splits=geo.get("splits", ""), merges=geo.get("merges", ""), n_scored=geo["n_scored"], n_2d=g2["n_pred"],
               n_scored_2d=g2["n_scored"], recall_2d=g2["recall"], precision_2d=g2["precision"], dice_2d=g2["mean_dice_matched"],
               breadth="" if br["pred"] is None else br["pred"], breadth_manual="" if br["manual"] is None else br["manual"])
    if partly_drawn:
        iou, g_ids, p_ids, _ = metrics.iou_matrix_partly_drawn(gt, labels)
    else:
        iou, g_ids, p_ids = metrics.iou_matrix(gt, labels)
    lut, identities = fly["lut"], identities or {}
    correct = total = 0
    lines = []
    for j, pl in enumerate(p_ids):
        manual = lut.get(int(g_ids[iou[:, j].argmax()])) if iou.size and iou[:, j].max() > NAME_IOU else None
        name, cost, conf = identities.get(int(pl), (None, np.nan, np.nan))
        lines.append([int(pl), name, manual, f"{cost:.3f}", f"{conf:.3f}"])
        if manual:
            total += 1
            correct += name == manual
    return {"row": row, "identity": {"correct": correct, "total": total}, "lines": lines, "match": (iou, g_ids, p_ids)}


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
    all_odors = list(odors or train_f[0]["odors"])
    if o.image_odors is not None and (len(o.image_odors) < 2 or not set(o.image_odors) <= set(all_odors)):
        raise ValueError(f"image_odors: at least 2 of the flies' odors ({', '.join(all_odors)}); given {', '.join(o.image_odors)}")
    img = [i for i, x in enumerate(all_odors) if o.image_odors is None or x in set(o.image_odors)]
    pca = responses.SharedPCA(3).fit([f["responses"][img] for f in train_f], masks=[f["labels"] > 0 for f in train_f])
    for f in flies:
        f["channels"] = responses.build_channels(f["average"], f["responses"][img], pca)
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
    return Model(seg, xy_um, pca, template, atl, all_odors, dict(aliases or {}), names, o, info)
