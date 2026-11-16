"""Steps that clean a fly's ROIs of their neighbours' signal, after they are made and named.

A step is a function ``(labels (Z, Y, X), fly, **params) -> labels`` that only removes pixels, in-plane,
and keeps each ROI's label (so its name carries over). ``fly`` is as `alseg.fly.load_fly` gives it,
with ``evidence`` (`alseg.evidence`, read by `ensure_evidence` from its folder, ``dir``) for the
steps that go by tuning, and ``cellprob`` (Cellpose's cell probability map) for ``core``.

`STEPS`, in the order they are applied when several are asked for (`Options.clean`):

- ``trim``: `alseg.clean_trim.trim`, an ROI's edge cut back where it carries a neighbouring ROI's tuning;
- ``margin``: `alseg.clean_margin.margin`, every ROI eroded, and kept a distance from the others;
- ``core``: `alseg.clean_margin.core`, an ROI's pixels Cellpose is surer of.
"""
from __future__ import annotations

import importlib

import numpy as np

STEPS = {"trim": ("clean_trim", "trim"), "margin": ("clean_margin", "margin"), "core": ("clean_margin", "core")}
EVIDENCE = ("trim",)  # the steps that read the fly's evidence
REPORTS = ("trim",)  # the steps that say what they did (`apply`'s reports)
KIND = "clean"  # the kind of ROIs the steps asked for make (`Model.predict`); one step alone: clean_<step>


def step(name: str):
    """The function of step ``name``.

    Raises:
        ValueError: if there is no such step.
    """
    if name not in STEPS:
        raise ValueError(f"clean: no step {name!r} (the steps: {', '.join(STEPS)})")
    module, fn = STEPS[name]
    return getattr(importlib.import_module(f"{__package__}.{module}"), fn)


def ordered(steps) -> list[str]:
    """``steps`` in the order they are applied (`STEPS`'), each once."""
    for s in steps:
        step(s)
    return [s for s in STEPS if s in steps]


def ensure_evidence(fly: dict, say=print) -> None:
    """Reads ``fly``'s evidence (`alseg.evidence.load`) from its folder (``dir``), once. A fly without
    one, or without recordings of it, keeps none: the steps then go by its trial-mean responses."""
    if fly.get("evidence") is not None or not fly.get("dir"):
        return
    from . import evidence

    try:
        fly["evidence"] = evidence.load(fly["dir"], fly["average"].shape)
    except (OSError, ValueError) as e:
        say(f"  {fly.get('name')}: no per-trial responses read ({e}): cleaning by its trial means")


def usable(name: str, fly: dict) -> bool:
    """Whether step ``name`` can run on ``fly`` (``core`` needs Cellpose's probabilities)."""
    return name != "core" or fly.get("cellprob") is not None


def apply(labels: np.ndarray, fly: dict, steps, params: dict | None = None, reports: dict | None = None) -> np.ndarray:
    """``labels`` after ``steps`` (in `STEPS`' order), each with its parameters
    (``params[step]``: {name: value}, the step's defaults otherwise). Steps that cannot run on the
    fly (`usable`) are left out. ``reports``, if given, gets the report of each step that makes one
    (`REPORTS`: {step: its report}), as the step's ``return_report`` gives it."""
    params = params or {}
    out = labels
    for s in ordered(steps):
        if not usable(s, fly):
            continue
        if s in EVIDENCE:
            ensure_evidence(fly)
        if reports is not None and s in REPORTS:
            out, reports[s] = step(s)(out, fly, return_report=True, **(params.get(s) or {}))
        else:
            out = step(s)(out, fly, **(params.get(s) or {}))
    return out
