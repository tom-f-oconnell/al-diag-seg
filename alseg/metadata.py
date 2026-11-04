"""Voxel sizes from ThorImage Experiment.xml files (read with hong2p)."""
from __future__ import annotations

from pathlib import Path


def thorimage_spacing(xml_path, shape_zyx=None) -> tuple[float, float, float]:
    """(z, y, x) voxel size in um of the ThorImage recording described by
    ``xml_path``. If ``shape_zyx`` is given, checks that it matches the recording's
    dimensions, since the XML's pixel size would be wrong for cropped or binned data.
    """
    from hong2p import thor

    xml = thor.xmlroot(str(xml_path))
    dxy = thor.get_thorimage_pixelsize_um(xml)
    (nx, ny), nz, _ = thor.get_thorimage_dims(xml)
    dz = abs(thor.get_thorimage_zstep_um(xml)) if nz > 1 else 1.0
    if shape_zyx is not None and tuple(shape_zyx) != (nz, ny, nx):
        raise ValueError(f"{xml_path} describes a (Z, Y, X) = {(nz, ny, nx)} recording, but the "
                         f"data are {tuple(shape_zyx)}. Its voxel size only applies to "
                         "uncropped, unbinned data.")
    if nz > 1 and dz == 0:
        raise ValueError(f"{xml_path}: z step is 0 for a {nz}-plane recording")
    return (dz, dxy, dxy)


def has_xml(d: Path) -> bool:
    return (Path(d) / "Experiment.xml").exists()
