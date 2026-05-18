"""
PyVista I/O registry entry points.

These thin adapters bridge :func:`pvresqml.read` and :func:`pvresqml.save`
to the signatures PyVista's pluggable reader/writer registry expects, so
that ``pyvista.read(...)`` and ``DataSet.save(...)`` handle ``.epc`` files
once ``pyvista-resqml`` is installed. They are wired up through the
``pyvista.readers`` and ``pyvista.writers`` entry-point groups declared in
``pyproject.toml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._read import read
from ._save import save


if TYPE_CHECKING:
    import pyvista as pv


def read_resqml(path: str, /, **kwargs: Any) -> pv.DataSet:
    """
    Read a RESQML EPC file for the PyVista reader registry.

    Parameters
    ----------
    path : str
        Input file name.
    **kwargs : Any
        Forwarded to :func:`pvresqml.read` (e.g., ``grid_uuid``).

    Returns
    -------
    pyvista.DataSet
        Output mesh.

    """
    return read(path, **kwargs)


def write_resqml(dataset: pv.DataObject, path: str, /, **kwargs: Any) -> None:
    """
    Write a mesh to a RESQML EPC file for the PyVista writer registry.

    Parameters
    ----------
    dataset : pyvista.DataObject
        Mesh to export.
    path : str
        Output file name.
    **kwargs : Any
        Forwarded to :func:`pvresqml.save` (e.g., ``uom``).

    """
    save(path, dataset, **kwargs)
