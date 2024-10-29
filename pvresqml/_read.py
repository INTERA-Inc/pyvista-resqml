from __future__ import annotations

import itertools
import os
from typing import Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike
from resqpy.crs import Crs
from resqpy.grid import Grid, any_grid
from resqpy.model import Model
from resqpy.property import Property
from resqpy.unstructured import (
    HexaGrid,
    PrismGrid,
    PyramidGrid,
    TetraGrid,
    UnstructuredGrid,
)

from ._common import generate_polyhedron_connectivity


def read(
    filename: str | os.PathLike,
    grid_uuid: Optional[str] = None,
) -> pv.ExplicitStructuredGrid | pv.UnstructuredGrid:
    """
    Read RESQML EPC file.

    Parameters
    ----------
    filename : str | PathLike
        Input file name.
    grid_uuid : str, optional
        UUID of the grid to be imported.

    Returns
    -------
    :class:`pyvista.ExplicitStructuredGrid` | :class:`pyvista.UnstructuredGrid`
        Output mesh.

    """
    model = Model(str(filename))

    if grid_uuid is None:
        for uuid, part in zip(model.uuids(), model.parts()):
            if (
                "IjkGridRepresentation" in part
                or "UnstructuredGridRepresentation" in part
            ):
                grid_uuid = uuid
                break

    try:
        grid = any_grid(model, uuid=grid_uuid)
        grid.cache_all_geometry_arrays()

    except AssertionError:
        raise ValueError("could not find any compatible grid")

    if isinstance(grid, Grid):
        mesh = _read_grid(grid)

    elif isinstance(
        grid, (HexaGrid, PrismGrid, PyramidGrid, TetraGrid, UnstructuredGrid)
    ):
        mesh = _read_unstructured_grid(grid)

    else:
        raise NotImplementedError()

    # Read coordinate system data
    crs = Crs(model, uuid=grid.crs_uuid)
    crs_dict = {}

    for key in _crs_keys:
        try:
            crs_dict[key] = getattr(crs, key)

        except AttributeError:
            pass

    mesh.user_dict["crs"] = crs_dict

    # Read data arrays
    pc = grid.property_collection

    if pc.number_of_parts():
        property_dict = {}

        for uuid, title in zip(pc.uuids(), pc.titles()):
            prop = Property(model, uuid=uuid)
            data = prop.array_ref().ravel()
            data = data.astype(float) if prop.is_continuous() else data.astype(int)
            indexable_element = prop.indexable_element()

            if indexable_element == "nodes":
                mesh.point_data[title] = data

            elif indexable_element == "cells":
                mesh.cell_data[title] = data

            property_dict[title] = {"uom": prop.uom()}

        mesh.user_dict["property"] = property_dict

    return mesh


def _read_grid(grid: Grid) -> pv.ExplicitStructuredGrid:
    """Read a Grid object."""
    corner_points = grid.corner_points()

    corners = np.empty((2 * grid.nk, 2 * grid.nj, 2 * grid.ni, 3))
    corners[::2, ::2, ::2] = corner_points[:, :, :, ::2, ::2, ::2].squeeze()
    corners[1::2, ::2, ::2] = corner_points[:, :, :, 1::2, ::2, ::2].squeeze()
    corners[1::2, 1::2, ::2] = corner_points[:, :, :, 1::2, 1::2, ::2].squeeze()
    corners[::2, 1::2, ::2] = corner_points[:, :, :, ::2, 1::2, ::2].squeeze()
    corners[::2, ::2, 1::2] = corner_points[:, :, :, ::2, ::2, 1::2].squeeze()
    corners[1::2, ::2, 1::2] = corner_points[:, :, :, 1::2, ::2, 1::2].squeeze()
    corners[1::2, 1::2, 1::2] = corner_points[:, :, :, 1::2, 1::2, 1::2].squeeze()
    corners[::2, 1::2, 1::2] = corner_points[:, :, :, ::2, 1::2, 1::2].squeeze()

    corners = corners.reshape((8 * grid.ni * grid.nj * grid.nk, 3))
    mesh = pv.ExplicitStructuredGrid((grid.ni + 1, grid.nj + 1, grid.nk + 1), corners)

    # Inactive cells
    inactive = grid.extract_inactive_mask().astype(bool)

    if inactive.any():
        mesh.hide_cells(inactive.ravel(), inplace=True)

    return mesh


def _read_unstructured_grid(
    grid: HexaGrid | PrismGrid | PyramidGrid | TetraGrid | UnstructuredGrid,
) -> pv.UnstructuredGrid:
    """Read an UnstructuredGrid object."""
    points = grid.points_ref()

    nodes_per_face_cl = np.insert(grid.nodes_per_face_cl, 0, 0)
    nodes_per_face = [
        grid.nodes_per_face[ibeg:iend]
        for ibeg, iend in zip(nodes_per_face_cl[:-1], nodes_per_face_cl[1:])
    ]

    faces_per_cell_cl = np.insert(grid.faces_per_cell_cl, 0, 0)
    faces_per_cell = [
        grid.faces_per_cell[ibeg:iend]
        for ibeg, iend in zip(faces_per_cell_cl[:-1], faces_per_cell_cl[1:])
    ]

    cells_ = []
    celltypes = []
    for cell in faces_per_cell:
        nodes_per_cell = [list(nodes_per_face[face_cell]) for face_cell in cell]
        node_count = tuple(sorted(map(len, nodes_per_cell)))

        try:
            celltypes.append(_node_count_to_cell_type[node_count])

        except KeyError:
            celltypes.append("POLYHEDRON")

        cells_.append(nodes_per_cell)

    cells = []
    for celltype, cell in zip(celltypes, cells_):
        if celltype == "TETRA":
            cell = to_tetra(cell)

        elif celltype == "PYRAMID":
            cell = to_pyramid(cell)

        elif celltype == "WEDGE":
            cell = to_wedge(cell)

        elif celltype == "HEXAHEDRON":
            cell = to_hexahedron(cell)

        else:
            cell = generate_polyhedron_connectivity(cell)[1:]

        cells += [len(cell), *cell]

    celltypes = [_celltype_map[celltype] for celltype in celltypes]

    return pv.UnstructuredGrid(cells, celltypes, points)


def to_tetra(cell: ArrayLike) -> ArrayLike:
    """Convert a face-based tetra to a node-based tetra."""
    base = cell[0]
    apex = list(set(cell[1]).difference(base))

    if len(apex) != 1:
        raise ValueError("could not find apex for tetra")

    return base + apex


def to_pyramid(cell: ArrayLike) -> ArrayLike:
    """Convert a face-based pyramid to a node-based pyramid."""
    apex = None

    for c in cell:
        if len(c) == 4:
            base = c
            break

    for c in cell:
        diff = set(c).difference(base)

        if len(diff) == 1:
            apex = list(diff)
            break

    if apex is None:
        raise ValueError("could not find apex for pyramid")

    return base + apex


def to_wedge(cell: ArrayLike) -> ArrayLike:
    """Convert a face-based wedge to a node-based wedge."""
    face1, face_ = [c for c in cell if len(c) == 3]
    face2 = []

    edge = face1[:2]
    edge_set = set(edge)

    for face in cell:
        if len(face) == 3:
            continue

        if len(edge_set.intersection(face)) == 2:
            hankel = np.column_stack((face, np.append(face[1:], face[0])))
            face = face[::-1] if (hankel == edge).all(axis=1).any() else face
            face2 += [i for i in face if i not in edge]

            break

    for i in face_:
        if i not in face2:
            face2.append(i)
            break

    if len(face2) != 3:
        raise ValueError("could not identify opposing faces for wedge")

    return face1 + face2


def to_hexahedron(cell: ArrayLike) -> ArrayLike:
    """Convert a face-based hexahedron to a node-based hexahedron."""
    face1 = cell[0]
    face2 = []

    for edge in (face1[:2], face1[2:]):
        edge_set = set(edge)

        for face in cell[1:]:
            if len(edge_set.intersection(face)) == 2:
                hankel = np.column_stack((face, np.append(face[1:], face[0])))
                face = face[::-1] if (hankel == edge).all(axis=1).any() else face
                face2 += [i for i in face if i not in edge]

                break

    if len(face2) != 4:
        raise ValueError("could not identify opposing faces for hexahedron")

    return face1 + face2


_node_count_to_cell_type = {
    (3, 3, 3, 3): "TETRA",
    (3, 3, 3, 3, 4): "PYRAMID",
    (3, 3, 4, 4, 4): "WEDGE",
    (4, 4, 4, 4, 4, 4): "HEXAHEDRON",
}

_celltype_map = {celltype.name: int(celltype) for celltype in pv.CellType}

_crs_keys = (
    "x_offset",
    "y_offset",
    "z_offset",
    "rotation",
    "rotation_units",
    "xy_units",
    "z_units",
    "z_inc_down",
    "axis_order",
    "axis_order",
    "time_units",
    "epsg_code",
    "title",
    "originator",
    "extra_metadata",
)
