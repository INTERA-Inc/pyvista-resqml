from __future__ import annotations
from typing import Optional, Union
from numpy.typing import ArrayLike

import itertools
import meshio
import numpy as np
import pathlib

from resqpy.grid import any_grid, Grid
from resqpy.model import Model
from resqpy.property import Property
from resqpy.unstructured import HexaGrid, PrismGrid, PyramidGrid, TetraGrid, UnstructuredGrid


node_count_to_cell_type = {
    (3, 3, 3, 3): "tetra",
    (3, 3, 3, 3, 4): "pyramid",
    (3, 3, 4, 4, 4): "wedge",
    (4, 4, 4, 4, 4, 4): "hexahedron",
}


def read(
    filename: Union[str, pathlib.Path],
    grid_uuid: Optional[str] = None,
) -> meshio.Mesh:
    """
    Read RESQML EPC file.

    Parameters
    ----------
    filename : str or :class:`pathlib.Path`
        Input file name.
    grid_uuid : str or None, optional, default None
        UUID of the grid to be imported.

    Returns
    -------
    :class:`meshio.Mesh`
        Output mesh.

    """
    model = Model(filename)

    if grid_uuid is None:
        for uuid, part in zip(model.uuids(), model.parts()):
            if "IjkGridRepresentation" in part or "UnstructuredGridRepresentation" in part:
                grid_uuid = uuid
                break

    try:
        grid = any_grid(model, uuid=grid_uuid)
        grid.cache_all_geometry_arrays()

    except AssertionError:
        raise ValueError("no compatible grid found.")

    if isinstance(grid, Grid):
        points, cells = _read_grid(grid)

    elif isinstance(grid, (HexaGrid, PrismGrid, PyramidGrid, TetraGrid, UnstructuredGrid)):
        points, cells = _read_unstructured_grid(grid)

    else:
        raise NotImplementedError()

    # Read data arrays
    point_data = {}
    cell_data = {}

    pc = grid.property_collection
    if pc.number_of_parts():
        sizes = np.cumsum([len(c[1]) for c in cells])

        for uuid, title in zip(pc.uuids(), pc.titles()):
            prop = Property(model, uuid=uuid)
            data = prop.array_ref().ravel(order="C")
            data = (
                data.astype(float)
                if prop.is_continuous()
                else data.astype(int)
            )
            indexable_element = prop.indexable_element()

            if indexable_element == "nodes":
                point_data[title] = data
            
            elif indexable_element == "cells":
                cell_data[title] = np.split(data, sizes[:-1])

    return meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
    )


def _read_grid(grid: Grid) -> tuple[ArrayLike, list[tuple[str, ArrayLike]]]:
    """Read a Grid object."""
    corner_points = grid.corner_points().reshape((grid.nk, grid.nj, grid.ni, 8, 3))
    point_map = {}
    cells = []
    count = 0
    for k, j, i in itertools.product(range(grid.nk), range(grid.nj), range(grid.ni)):
        cell = []

        for point in corner_points[k, j, i]:
            point = tuple(point)

            try:
                idx = point_map[point]

            except KeyError:
                point_map[point] = count
                idx = count
                count += 1

            cell.append(idx)

        cells.append(cell)

    points = np.array(list(point_map))
    cells = np.array(cells, dtype=int)
    cells[:, [6, 7, 2, 3]] = cells[:, [7, 6, 3, 2]]
    cells = [("hexahedron", cells)]

    return points, cells


def _read_unstructured_grid(
    grid: Union[HexaGrid, PrismGrid, PyramidGrid, TetraGrid, UnstructuredGrid]
) -> tuple[ArrayLike, list[tuple[str, ArrayLike]]]:
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
    cell_types = []
    polyhedral = False
    for cell in faces_per_cell:
        nodes_per_cell = [list(nodes_per_face[face_cell]) for face_cell in cell]
        node_count = tuple(sorted(len(nodes) for nodes in nodes_per_cell))

        try:
            cell_types.append(node_count_to_cell_type[node_count])

        except KeyError:
            polyhedral = True
            cell_types.append("polyhedron")

        cells_.append(nodes_per_cell)

    cells = []
    for cell_type, cell in zip(cell_types, cells_):
        if polyhedral:
            cell_type = f"polyhedron{len(cell)}"

        elif cell_type == "tetra":
            cell = to_tetra(cell)

        elif cell_type == "pyramid":
            cell = to_pyramid(cell)

        elif cell_type == "wedge":
            cell = to_wedge(cell)

        elif cell_type == "hexahedron":
            cell = to_hexahedron(cell)

        if len(cells) == 0 or cells[-1][0] != cell_type:
            cells.append((cell_type, [cell]))

        else:
            cells[-1][1].append(cell)

    return points, cells


def to_tetra(cell: ArrayLike) -> list[int]:
    """Convert a face-based tetra to a node-based tetra."""
    base = cell[0]
    apex = list(set(cell[1]).difference(base))

    if len(apex) != 1:
        raise ValueError("failed to find apex for tetra")
    
    return base + apex


def to_pyramid(cell: ArrayLike) -> list[int]:
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
        raise ValueError("failed to find apex for pyramid")
    
    return base + apex


def to_wedge(cell: ArrayLike) -> list[int]:
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
        raise ValueError("failed to identify opposing faces for wedge")

    return face1 + face2


def to_hexahedron(cell: ArrayLike) -> list[int]:
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
        raise ValueError("failed to identify opposing faces for hexahedron")

    return face1 + face2
