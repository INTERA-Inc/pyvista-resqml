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
from resqpy.unstructured import HexaGrid, TetraGrid, UnstructuredGrid


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

    except AssertionError:
        raise ValueError("no compatible grid found.")

    if isinstance(grid, Grid):
        points, cells = _read_grid(grid)

    elif isinstance(grid, HexaGrid):
        points, cells = _read_hexagrid(grid)

    elif isinstance(grid, TetraGrid):
        points, cells = _read_tetragrid(grid)

    else:
        raise NotImplementedError()

    # Read data arrays
    point_data = {}
    cell_data = {}

    pc = grid.property_collection
    if pc.number_of_parts():
        for uuid, title in zip(pc.uuids(), pc.titles()):
            prop = Property(model, uuid=uuid)
            data = prop.array_ref().ravel(order="C")
            data = (
                data.astype(float)
                if prop.is_continuous()
                else data.astype(int)
            )

            if prop.is_points():
                point_data[title] = data
            
            else:
                cell_data[title] = [data]

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


def _read_hexagrid(grid: HexaGrid) -> tuple[ArrayLike, list[tuple[str, ArrayLike]]]:
    """Read an HexaGrid object."""
    points = grid.points_ref()
    cells = np.empty((grid.cell_count, 8), dtype=int)

    nodes_per_face = grid.nodes_per_face.reshape((grid.face_count, 4), order="C")
    faces = grid.faces_per_cell.reshape((grid.cell_count, 6), order="C")

    for i, cell in enumerate(faces):
        cell = nodes_per_face[cell]
        face1 = cell[0]
        face2 = []

        for edge in (face1[:2], face1[2:]):
            for face in cell[1:]:
                idx = np.intersect1d(face, edge, assume_unique=True)

                if idx.size == 2:
                    hankel = np.column_stack((face, np.append(face[1:], face[0])))
                    face = face[::-1] if (hankel == edge).all(axis=1).any() else face
                    face2 += [i for i in face if i not in edge]

                    break

        if len(face2) != 4:
            raise ValueError(f"failed to identify opposing faces for cell {i}")

        cells[i] = np.concatenate((face1, face2))

    cells = [("hexahedron", cells)]

    return points, cells


def _read_tetragrid(grid: TetraGrid) -> tuple[ArrayLike, list[tuple[str, ArrayLike]]]:
    """Read a TetraGrid object."""
    points = grid.points_ref()
    cells = np.empty((grid.cell_count, 4), dtype=int)

    nodes_per_face = grid.nodes_per_face.reshape((grid.face_count, 3), order="C")
    faces = grid.faces_per_cell.reshape((grid.cell_count, 4), order="C")
    cells = np.row_stack([np.unique(cell) for cell in nodes_per_face[faces]])
    cells = [("tetra", cells)]

    return points, cells
