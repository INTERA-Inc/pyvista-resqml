from __future__ import annotations

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike

from .._common import generate_polyhedron_connectivity


def load_structured() -> pv.StructuredGrid:
    """
    Load structured grid.

    Returns
    -------
    :class:`pv.StructuredGrid`
        Structured grid.

    """
    x = np.linspace(0.0, 5.0, 6)
    y = np.linspace(0.0, 4.0, 5)
    z = np.linspace(0.0, 3.0, 4)
    mesh = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing="ij"))

    return _add_data(mesh)


def load_tetra() -> pv.UnstructuredGrid:
    """
    Load unstructured grid with two tetrahedra.

    Returns
    -------
    :class:`pv.UnstructuredGrid`
        Unstructured grid.

    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    cells = {pv.CellType.TETRA: np.array([[0, 1, 2, 4], [0, 2, 3, 4]])}

    return _load_from_points_cells(points, cells)


def load_pyramid() -> pv.UnstructuredGrid:
    """
    Load unstructured grid with two pyramids.

    Returns
    -------
    :class:`pv.UnstructuredGrid`
        Unstructured grid.

    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, -1.0],
        ]
    )
    cells = {pv.CellType.PYRAMID: np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]])}

    return _load_from_points_cells(points, cells)


def load_wedge() -> pv.UnstructuredGrid:
    """
    Load unstructured grid with two wedges.

    Returns
    -------
    :class:`pv.UnstructuredGrid`
        Unstructured grid.

    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 1.0],
            [0.0, 0.5, -1.0],
            [1.0, 0.5, -1.0],
        ]
    )
    cells = {pv.CellType.WEDGE: np.array([[0, 3, 4, 1, 2, 5], [0, 3, 6, 1, 2, 7]])}

    return _load_from_points_cells(points, cells)


def load_hexahedron() -> pv.UnstructuredGrid:
    """
    Load unstructured grid with two hexahedra.

    Returns
    -------
    :class:`pv.UnstructuredGrid`
        Unstructured grid.

    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ]
    )
    cells = {
        pv.CellType.HEXAHEDRON: np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]]
        )
    }

    return _load_from_points_cells(points, cells)


def load_hybrid() -> pv.UnstructuredGrid:
    """
    Load unstructured grid with different cell types.

    Returns
    -------
    :class:`pv.UnstructuredGrid`
        Unstructured grid.

    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 1.5],
            [0.0, 0.5, 1.5],
            [1.0, 0.5, 1.5],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [2.0, 0.5, 1.0],
            [-1.0, 0.5, 1.0],
        ]
    )
    cells = [
        8, 0, 1, 2, 3, 4, 5, 6, 7,
        5, 4, 5, 6, 7, 8,
        4, 4, 8, 7, 9,
        4, 5, 6, 8, 10,
        6, 1, 11, 5, 2, 12, 6,
        6, 13, 0, 4, 14, 3, 7,
        30, 7,
        3, 5, 6, 10,
        3, 11, 12, 15,
        3, 5, 11, 15,
        3, 5, 15, 10,
        3, 6, 10, 15,
        3, 6, 15, 12,
        4, 11, 12, 6, 5,
        30, 7,
        3, 4, 7, 9,
        3, 13, 14, 16,
        3, 4, 16, 13,
        3, 4, 9, 16,
        3, 7, 9, 16,
        3, 7, 16, 14,
        4, 13, 14, 7, 4,
    ]  # fmt: skip
    celltypes = [
        pv.CellType.HEXAHEDRON,
        pv.CellType.PYRAMID,
        pv.CellType.TETRA,
        pv.CellType.TETRA,
        pv.CellType.WEDGE,
        pv.CellType.WEDGE,
        pv.CellType.POLYHEDRON,
        pv.CellType.POLYHEDRON,
    ]

    return _load_from_points_cells(points, cells, celltypes)


def load_polyhedron() -> pv.UnstructuredGrid:
    """
    Load unstructured grid with three polyhedra.

    Returns
    -------
    :class:`pv.UnstructuredGrid`
        Unstructured grid.

    """
    points = np.array(
        [
            [0.3568221, -0.49112344, 0.79465446],
            [-0.3568221, -0.49112344, 0.79465446],
            [0.3568221, 0.49112344, -0.79465446],
            [-0.3568221, 0.49112344, -0.79465446],
            [0.0, 0.98224693, 0.18759243],
            [0.0, 0.60706198, 0.79465446],
            [0.0, -0.60706198, -0.79465446],
            [0.0, -0.98224693, -0.18759243],
            [0.93417233, 0.30353101, 0.18759247],
            [0.93417233, -0.30353101, -0.18759247],
            [-0.93417233, 0.30353101, 0.18759247],
            [-0.93417233, -0.30353101, -0.18759247],
            [-0.57735026, 0.18759249, 0.79465446],
            [0.57735026, -0.79465446, 0.18759249],
            [-0.57735026, -0.18759249, -0.79465446],
            [0.57735026, 0.79465446, -0.18759249],
            [0.57735026, 0.18759249, 0.79465446],
            [-0.57735026, 0.79465446, -0.18759249],
            [-0.57735026, -0.79465446, 0.18759249],
            [0.57735026, -0.18759249, -0.79465446],
            [0.3568221, 0.49112344, -1.0],
            [0.57735026, -0.18759249, -1.0],
            [0.0, -0.60706198, -1.0],
            [-0.57735026, -0.18759249, -1.0],
            [-0.3568221, 0.49112344, -1.0],
            [0.3568221, -0.49112344, 1.0],
            [0.57735026, 0.18759249, 1.0],
            [0.0, 0.60706198, 1.0],
            [-0.57735026, 0.18759249, 1.0],
            [-0.3568221, -0.49112344, 1.0],
        ]
    )
    faces = [
        [
            [0, 16, 5, 12, 1],
            [1, 18, 7, 13, 0],
            [2, 19, 6, 14, 3],
            [3, 17, 4, 15, 2],
            [4, 5, 16, 8, 15],
            [5, 4, 17, 10, 12],
            [6, 7, 18, 11, 14],
            [7, 6, 19, 9, 13],
            [8, 16, 0, 13, 9],
            [9, 19, 2, 15, 8],
            [10, 17, 3, 14, 11],
            [11, 18, 1, 12, 10],
        ],
        [
            [2, 19, 6, 14, 3],
            [20, 21, 19, 2],
            [21, 22, 6, 19],
            [22, 23, 14, 6],
            [23, 24, 3, 14],
            [24, 20, 2, 3],
            [20, 21, 22, 23, 24],
        ],
        [
            [0, 16, 5, 12, 1],
            [0, 16, 26, 25],
            [16, 5, 27, 26],
            [5, 12, 28, 27],
            [12, 1, 29, 28],
            [1, 0, 25, 29],
            [25, 26, 27, 28, 29],
        ],
    ]
    celltypes = [pv.CellType.POLYHEDRON] * 3

    cells = []

    for faces_ in faces:
        cells += generate_polyhedron_connectivity(faces_)

    return _load_from_points_cells(points, cells, celltypes)


def _add_data(
    mesh: pv.StructuredGrid | pv.UnstructuredGrid,
    add_point_data: bool = True,
    add_cell_data: bool = True,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    """Add data to mesh."""
    if add_point_data:
        mesh.point_data["PointIndex"] = np.arange(mesh.n_points)

    if add_cell_data:
        mesh.cell_data["CellIndex"] = np.arange(mesh.n_cells)

    return mesh


def _load_from_points_cells(
    points: ArrayLike,
    cells: dict | ArrayLike,
    celltypes: Optional[ArrayLike] = None,
    add_point_data: bool = True,
    add_cell_data: bool = True,
) -> pv.UnstructuredGrid:
    """Load mesh using points and cells."""
    if isinstance(cells, dict):
        mesh = pv.UnstructuredGrid(cells, points)

    elif celltypes is not None:
        mesh = pv.UnstructuredGrid(cells, celltypes, points)

    else:
        raise ValueError("could not load grid from points and cells")

    return _add_data(mesh, add_point_data, add_cell_data)
