from __future__ import annotations
from typing import Optional, Union
from numpy.typing import ArrayLike

import meshio
import numpy as np
import pathlib

from resqpy.crs import Crs
from resqpy.model import new_model
from resqpy.property import GridPropertyCollection
from resqpy.unstructured import UnstructuredGrid, HexaGrid, TetraGrid


meshio_type_to_faces = {
    "tetra": {
        "triangle": np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]),
    },
    "pyramid": {
        "triangle": np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
        "quad": np.array([[0, 3, 2, 1]]),
    },
    "wedge": {
        "triangle": np.array([[0, 2, 1], [3, 4, 5]]),
        "quad": np.array([[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]]),
    },
    "hexahedron": {
        "quad": np.array(
            [
                [0, 3, 2, 1],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [0, 4, 7, 3],
            ]
        ),
    },
}


def write(
    filename: Union[str, pathlib.Path],
    mesh: meshio.Mesh,
    uom: Optional[dict] = None,
) -> None:
    """
    Write RESQML EPC and H5 files.

    Parameters
    ----------
    filename : str or :class:`pathlib.Path`
        Output file name.
    mesh : :class:`meshio.Mesh`
        Mesh to export.
    uom : dict or None, optional, default None
        Unit of measures for each data arrays.

    """
    uom = uom if uom else {}

    # Filter out 1D and 2D cells
    idx = [
        i for i, cell in enumerate(mesh.cells)
        if cell.type in {"tetra", "pyramid", "wedge", "hexahedron"}
        or cell.type.startswith("polyhedron")
    ]
    if not idx:
        raise ValueError("no 3D cell found in input mesh")

    cells = [mesh.cells[i] for i in idx]
    cell_data = {k: [v[i] for i in idx] for k, v in mesh.cell_data.items()}

    # Generate face data
    faces = [
        [c[v] for v in meshio_type_to_faces[cell.type].values()]
        for cell in cells
        for c in cell.data
    ]

    face_map = {}
    nodes_per_face = []
    faces_per_cell = []

    count = 0
    for cell in faces:
        for face_cell in cell:
            for face in face_cell:
                face_ = tuple(sorted(face))

                try:
                    idx = face_map[face_]

                except KeyError:
                    face_map[face_] = count
                    nodes_per_face.append(face)
                    idx = count
                    count += 1

                faces_per_cell.append(idx)

    # Initialize model
    model = new_model(filename)

    # Generate unstructured grid
    n_cells = sum(len(c) for c in cells)
    cell_types = [c.type for c in cells]

    if len(cell_types) == 1:
        if cell_types[0] == "tetra":
            grid = TetraGrid(model, find_properties=False)
            face_count_per_cell = 4
            node_count_per_face = 3

        elif cell_types[0] == "hexahedron":
            grid = HexaGrid(model, find_properties=False)
            face_count_per_cell = 6
            node_count_per_face = 4

        grid.set_cell_count(n_cells)
        grid.face_count = len(face_map)
        grid.nodes_per_face = np.concatenate(nodes_per_face).astype(int)
        grid.faces_per_cell_cl = np.arange(1, grid.cell_count + 1, dtype=int) * face_count_per_cell
        grid.faces_per_cell = np.array(faces_per_cell, dtype=int)
        grid.nodes_per_face_cl = np.arange(1, grid.face_count + 1, dtype=int) * node_count_per_face

    else:
        grid = UnstructuredGrid(model, find_properties=False, geometry_required=False)
        raise NotImplementedError()

    # Set point array
    grid.points_cached = np.array(mesh.points)
    grid.node_count = len(mesh.points)

    # Determine right handedness of cell faces w.r.t. cell center
    # The calculation is based on the sign of the scalar product of the face normal vector
    # and a vector defined by the cell center and any point on the face
    cell_centers = np.concatenate([mesh.points[cell.data].mean(axis=1) for cell in cells])
    face_to_cell_idx = np.searchsorted(
        grid.faces_per_cell_cl - 1,
        np.arange(grid.faces_per_cell.size),
        side="left",
    )

    face_first_node = np.insert(grid.nodes_per_face_cl[:-1], 0, 0)
    face_three_first_idx = (face_first_node[:, None] + np.arange(3)).ravel()
    face_three_first_nodes = grid.nodes_per_face[face_three_first_idx].reshape((grid.face_count, 3))
    tri_face_points = mesh.points[face_three_first_nodes[grid.faces_per_cell]]
    
    det = slicing_summing(
        tri_face_points[:, 2] - tri_face_points[:, 1],
        tri_face_points[:, 0] - tri_face_points[:, 1],
        cell_centers[face_to_cell_idx] - tri_face_points[:, 1],
    )
    grid.cell_face_is_right_handed = det >= 0.0

    # Generate property collection
    pc = None

    if mesh.point_data or cell_data:
        pc = GridPropertyCollection(grid)

        for k, v in mesh.point_data.items():
            _ = pc.add_cached_array_to_imported_list(
                v,
                source_info="meshio-resqml",
                keyword=k,
                indexable_element="points",
                discrete=v[0].dtype.kind in {"i", "u"},
                uom=uom[k] if k in uom else None,
            )

        for k, v in cell_data.items():
            _ = pc.add_cached_array_to_imported_list(
                np.concatenate(v),
                source_info="meshio-resqml",
                keyword=k,
                indexable_element="cells",
                discrete=v[0][0].dtype.kind in {"i", "u"},
                uom=uom[k] if k in uom else None,
            )

    # Add a coordinate system
    crs = Crs(model, z_inc_down=False)
    grid.crs_uuid = crs.uuid

    # Write files
    crs.create_xml()
    grid.write_hdf5(write_active=True)
    grid.create_xml(write_active=True)
    
    if pc is not None:
        pc.write_hdf5_for_imported_list()
        pc.create_xml_for_imported_list_and_add_parts_to_model()

    model.store_epc()


def slicing_summing(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> ArrayLike:
    """
    Calculate scalar triple product.

    Note
    ----
    See <https://stackoverflow.com/a/42386330/353337>.
    
    """
    c0 = b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]
    c1 = b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2]
    c2 = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]

    return a[:, 0] * c0 + a[:, 1] * c1 + a[:, 2] * c2
