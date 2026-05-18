from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from resqpy.crs import Crs
from resqpy.grid import Grid
from resqpy.model import Model, new_model
from resqpy.property import GridPropertyCollection
from resqpy.unstructured import UnstructuredGrid


if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import ArrayLike, NDArray


def save(
    filename: str | os.PathLike,
    mesh: pv.DataObject
    | pv.ExplicitStructuredGrid
    | pv.StructuredGrid
    | pv.UnstructuredGrid,
    uom: Optional[dict] = None,
) -> None:
    """
    Write RESQML EPC and H5 files.

    Parameters
    ----------
    filename : str | PathLike
        Output file name.
    mesh : pyvista.DataObject | pyvista.ExplicitStructuredGrid | pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Mesh to export.
    uom : dict, optional
        Unit of measure for data arrays. Supercede unit of measures defined in key *property* of *pyvista.DataSet.user_dict*.

    """
    from . import __version__ as version

    uom = uom if uom else {}

    # Initialize path
    path = pathlib.Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = new_model(str(filename))

    # Generate grid
    if isinstance(mesh, pv.StructuredGrid):
        grid = _save_structured(mesh, model)

    elif isinstance(mesh, (pv.ExplicitStructuredGrid, pv.UnstructuredGrid)):
        mesh = mesh.cast_to_unstructured_grid()

        for key in "IJK":
            if f"BLOCK_{key}" in mesh.cell_data:
                mesh.cell_data.pop(f"BLOCK_{key}", None)

        try:
            z_inc_down = mesh.user_dict["crs"]["z_inc_down"]

        except KeyError:
            z_inc_down = False

        grid = _save_unstructured(mesh, model, z_inc_down)

    # Generate property collection
    pc = None

    if mesh.point_data or mesh.cell_data:
        pc = GridPropertyCollection(grid)

        for k, v in mesh.point_data.items():
            _ = pc.add_cached_array_to_imported_list(
                (
                    v
                    if isinstance(grid, UnstructuredGrid)
                    else v.reshape(grid.extent_kji[[2, 1, 0]] + 1).transpose((2, 1, 0))  # type: ignore
                ),
                source_info=f"pyvista-resqml v{version}",
                keyword=k,
                indexable_element="nodes",
                discrete=v[0].dtype.kind in {"i", "u"},
                uom=uom[k] if k in uom else _get_property_uom(mesh, k),
            )

        for k, v in mesh.cell_data.items():
            _ = pc.add_cached_array_to_imported_list(
                v if isinstance(grid, UnstructuredGrid) else v.reshape(grid.extent_kji),  # type: ignore
                source_info=f"pyvista-resqml v{version}",
                keyword=k,
                indexable_element="cells",
                discrete=v[0].dtype.kind in {"i", "u"},
                uom=uom[k] if k in uom else _get_property_uom(mesh, k),
            )

    # Add a coordinate system
    crs = (
        Crs(model, **mesh.user_dict["crs"])
        if "crs" in mesh.user_dict
        else Crs(model, z_inc_down=False)
    )
    grid.crs_uuid = crs.uuid  # type: ignore

    # Write files
    h5_filename = f"{path.stem}.h5"
    kwargs = {} if isinstance(grid, Grid) else {"write_active": True}

    crs.create_xml()
    grid.write_hdf5(h5_filename, **kwargs)  # type: ignore
    grid.create_xml(**kwargs)

    if pc is not None:
        pc.write_hdf5_for_imported_list(h5_filename)
        pc.create_xml_for_imported_list_and_add_parts_to_model()

    model.store_epc()


def _save_structured(mesh: pv.StructuredGrid, model: Model) -> Grid:
    """Save a structured grid."""
    ni, nj, nk = mesh.dimensions
    extent_kji = np.array([nk - 1, nj - 1, ni - 1], dtype=int)

    points_cached = np.concatenate(
        (
            np.expand_dims(mesh.x.transpose((2, 1, 0)), axis=-1),
            np.expand_dims(mesh.y.transpose((2, 1, 0)), axis=-1),
            np.expand_dims(mesh.z.transpose((2, 1, 0)), axis=-1),
        ),
        axis=-1,
    )

    grid = Grid(model, find_properties=False, geometry_required=False)
    grid.grid_representation = "IjkGrid"
    grid.extent_kji = extent_kji  # type: ignore
    grid.nk, grid.nj, grid.ni = extent_kji
    grid.points_cached = points_cached  # type: ignore
    grid.inactive = np.zeros(extent_kji, dtype=bool)  # type: ignore

    grid.k_direction_is_down = True  # type: ignore
    grid.grid_is_right_handed = True  # type: ignore
    grid.pillar_shape = "straight"
    grid.has_split_coordinate_lines = False  # type: ignore
    grid.geometry_defined_for_all_pillars_cached = True  # type: ignore
    grid.geometry_defined_for_all_cells_cached = True  # type: ignore

    return grid


def _save_unstructured(
    mesh: pv.UnstructuredGrid, model: Model, z_inc_down: bool
) -> UnstructuredGrid:
    """Save an unstructured grid."""
    from pvgridder import get_cell_connectivity

    # Generate face data
    celltypes = mesh.celltypes
    connectivity = get_cell_connectivity(mesh, flatten=False)

    if celltypes.min() == celltypes.max():
        celltype = celltypes[0]
        cell_shape = _celltype_to_cell_shape[celltype]
        cell_faces = (
            connectivity
            if celltype == pv.CellType.POLYHEDRON
            else [
                [
                    face
                    for v in _celltype_to_faces[celltype].values()
                    for face in cell[v]
                ]
                for cell in connectivity
            ]
        )

    else:
        cell_shape = "polyhedral"
        cell_faces = [
            cell
            if celltype == pv.CellType.POLYHEDRON
            else [
                face for v in _celltype_to_faces[celltype].values() for face in cell[v]
            ]
            for cell, celltype in zip(connectivity, celltypes)
        ]

    face_map = {}
    nodes_per_face = []
    nodes_per_face_cl = [0]
    faces_per_cell = []
    faces_per_cell_cl = [0]

    count = 0
    for cell in cell_faces:
        for face in cell:
            face_ = tuple(sorted(face))

            try:
                idx = face_map[face_]

            except KeyError:
                face_map[face_] = count
                nodes_per_face.append(face)
                nodes_per_face_cl.append(nodes_per_face_cl[-1] + len(face))
                idx = count
                count += 1

            faces_per_cell.append(idx)

        faces_per_cell_cl.append(faces_per_cell_cl[-1] + len(cell))

    # Generate unstructured grid
    grid = UnstructuredGrid(
        model, find_properties=False, geometry_required=False, cell_shape=cell_shape
    )
    grid.set_cell_count(mesh.n_cells)
    grid.face_count = len(face_map)
    grid.nodes_per_face = np.concatenate(nodes_per_face).astype(int)  # type: ignore
    grid.nodes_per_face_cl = np.array(nodes_per_face_cl[1:], dtype=int)  # type: ignore
    grid.faces_per_cell = np.array(faces_per_cell, dtype=int)  # type: ignore
    grid.faces_per_cell_cl = np.array(faces_per_cell_cl[1:], dtype=int)  # type: ignore

    # Set point array
    grid.points_cached = np.array(mesh.points)  # type: ignore
    grid.node_count = len(mesh.points)

    # Determine right handedness of cell faces w.r.t. cell center
    # The calculation is based on the sign of the scalar product of the face normal vector
    # and a vector defined by the cell center and any point on the face
    face_to_cell_idx = np.searchsorted(
        grid.faces_per_cell_cl - 1,  # type: ignore
        np.arange(grid.faces_per_cell.size),  # type: ignore
        side="left",
    )

    face_first_node = np.insert(grid.nodes_per_face_cl[:-1], 0, 0)  # type: ignore
    face_three_first_idx = (face_first_node[:, None] + np.arange(3)).ravel()
    face_three_first_nodes = grid.nodes_per_face[face_three_first_idx].reshape(  # type: ignore
        (grid.face_count, 3)
    )
    tri_face_points = mesh.points[face_three_first_nodes[grid.faces_per_cell]]

    cell_centers = mesh.cell_centers().points
    det = _slicing_summing(
        tri_face_points[:, 2] - tri_face_points[:, 1],
        tri_face_points[:, 0] - tri_face_points[:, 1],
        cell_centers[face_to_cell_idx] - tri_face_points[:, 1],
    )
    grid.cell_face_is_right_handed = det >= 0.0 if z_inc_down else det <= 0.0  # type: ignore

    return grid


def _slicing_summing(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> NDArray:
    """
    Calculate scalar triple product.

    Note
    ----
    See <https://stackoverflow.com/a/42386330/353337>.

    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    c = np.asanyarray(c)

    c0 = b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]
    c1 = b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2]
    c2 = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]

    return a[:, 0] * c0 + a[:, 1] * c1 + a[:, 2] * c2


def _get_property_uom(
    mesh: pv.DataObject
    | pv.ExplicitStructuredGrid
    | pv.StructuredGrid
    | pv.UnstructuredGrid,
    key: str,
) -> str | None:
    """Get property's unit of measure, if any."""
    try:
        return mesh.user_dict["property"][key]["uom"]

    except (KeyError, TypeError):
        return None


_celltype_to_faces = {
    pv.CellType.TETRA: {
        "TRIANGLE": np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]),
    },
    pv.CellType.PYRAMID: {
        "QUAD": np.array([[0, 3, 2, 1]]),
        "TRIANGLE": np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
    },
    pv.CellType.WEDGE: {
        "TRIANGLE": np.array([[0, 2, 1], [3, 4, 5]]),
        "QUAD": np.array([[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]]),
    },
    pv.CellType.HEXAHEDRON: {
        "QUAD": np.array(
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

_celltype_to_cell_shape = {
    pv.CellType.TETRA: "tetrahedral",
    pv.CellType.PYRAMID: "pyramidal",
    pv.CellType.WEDGE: "prism",
    pv.CellType.HEXAHEDRON: "hexahedral",
    pv.CellType.POLYHEDRON: "polyhedral",
}
