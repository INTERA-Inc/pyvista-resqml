import pathlib

import helpers
import pytest

import meshio_resqml


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tetra_mesh,
        helpers.pyramid_mesh,
        helpers.wedge_mesh,
        helpers.hexahedron_mesh,
        helpers.hybrid_mesh,
        helpers.poly_hybrid_mesh,
        helpers.polyhedron_mesh,
        "block.epc",
        "s_bend.epc",
    ],
)
def test_mesh(mesh, tmp_path):
    if isinstance(mesh, str):
        this_dir = pathlib.Path(__file__).resolve().parent
        filename = this_dir / "support_files" / mesh
        mesh = meshio_resqml.read(filename)

    helpers.write_read(tmp_path, meshio_resqml.write, meshio_resqml.read, mesh, 1.0e-15)
