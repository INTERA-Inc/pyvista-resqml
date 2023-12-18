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
        helpers.dodecahedron_mesh,
    ],
)
def test(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio_resqml.write, meshio_resqml.read, mesh, 1.0e-15)
