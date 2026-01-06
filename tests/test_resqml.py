import pytest


@pytest.mark.parametrize(
    "mesh_fixture",
    [
        pytest.param("structured", id="structured"),
        pytest.param("tetra", id="tetra"),
        pytest.param("pyramid", id="pyramid"),
        pytest.param("wedge", id="wedge"),
        pytest.param("hexahedron", id="hexahedron"),
        pytest.param("hybrid", id="hybrid"),
        pytest.param("polyhedron", id="polyhedron"),
        pytest.param("block", id="block"),
        pytest.param("sbend", id="sbend"),
    ],
)
def test_mesh(mesh_fixture, helpers, tmp_path, request):
    mesh = request.getfixturevalue(mesh_fixture)
    helpers.write_read(mesh, 1.0e-15, tmp_path)
