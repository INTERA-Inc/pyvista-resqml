"""Pytest fixtures for pyvista-resqml tests."""

import numpy as np
import pathlib
import pytest
import pyvista as pv

import pvresqml


class Helpers:
    def write_read(self, input_mesh, atol, tmp_path):
        in_mesh = input_mesh.copy(deep=True)

        p = tmp_path / "test.epc"
        pvresqml.save(p, input_mesh)
        mesh = pvresqml.read(p)
        
        assert type(mesh) == type(in_mesh)

        if isinstance(mesh, pv.StructuredGrid):
            assert np.allclose(mesh.x, in_mesh.x, atol=atol)
            assert np.allclose(mesh.y, in_mesh.y, atol=atol)
            assert np.allclose(mesh.z, in_mesh.z, atol=atol)

        else:
            assert np.allclose(mesh.points, in_mesh.points, atol=atol)
            assert np.allclose(mesh.celltypes, in_mesh.celltypes, atol=atol)
            assert np.allclose(mesh.offset, in_mesh.offset, atol=atol)
            assert mesh.n_cells == in_mesh.n_cells

            for i1, i2 in zip(mesh.offset[:-1], mesh.offset[1:]):
                cell = np.sort(mesh.cell_connectivity[i1 : i2])
                in_cell = np.sort(in_mesh.cell_connectivity[i1 : i2])
                assert np.allclose(cell, in_cell, atol=atol)

        for k, v in in_mesh.point_data.items():
            assert np.allclose(v, mesh.point_data[k], atol=atol)

        for k, v in in_mesh.cell_data.items():
            for data, in_data in zip(v, mesh.cell_data[k]):
                assert np.allclose(data, in_data, atol=atol)


@pytest.fixture
def helpers():
    """Fixture for helper functions."""
    return Helpers()


@pytest.fixture
def block():
    """Fixture for block.epc file."""
    return read_epc_file("block.epc")


@pytest.fixture
def sbend():
    """Fixture for s_bend.epc file."""
    return read_epc_file("s_bend.epc")


@pytest.fixture
def structured():
    """Fixture for structured grid mesh."""
    return pvresqml.examples.load_structured()


@pytest.fixture
def tetra():
    """Fixture for tetrahedral mesh."""
    return pvresqml.examples.load_tetra()


@pytest.fixture
def pyramid():
    """Fixture for pyramid mesh."""
    return pvresqml.examples.load_pyramid()


@pytest.fixture
def wedge():
    """Fixture for wedge mesh."""
    return pvresqml.examples.load_wedge()


@pytest.fixture
def hexahedron():
    """Fixture for hexahedral mesh."""
    return pvresqml.examples.load_hexahedron()


@pytest.fixture
def hybrid():
    """Fixture for hybrid mesh."""
    return pvresqml.examples.load_hybrid()


@pytest.fixture
def polyhedron():
    """Fixture for polyhedral mesh."""
    return pvresqml.examples.load_polyhedron()


def read_epc_file(filename):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "support_files" / filename
    mesh = pvresqml.read(filename)

    if isinstance(mesh, pv.ExplicitStructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
        
        for key in "IJK":
            if f"BLOCK_{key}" in mesh.cell_data:
                mesh.cell_data.pop(f"BLOCK_{key}", None)

    return mesh
