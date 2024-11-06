import pathlib

import helpers
import pytest

import pyvista as pv
import pvresqml


@pytest.mark.parametrize(
    "mesh",
    [
        pvresqml.examples.load_structured(),
        pvresqml.examples.load_tetra(),
        pvresqml.examples.load_pyramid(),
        pvresqml.examples.load_wedge(),
        pvresqml.examples.load_hexahedron(),
        pvresqml.examples.load_hybrid(),
        pvresqml.examples.load_polyhedron(),
        "block.epc",
        "s_bend.epc",
    ],
)
def test_mesh(mesh, tmp_path):
    if isinstance(mesh, str):
        this_dir = pathlib.Path(__file__).resolve().parent
        filename = this_dir / "support_files" / mesh
        mesh = pvresqml.read(filename)

        if isinstance(mesh, pv.ExplicitStructuredGrid):
            mesh = mesh.cast_to_unstructured_grid()
            
            for key in "IJK":
                if f"BLOCK_{key}" in mesh.cell_data:
                    mesh.cell_data.pop(f"BLOCK_{key}", None)

    helpers.write_read(tmp_path, pvresqml.save, pvresqml.read, mesh, 1.0e-15)
