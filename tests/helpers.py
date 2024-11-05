import numpy as np
import pyvista as pv


def write_read(tmp_path, writer, reader, input_mesh, atol):
    in_mesh = input_mesh.copy(deep=True)

    p = tmp_path / "test.epc"
    writer(p, input_mesh)
    mesh = reader(p)
    
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
