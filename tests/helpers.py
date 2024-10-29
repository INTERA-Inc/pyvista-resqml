import numpy as np


def write_read(tmp_path, writer, reader, input_mesh, atol):
    in_mesh = input_mesh.copy(deep=True)

    p = tmp_path / "test.epc"
    writer(p, input_mesh)
    mesh = reader(p)

    assert np.allclose(mesh.points, in_mesh.points, atol=atol)
    assert np.allclose(mesh.celltypes, in_mesh.celltypes, atol=atol)
    assert np.allclose(mesh.offset, in_mesh.offset, atol=atol)
    assert mesh.n_cells == in_mesh.n_cells

    for i1, i2 in zip(mesh.offset[:-1], mesh.offset[1:]):
        cell = np.sort(mesh.cell_connectivity[i1 : i2])
        in_cell = np.sort(in_mesh.cell_connectivity[i1 : i2])
        assert np.allclose(cell, in_cell, atol=atol)

    assert len(mesh.point_data) == len(in_mesh.point_data)
    for k, v in mesh.point_data.items():
        assert np.allclose(v, in_mesh.point_data[k], atol=atol)

    assert len(mesh.cell_data) == len(in_mesh.cell_data)
    for k, v in mesh.cell_data.items():
        for data, in_data in zip(v, in_mesh.cell_data[k]):
            assert np.allclose(data, in_data, atol=atol)
