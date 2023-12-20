import copy

import meshio
import numpy as np

tetra_mesh = meshio.Mesh(
    points=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    ),
    cells=[("tetra", np.array([[0, 1, 2, 4], [0, 2, 3, 4]]))],
    point_data={"a": np.random.rand(5), "b": np.random.randint(100, size=5)},
    cell_data={"c": [np.arange(2)], "d": [np.random.rand(2)]},
)

pyramid_mesh = meshio.Mesh(
    points=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, -1.0],
        ]
    ),
    cells=[("pyramid", np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]))],
    point_data={"a": np.random.rand(6), "b": np.random.randint(100, size=6)},
    cell_data={"c": [np.arange(2)], "d": [np.random.rand(2)]},
)

wedge_mesh = meshio.Mesh(
    points=np.array(
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
    ),
    cells=[("wedge", np.array([[0, 3, 4, 1, 2, 5], [0, 3, 6, 1, 2, 7]]))],
    point_data={"a": np.random.rand(8), "b": np.random.randint(100, size=8)},
    cell_data={"c": [np.arange(2)], "d": [np.random.rand(2)]},
)

hexahedron_mesh = meshio.Mesh(
    points=np.array(
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
    ),
    cells=[
        ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]]))
    ],
    point_data={"a": np.random.rand(12), "b": np.random.randint(100, size=12)},
    cell_data={"c": [np.arange(2)], "d": [np.random.rand(2)]},
)

hybrid_mesh = meshio.Mesh(
    points=np.array(
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
        ]
    ),
    cells=[
        ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
        ("pyramid", np.array([[4, 5, 6, 7, 8]])),
        ("tetra", np.array([[4, 8, 7, 9], [5, 6, 8, 10]])),
        ("wedge", np.array([[1, 11, 5, 2, 12, 6], [13, 0, 4, 14, 3, 7]])),
    ],
    point_data={"a": np.random.rand(15), "b": np.random.randint(100, size=15)},
    cell_data={
        "c": np.split(np.arange(6), [1, 2, 4]),
        "d": np.split(np.random.rand(6), [1, 2, 4]),
    },
)

poly_hybrid_mesh = meshio.Mesh(
    points=np.array(
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
        ]
    ),
    cells=[
        (
            "polyhedron8",
            [
                [
                    [0, 3, 2, 1],
                    [4, 5, 6, 7],
                    [0, 1, 5, 4],
                    [1, 2, 6, 5],
                    [2, 3, 7, 6],
                    [0, 4, 7, 3],
                ],
            ],
        ),
        (
            "polyhedron5",
            [
                [
                    [4, 5, 6, 7],
                    [4, 5, 8],
                    [5, 6, 8],
                    [6, 7, 8],
                    [7, 4, 8],
                ],
            ],
        ),
        (
            "polyhedron4",
            [
                [
                    [8, 7, 9],
                    [4, 9, 7],
                    [4, 8, 9],
                    [7, 4, 8],
                ],
                [
                    [6, 8, 10],
                    [5, 10, 8],
                    [5, 6, 10],
                    [5, 6, 8],
                ],
            ],
        ),
        (
            "polyhedron6",
            [
                [
                    [1, 5, 11],
                    [2, 12, 6],
                    [1, 11, 12, 2],
                    [11, 5, 6, 12],
                    [1, 2, 6, 5],
                ],
                [
                    [13, 4, 0],
                    [14, 3, 7],
                    [13, 0, 3, 14],
                    [0, 4, 7, 3],
                    [13, 14, 7, 4],
                ],
            ],
        ),
    ],
    point_data={"a": np.random.rand(15), "b": np.random.randint(100, size=15)},
    cell_data={
        "c": np.split(np.arange(6), [1, 2, 4]),
        "d": np.split(np.random.rand(6), [1, 2, 4]),
    },
)

polyhedron_mesh = meshio.Mesh(
    points=[
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
    ],
    cells=[
        (
            "polyhedron20",
            [
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
            ],
        ),
        (
            "polyhedron10",
            [
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
            ],
        ),
    ],
    point_data={"a": np.random.rand(30), "b": np.random.randint(100, size=30)},
    cell_data={"c": np.split(np.arange(3), (1,)), "d": np.split(np.random.rand(3), (1,))},
)


def write_read(tmp_path, writer, reader, input_mesh, atol):
    in_mesh = copy.deepcopy(input_mesh)

    p = tmp_path / "test.epc"
    writer(p, input_mesh)
    mesh = reader(p)

    assert np.allclose(mesh.points, in_mesh.points, atol=atol)

    assert len(mesh.cells) == len(in_mesh.cells)
    for cell, in_cell in zip(mesh.cells, in_mesh.cells):
        assert len(cell.data) == len(in_cell.data)

        if in_cell.type.startswith("polyhedron"):
            if cell.type == in_cell.type:
                for face, in_face in zip(cell.data, in_cell.data):
                    assert np.allclose(np.concatenate(face), np.concatenate(in_face), atol=atol)

            else:
                # meshes like poly_hybrid_mesh are not read as polyhedral meshes
                for face, in_face in zip(cell.data, in_cell.data):
                    assert np.allclose(
                        np.unique(face),
                        np.unique(np.concatenate(in_face)),
                        atol=atol,
                    )

        else:
            assert cell.type == in_cell.type

            # nodes order is not preserved due to node-based to face-based conversion
            assert np.allclose(
                np.sort(cell.data, axis=1),
                np.sort(in_cell.data, axis=1),
                atol=atol,
            )

    assert len(mesh.point_data) == len(in_mesh.point_data)
    for k, v in mesh.point_data.items():
        assert np.allclose(v, in_mesh.point_data[k], atol=atol)

    assert len(mesh.cell_data) == len(in_mesh.cell_data)
    for k, v in mesh.cell_data.items():
        for data, in_data in zip(v, in_mesh.cell_data[k]):
            assert np.allclose(data, in_data, atol=atol)
