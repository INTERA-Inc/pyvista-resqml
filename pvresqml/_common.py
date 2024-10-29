from __future__ import annotations
from numpy.typing import ArrayLike

import pyvista as pv


def generate_polyhedron_connectivity(faces: list[ArrayLike]) -> ArrayLike:
    connectivity = [len(faces)]
    for face in faces:
        connectivity += [len(face), *face]

    connectivity.insert(0, len(connectivity))

    return connectivity
    