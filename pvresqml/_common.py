from __future__ import annotations

import pyvista as pv
from numpy.typing import ArrayLike


def generate_polyhedron_connectivity(faces: list[ArrayLike]) -> ArrayLike:
    """Generate connectivity for polyhedral cells."""
    connectivity = [len(faces)]
    for face in faces:
        connectivity += [len(face), *face]

    connectivity.insert(0, len(connectivity))

    return connectivity
