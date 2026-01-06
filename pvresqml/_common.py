from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def generate_polyhedron_connectivity(faces: Sequence[Sequence[int]]) -> list[int]:
    """Generate connectivity for polyhedral cells."""
    connectivity = [len(faces)]
    for face in faces:
        connectivity += [len(face), *face]

    connectivity.insert(0, len(connectivity))

    return connectivity
