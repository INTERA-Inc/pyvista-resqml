import meshio

from ._read import read
from ._write import write

__all__ = [
    "read",
    "write",
]

meshio.register_format("resqml", [".epc"], read, {"resqml": write})
