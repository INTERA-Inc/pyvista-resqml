import meshio

from .__about__ import __version__
from ._read import read
from ._write import write

__all__ = [
    "read",
    "write",
    "__version__",
]

meshio.register_format("resqml", [".epc"], read, {"resqml": write})
