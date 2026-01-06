<p align="center">
   <h1 align="center"><b>PyVista RESQML</b></h1>
</p>

[![Pyversions](https://img.shields.io/pypi/pyversions/pyvista-resqml.svg?style=flat)](https://pypi.org/pypi/pyvista-resqml/)
[![Version](https://img.shields.io/pypi/v/pyvista-resqml.svg?style=flat)](https://pypi.org/project/pyvista-resqml)
[![Downloads](https://pepy.tech/badge/pyvista-resqml)](https://pepy.tech/project/pyvista-resqml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black)

A PyVista extension for seamless reading and writing of RESQML data (EPC files).

## Features

- **Read RESQML EPC files**: Load subsurface models and associated metadata directly into Python.

- **Write RESQML EPC files**: Export your processed or generated models back to RESQML format for interoperability.

- **PyVista Integration**: Visualize and manipulate geological and reservoir models using PyVista’s powerful 3D plotting capabilities.

- **Powered by resqpy**: Leverages the robust RESQML handling capabilities of [**resqpy**](https://github.com/bp/resqpy) for reliable data parsing and writing.

## Installation

The recommended way to install **pyvista-resqml** and all its dependencies is through the Python Package Index:

```bash
pip install pyvista-resqml --user
```

Otherwise, clone and extract the package, then run from the package location:

```bash
pip install .[full] --user
```

To test the integrity of the installed package, check out this repository and run:

```bash
pytest
```

# Usage

```python
import pyvista as pv
import pvresqml

# Load a mesh (e.g., from GRDECL)
mesh = pv.read("mesh.GRDECL")

# Save to a RESQML file
pvresqml.save("mesh.epc", mesh)

# Read a RESQML file
mesh_epc = pvresqml.read("mesh.epc")
```

## Acknowledgements

This project is supported by Nagra (National Cooperative for the Disposal of Radioactive Waste), Switzerland.