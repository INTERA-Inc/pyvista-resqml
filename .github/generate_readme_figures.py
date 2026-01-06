import pyvista as pv
import pvresqml

# Load a mesh (e.g., from GRDECL)
mesh = pv.read("mesh.GRDECL")

# Save to a RESQML file
pvresqml.save("mesh.epc", mesh)

# Read a RESQML file
mesh_epc = pvresqml.read("mesh.epc")

# Plot
p = pv.Plotter(
    window_size=(1000, 600),
    shape=(1, 2),
    off_screen=True,
)
p.link_views()
p.subplot(0, 0)
p.add_mesh(mesh, scalars="PORO", show_edges=True)
p.add_title("Original GRDECL", font_size=12)
p.show_axes()
p.subplot(0, 1)
p.add_mesh(mesh_epc, scalars="PORO", show_edges=True)
p.add_title("Converted RESQML", font_size=12)
p.show_axes()
p.camera_position = [
    (1616638.3808622917, -166492.38668659984, 2782.255166003334),
    (1606983.727748393, -176147.03980049834, -6872.397947895156),
    (0.0, 0.0, 1.0),
]
p.screenshot("mesh.png", transparent_background=True, return_img=False)
