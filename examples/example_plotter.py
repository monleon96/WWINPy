from wwpy.plotter import WWINPPlotter
from wwpy.models import WWINPData
from wwpy.parser import parse_wwinp_file
import sys


ww_file_path = '/home/MONLEON-JUAN/WWPy/examples/wwinp_ueki'

try:
    wwinp: WWINPData = parse_wwinp_file(ww_file_path)
except Exception as e:
    print(f"Error parsing file: {e}")
    sys.exit(1)

# Assuming you have a WWINPData object called 'wwinp'
plotter = WWINPPlotter(wwinp)

# Create a 3D visualization for particle type 0
plotter.plot_3d(particle_type=0)

## Or create a slice view
#plotter.plot_slices(particle_type=0)
#
## Or create a volume rendering
#plotter.plot_volume(particle_type=0)