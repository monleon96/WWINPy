# plotter.py
# Functions and classes for visualization

import numpy as np
import pyvista as pv
from typing import Tuple, List
from .ww_data import WWData

class WWPlotter:
    def __init__(self, wwinp: WWData):
        """
        Initialize the plotter with WWData.
        
        Parameters:
            wwinp (WWData): The weight window data to visualize
        """
        self.wwinp = wwinp
        self.plotter = pv.Plotter()
        self._current_grid = None
        self._energy_idx = 0
        self.mesh_opacity = 0.7

    def _create_mesh_grid(self, particle_type: int, time_idx: int = 0) -> pv.RectilinearGrid:
        """
        Create a PyVista RectilinearGrid from the mesh data.
        
        Parameters:
            particle_type (int): Particle type index
            time_idx (int): Time index (default=0)
            
        Returns:
            pv.RectilinearGrid: PyVista grid object
        """
        mesh = self.wwinp.mesh.fine_geometry_mesh
        
        # Get coordinate arrays
        x = mesh['x']
        y = mesh['y']
        z = mesh['z']
        
        # Create the grid
        grid = pv.RectilinearGrid(x, y, z)
        
        # Get the weight window values for all energies
        values = self.wwinp.values.particles[particle_type].ww_values
        if self.wwinp.header.has_time_dependency:
            values = values[time_idx]
        else:
            values = values[0]
            
        # Reshape the values for each energy level
        ne = values.shape[0]
        nz, ny, nx = len(z)-1, len(y)-1, len(x)-1
        shaped_values = values.reshape(ne, nz, ny, nx)
        
        # Add each energy level as a separate array
        for e in range(ne):
            grid.cell_data[f'energy_{e}'] = shaped_values[e].flatten(order='F')
            
        return grid

    def _update_energy(self, energy_idx: int) -> None:
        """Callback for energy slider updates."""
        if self._current_grid is None:
            return
            
        self.plotter.remove_actor('mesh')
        active_arr = f'energy_{energy_idx}'
        mesh_actor = self.plotter.add_mesh(
            self._current_grid,
            scalars=active_arr,
            opacity=self.mesh_opacity,
            name='mesh',
            show_edges=True,
            cmap='viridis',
            clim=self._get_value_range()
        )
        self._energy_idx = energy_idx
        self.plotter.render()

    def _get_value_range(self) -> Tuple[float, float]:
        """Get the global min/max range for all energy levels."""
        values = self._current_grid.cell_data.values()
        all_values = np.concatenate([arr.flatten() for arr in values])
        return float(np.min(all_values)), float(np.max(all_values))

    def _get_energy_values(self, particle_type: int) -> List[float]:
        """Get the energy values for slider labels."""
        return [f"{e:.2e}" for e in self.wwinp.mesh.energy_mesh[particle_type]]

    def plot_3d(self, particle_type: int, time_idx: int = 0) -> None:
        """
        Create an interactive 3D visualization with energy slider.
        
        Parameters:
            particle_type (int): Particle type to visualize
            time_idx (int): Time index for time-dependent data
        """
        self._current_grid = self._create_mesh_grid(particle_type, time_idx)
        
        # Set up the plotter
        self.plotter = pv.Plotter()
        self.plotter.add_mesh(
            self._current_grid,
            scalars=f'energy_0',
            opacity=self.mesh_opacity,
            name='mesh',
            show_edges=True,
            cmap='viridis',
            clim=self._get_value_range()
        )
        
        # Add energy slider
        energy_values = self._get_energy_values(particle_type)
        self.plotter.add_slider_widget(
            callback=self._update_energy,
            rng=[0, len(energy_values)-1],
            value=0,
            title=f'Energy (MeV)',
            pointa=(0.025, 0.1),
            pointb=(0.31, 0.1),
            style='modern'
        )
        
        # Add text for current energy value
        self.plotter.add_text(
            f"Particle Type: {particle_type}\nEnergy: {energy_values[0]} MeV",
            position='upper_left'
        )
        
        self.plotter.show()

    def plot_slices(self, particle_type: int, time_idx: int = 0) -> None:
        """
        Create an interactive visualization with orthogonal slices.
        
        Parameters:
            particle_type (int): Particle type to visualize
            time_idx (int): Time index for time-dependent data
        """
        self._current_grid = self._create_mesh_grid(particle_type, time_idx)
        
        # Set up the plotter
        self.plotter = pv.Plotter()
        
        # Add slice planes
        mesh_bounds = self._current_grid.bounds
        center = [(mesh_bounds[i+1] + mesh_bounds[i])/2 for i in (0,2,4)]
        
        # Add slices along each axis
        self.plotter.add_mesh_slice_orthogonal(
            self._current_grid,
            scalars=f'energy_0',
            cmap='viridis',
            clim=self._get_value_range()
        )
        
        # Add energy slider
        energy_values = self._get_energy_values(particle_type)
        self.plotter.add_slider_widget(
            callback=self._update_energy,
            rng=[0, len(energy_values)-1],
            value=0,
            title=f'Energy (MeV)',
            pointa=(0.025, 0.1),
            pointb=(0.31, 0.1),
            style='modern'
        )
        
        # Add text for current energy value
        self.plotter.add_text(
            f"Particle Type: {particle_type}\nEnergy: {energy_values[0]} MeV",
            position='upper_left'
        )
        
        self.plotter.show()

    def plot_volume(self, particle_type: int, time_idx: int = 0) -> None:
        """
        Create a volume rendering visualization.
        
        Parameters:
            particle_type (int): Particle type to visualize
            time_idx (int): Time index for time-dependent data
        """
        self._current_grid = self._create_mesh_grid(particle_type, time_idx)
        
        # Set up the plotter
        self.plotter = pv.Plotter()
        
        # Add volume rendering
        self.plotter.add_volume(
            self._current_grid,
            scalars=f'energy_0',
            cmap='viridis',
            opacity='linear',
            clim=self._get_value_range()
        )
        
        # Add energy slider
        energy_values = self._get_energy_values(particle_type)
        self.plotter.add_slider_widget(
            callback=self._update_energy,
            rng=[0, len(energy_values)-1],
            value=0,
            title=f'Energy (MeV)',
            pointa=(0.025, 0.1),
            pointb=(0.31, 0.1),
            style='modern'
        )
        
        # Add text for current energy value
        self.plotter.add_text(
            f"Particle Type: {particle_type}\nEnergy: {energy_values[0]} MeV",
            position='upper_left'
        )
        
        self.plotter.show()



