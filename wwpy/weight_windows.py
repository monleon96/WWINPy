from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np
from wwpy.models import Header, Mesh, ParticleBlock
from wwpy.query import QueryResult
from wwpy.utils import get_closest_indices, get_range_indices, get_bin_intervals_from_indices


@dataclass
class WeightWindowValues:
    """
    Stores the weight window values for all particles.
    """
    header: Header
    mesh: Mesh
    particles: List[ParticleBlock] = field(default_factory=list)

    def query_ww(
        self,
        particle_type: Optional[int] = None,
        time: Optional[Union[float, Tuple[float, float]]] = None,
        energy: Optional[Union[float, Tuple[float, float]]] = None,
        x: Optional[Union[float, Tuple[float, float]]] = None,
        y: Optional[Union[float, Tuple[float, float]]] = None,
        z: Optional[Union[float, Tuple[int, int]]] = None,
        geom_idx: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> QueryResult:
        """
        Query weight window values based on specified criteria.
        """
        # Handle particle type selection
        if particle_type is not None:
            if not 0 <= particle_type < self.header.ni:
                raise ValueError(f"Invalid particle type {particle_type}")
            particle_types = [particle_type]
        else:
            particle_types = list(range(self.header.ni))

        results = []
        energy_intervals = []
        time_intervals = []

        # Get spatial meshes
        spatial_mesh = self.mesh.fine_geometry_mesh
        
        # Handle spatial coordinates
        x_grid = spatial_mesh['x']
        y_grid = spatial_mesh['y']
        z_grid = spatial_mesh['z']

        # Process x coordinate
        if x is not None:
            if isinstance(x, tuple):
                x_indices = get_range_indices(x_grid, x)
            else:
                x_indices = get_closest_indices(x_grid, x)
            x_starts, x_ends = get_bin_intervals_from_indices(x_grid, x_indices)
        else:
            x_indices = np.arange(len(x_grid)-1)
            x_starts, x_ends = x_grid[:-1], x_grid[1:]

        # Process y coordinate
        if y is not None:
            if isinstance(y, tuple):
                y_indices = get_range_indices(y_grid, y)
            else:
                y_indices = get_closest_indices(y_grid, y)
            y_starts, y_ends = get_bin_intervals_from_indices(y_grid, y_indices)
        else:
            y_indices = np.arange(len(y_grid)-1)
            y_starts, y_ends = y_grid[:-1], y_grid[1:]

        # Process z coordinate
        if z is not None:
            if isinstance(z, tuple):
                z_indices = get_range_indices(z_grid, z)
            else:
                z_indices = get_closest_indices(z_grid, z)
            z_starts, z_ends = get_bin_intervals_from_indices(z_grid, z_indices)
        else:
            z_indices = np.arange(len(z_grid)-1)
            z_starts, z_ends = z_grid[:-1], z_grid[1:]

        # Process each particle type
        for p_type in particle_types:
            # Get original energy grid and create extended version with 0.0
            orig_energy_grid = self.mesh.energy_mesh[p_type]
            energy_grid = np.insert(orig_energy_grid, 0, 0.0)

            # Handle energy query
            if energy is not None:
                if isinstance(energy, tuple):
                    # Range query
                    e_indices = get_range_indices(energy_grid, energy)
                    # Get the actual energy intervals
                    e_starts, e_ends = get_bin_intervals_from_indices(energy_grid, e_indices)
                    energy_intervals.append((e_starts, e_ends))
                    # Adjust indices for ww_values access (shift left by 1 and remove if negative)
                    e_indices = e_indices - 1
                    e_indices = e_indices[e_indices >= 0]
                else:
                    # Single value query
                    e_indices = get_closest_indices(energy_grid, energy)
                    # Get the actual energy intervals
                    e_starts, e_ends = get_bin_intervals_from_indices(energy_grid, e_indices)
                    energy_intervals.append((e_starts, e_ends))
                    # Adjust indices for ww_values access
                    e_indices = e_indices - 1
                    e_indices = e_indices[e_indices >= 0]
            else:
                # If no energy query, use all indices but account for the added 0.0
                e_indices = np.arange(len(orig_energy_grid))  # Indices for original grid
                energy_intervals.append((energy_grid[:-1], energy_grid[1:]))  # Use extended grid for intervals

            # Handle time query
            if self.header.has_time_dependency:
                time_grid = self.mesh.time_mesh[p_type]
                if time is not None:
                    if isinstance(time, tuple):
                        # Range query for time
                        t_indices = get_range_indices(time_grid, time)
                    else:
                        # Single value query for time
                        t_indices = get_closest_indices(time_grid, time)
                    # Get the actual time intervals
                    t_starts, t_ends = get_bin_intervals_from_indices(time_grid, t_indices)
                else:
                    # If no time query, use all indices
                    t_indices = np.arange(len(time_grid))
                    t_starts, t_ends = time_grid[:-1], time_grid[1:]
                time_intervals.append((t_starts, t_ends))
            else:
                t_indices = [0]
                time_intervals.append((np.array([]), np.array([])))

            # Get the ww values for this particle type
            particle_ww = self.particles[p_type].ww_values

            # Select the values using both time and energy indices
            if self.header.has_time_dependency:
                selected_ww = particle_ww[t_indices][:, e_indices]
            else:
                selected_ww = particle_ww[0:1, e_indices]  # Always use single time index for non-time-dependent

            # Reshape the ww values to match the spatial dimensions
            selected_ww = selected_ww.reshape(*selected_ww.shape[:-1], 
                                            int(self.header.nfz),
                                            int(self.header.nfy),
                                            int(self.header.nfx))
            
            # Create a view of the selected indices
            selected_ww = selected_ww[..., z_indices, :, :]
            selected_ww = selected_ww[..., :, y_indices, :]
            selected_ww = selected_ww[..., :, :, x_indices]

            results.append(selected_ww)

        return QueryResult(
            header=self.header,
            particle_types=particle_types,
            ww_values=results,
            energy_intervals=energy_intervals,
            time_intervals=time_intervals,
            x_intervals=(x_starts, x_ends),
            y_intervals=(y_starts, y_ends),
            z_intervals=(z_starts, z_ends)
        )