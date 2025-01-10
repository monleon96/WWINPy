"""
Weight window operations module.
Handles storage and manipulation of weight window values through the WeightWindowValues class.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict
import numpy as np
import pandas as pd
from wwinpy.header import Header
from wwinpy.mesh import Mesh
from wwinpy.query import QueryResult
from wwinpy._utils import get_closest_indices, get_range_indices, get_bin_intervals_from_indices
from wwinpy._ratios import calculate_max_ratio_array


@dataclass
class WeightWindowValues:
    """Weight window values container class.

    Internal container class for managing weight window values and operations.

    :param header: WWINP file header information
    :type header: Header
    :param mesh: Mesh geometry and binning information
    :type mesh: Mesh
    :param ww_values: Dictionary mapping particle types to weight window arrays
    :type ww_values: Dict[int, np.ndarray]
    """
    header: Header
    mesh: Mesh
    ww_values: Dict[int, np.ndarray] = field(default_factory=dict)

    def multiply(self, factor: float = 2.0, 
            particle_types: int | list[int] = -1) -> None:
        """Multiply weight window values by a factor.

        :param factor: Multiplication factor to apply
        :type factor: float
        :param particle_types: Particle type(s) to process. Use -1 for all types, or specify
                           individual types as an integer or list of integers
        :type particle_types: Union[int, List[int]]
        :raises ValueError: If an invalid particle type is specified
        """
        # Handle particle type selection
        if isinstance(particle_types, int):
            if particle_types == -1:
                particle_types = list(range(self.header.ni))
            else:
                particle_types = [particle_types]
        
        # Validate particle types
        for p_type in particle_types:
            if not 0 <= p_type < self.header.ni:
                raise ValueError(f"Invalid particle type {p_type}. Must be between 0 and {self.header.ni-1}")
                
        for particle_idx in particle_types:
            self.ww_values[particle_idx] *= factor

    def soften(self, power: float = 0.6,
           particle_types: int | list[int] = -1) -> None:
        """Modify weight window boundaries by raising values to a power.

        :param power: Exponent to apply to values (< 1 softens, > 1 hardens)
        :type power: float
        :param particle_types: Particle type(s) to process. Use -1 for all types, or specify
                           individual types as an integer or list of integers
        :type particle_types: Union[int, List[int]]
        :raises ValueError: If an invalid particle type is specified
        """
        # Handle particle type selection
        if isinstance(particle_types, int):
            if particle_types == -1:
                particle_types = list(range(self.header.ni))
            else:
                particle_types = [particle_types]
        
        # Validate particle types
        for p_type in particle_types:
            if not 0 <= p_type < self.header.ni:
                raise ValueError(f"Invalid particle type {p_type}. Must be between 0 and {self.header.ni-1}")
                
        for particle_idx in particle_types:
            self.ww_values[particle_idx] = np.power(self.ww_values[particle_idx], power)

    def apply_ratio_threshold(self, threshold: float = 10.0,
            particle_types: int | list[int] = -1,
            verbose: bool = False) -> None:
        """Apply a ratio threshold to identify and modify extreme weight window differences bewteen neighbor cells.

        :param threshold: Maximum allowed ratio between neighboring cells
        :type threshold: float
        :param particle_types: Particle type(s) to process. Use -1 for all types, or specify
                           individual types as an integer or list of integers
        :type particle_types: Union[int, List[int]]
        :param verbose: If True, prints detailed detailed detailed information about modifications
        :type verbose: bool
        :return: None
        :raises ValueError: If an invalid particle type is specified

        When verbose=True, the following information is printed:
        - Time and energy bin information
        - Modified voxel positions and values
        - Summary statistics of changes
        """
        # Handle particle type selection
        if isinstance(particle_types, int):
            if particle_types == -1:
                particle_types = list(range(self.header.ni))
            else:
                particle_types = [particle_types]
        
        # Validate particle types
        for p_type in particle_types:
            if not 0 <= p_type < self.header.ni:
                raise ValueError(f"Invalid particle type {p_type}. Must be between 0 and {self.header.ni-1}")

        # Get spatial mesh coordinates
        x_grid = self.mesh.fine_geometry_mesh['x']
        y_grid = self.mesh.fine_geometry_mesh['y']
        z_grid = self.mesh.fine_geometry_mesh['z']

        total_changes = 0
        total_cells = 0
        
        for p_idx in particle_types:
            ww = self.ww_values[p_idx]
            
            if verbose:
                print(f"\nProcessing particle type {p_idx}:")
            
            # Get energy bins (add 0.0 as lower bound for first bin)
            energy_mesh = self.mesh.energy_mesh[p_idx]
            energy_bins = np.insert(energy_mesh, 0, 0.0)
            
            # Get time bins (if time-dependent)
            if self.header.has_time_dependency:
                time_bins = self.mesh.time_mesh[p_idx]
            else:
                time_bins = np.array([0, float('inf')])
            
            # Process each time bin
            for t in range(ww.shape[0]):
                time_slice = ww[t]
                time_changes = 0
                time_cells = 0
                
                # Store changes per energy bin
                energy_changes = {}
                
                for e in range(time_slice.shape[0]):
                    spatial_view = time_slice[e].reshape(
                        int(self.header.nfz),
                        int(self.header.nfy),
                        int(self.header.nfx)
                    )
                    total_in_bin = spatial_view.size
                    ratios = calculate_max_ratio_array(spatial_view)
                    mask = ratios > threshold
                    
                    changes = np.sum(mask)
                    if changes > 0:
                        energy_changes[e] = (changes, total_in_bin)
                        time_changes += changes
                        
                        if verbose:
                            positions = np.where(mask)
                            energy_start = energy_bins[e]
                            energy_end = energy_bins[e + 1]
                            time_start = time_bins[t]
                            time_end = time_bins[t + 1] if t + 1 < len(time_bins) else float('inf')
                            
                            print(f"\nTime bin: [{time_start:.2e}, {time_end:.2e}]")
                            print(f"Energy bin: [{energy_start:.2e}, {energy_end:.2e}] MeV")
                            
                            # Create DataFrame with modified voxels information
                            data = []
                            for z, y, x in zip(*positions):
                                data.append({
                                    'Position': f"({z},{y},{x})",
                                    'X Range': f"[{x_grid[x]:.1f}, {x_grid[x+1]:.1f}]",
                                    'Y Range': f"[{y_grid[y]:.1f}, {y_grid[y+1]:.1f}]",
                                    'Z Range': f"[{z_grid[z]:.1f}, {z_grid[z+1]:.1f}]",
                                    'Ratio': f"{ratios[z,y,x]:.2f}",
                                    'Value': f"{spatial_view[z,y,x]:.2e}"
                                })
                            
                            if data:
                                df = pd.DataFrame(data)
                                print("\nModified voxels:")
                                # Set display options for better visualization
                                with pd.option_context('display.max_rows', None,
                                                     'display.max_columns', None,
                                                     'display.width', None):
                                    print(df.to_string(index=False))
                            
                        spatial_view[mask] = 0.0
                    
                    time_cells += total_in_bin  # Move this outside the if changes > 0 block
                
                total_changes += time_changes
                total_cells += time_cells  # This now has the correct count

        if verbose:
            # Summary statistics
            if total_changes > 0:
                summary_data = {
                    'Total Cells': [total_cells],
                    'Modified Cells': [total_changes],
                    'Percentage': [f"{total_changes/total_cells*100:.2f}%"]
                }
                summary_df = pd.DataFrame(summary_data)
                print("\nSummary:")
                print(summary_df.to_string(index=False))
            else:
                print("\nNo modifications were needed - all ratios are within threshold")


    def query_ww(self, particle_type: int | None = None,
            time: Optional[Union[float, Tuple[float, float]]] = None,
            energy: Optional[Union[float, Tuple[float, float]]] = None,
            x: Optional[Union[float, Tuple[float, float]]] = None,
            y: Optional[Union[float, Tuple[float, float]]] = None,
            z: Optional[Union[float, Tuple[int, int]]] = None) -> QueryResult:
        """Query weight window values based on specified criteria.

        :param particle_type: Specific particle type to query (None for all types)
        :type particle_type: Optional[int]
        :param time: Time value or (min, max) range for time-dependent data
        :type time: Optional[Union[float, Tuple[float, float]]]
        :param energy: Energy value or (min, max) range in MeV
        :type energy: Optional[Union[float, Tuple[float, float]]]
        :param x: X coordinate value or (min, max) range in geometry units
        :type x: Optional[Union[float, Tuple[float, float]]]
        :param y: Y coordinate value or (min, max) range in geometry units
        :type y: Optional[Union[float, Tuple[float, float]]]
        :param z: Z coordinate value or (min, max) range in geometry units
        :type z: Optional[Union[float, Tuple[int, int]]]
        :return: Object containing queried values and metadata
        :rtype: QueryResult
        :raises ValueError: If an invalid particle type is specified

        :Example:

            >>> ww = wwinpy.from_file(path/to/wwinp)
            >>> result = ww.query_ww(
            ...     particle_type=0,
            ...     energy=(1.0, 10.0),
            ...     x=(0, 10),
            ...     y=(0, 10),
            ...     z=(0, 10)
            ... ).to_dataframe()
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
            particle_ww = self.ww_values[p_type]

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

