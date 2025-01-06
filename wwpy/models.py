# wwpy/models.py

from wwpy.utils import (
    get_closest_indices, get_range_indices, get_bin_intervals_from_indices, 
    xyz_to_flat
    )
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple
import pandas as pd
import numpy as np

@dataclass
class Header:
    if_: int            # File type. Only 1 is supported.
    iv: int             # Time-dependent windows flag (1 / 2 = no / yes)
    ni: int             # Number of particle types   
    nr: int             # = 10 / 16 / 16 - rectangular / cylindrical / spherical
    probid: str = ""    # Made optional with default empty string

    # Optional arrays that might appear depending on 'iv' or 'nr'
    nt: List[int] = field(default_factory=list)
    ne: List[int] = field(default_factory=list)

    # Additional geometry specs
    nfx: Optional[float] = None
    nfy: Optional[float] = None
    nfz: Optional[float] = None
    x0: Optional[float] = None
    y0: Optional[float] = None
    z0: Optional[float] = None

    ncx: Optional[float] = None
    ncy: Optional[float] = None
    ncz: Optional[float] = None
    nwg: Optional[float] = None

    # Optional for nr = 16
    x1: Optional[float] = None
    y1: Optional[float] = None
    z1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    z2: Optional[float] = None

    @property
    def has_time_dependency(self) -> bool:
        """
        Returns True if 'iv' equals 2, indicating a time dependency.
        """
        return self.iv == 2

    @property
    def type_of_mesh(self) -> Optional[str]:
        """
        Returns the type of mesh based on the 'nwg' value.
        - 1: Cartesian
        - 2: Cylindrical
        - 3: Spherical
        Returns None if 'nwg' is not set or has an unexpected value.
        """
        mesh_types = {
            1: "cartesian",
            2: "cylindrical",
            3: "spherical"
        }
        if self.nwg is None:
            return None
        try:
            nwg_int = int(self.nwg)
            return mesh_types.get(nwg_int, "unknown")
        except ValueError:
            return "unknown"

    @property
    def number_of_time_bins(self) -> int:
        """
        Returns the number of time bins in a list. One element per particle type.
        """
        return self.nt

    @property
    def number_of_energy_bins(self) -> int:
        """
        Returns the number of energy bins in a list. One element per particle type.
        """
        return self.ne

    @property
    def number_of_particle_types(self) -> int:
        """
        Returns the number of particles.
        """
        return self.ni


@dataclass
class GeometryAxis:
    origin: float
    q: np.ndarray = field(default_factory=lambda: np.array([]))
    p: np.ndarray = field(default_factory=lambda: np.array([]))
    s: np.ndarray = field(default_factory=lambda: np.array([]))

    def add_segment(self, q: float, p: float, s: float):
        """
        Adds a coarse mesh segment to the axis.

        Parameters:
            q (float): Fine mesh ratio.
            p (float): Coarse mesh coordinate.
            s (float): Number of fine meshes in this segment.
        """
        self.q = np.append(self.q, np.float32(q))  # Explicitly cast to float32
        self.p = np.append(self.p, np.float32(p))  # Explicitly cast to float32
        self.s = np.append(self.s, np.int32(s))  # Explicitly cast to int32


@dataclass
class GeometryData:
    header: 'Header'
    # Cartesian axes
    x_axis: Optional[GeometryAxis] = None
    y_axis: Optional[GeometryAxis] = None
    z_axis: Optional[GeometryAxis] = None

    # Cylindrical axes
    r_axis: Optional[GeometryAxis] = None
    theta_axis: Optional[GeometryAxis] = None

    # Spherical axes
    phi_axis: Optional[GeometryAxis] = None

    def _generate_coarse_axis_mesh(self, axis: GeometryAxis) -> List[float]:
        """
        Generates the coarse mesh for a given GeometryAxis.

        Parameters:
            axis (GeometryAxis): The geometry axis to generate coarse mesh for.

        Returns:
            List[float]: The coarse mesh as a list of Python floats.
        """
        mesh = [float(axis.origin)]  # Ensure origin is a Python float
        for p in axis.p:
            mesh.append(float(p))  # Convert each point to Python float
        return mesh

    def _generate_fine_axis_mesh(self, axis: GeometryAxis) -> List[float]:
        """
        Generates the fine mesh for a given GeometryAxis.

        Parameters:
            axis (GeometryAxis): The geometry axis to generate fine mesh for.

        Returns:
            List[float]: The fine mesh as a list of Python floats.
        """
        fine_mesh = [float(axis.origin)]  # Ensure origin is a Python float
        current = axis.origin
        for p, s in zip(axis.p, axis.s):
            step = (p - current) / s
            s = int(s)  # Ensure s is an integer for range
            fine_mesh.extend(float(current + step * i) for i in range(1, s + 1))  # Convert to Python float
            current = p
        return fine_mesh

    @property
    def coarse_mesh(self) -> Dict[str, np.ndarray]:
        mesh_type = self.header.type_of_mesh 
        if mesh_type == "cartesian":
            if not all([self.x_axis, self.y_axis, self.z_axis]):
                raise ValueError("Cartesian mesh requires x_axis, y_axis, and z_axis to be defined.")
            return {
                'x': np.array(self._generate_coarse_axis_mesh(self.x_axis)),
                'y': np.array(self._generate_coarse_axis_mesh(self.y_axis)),
                'z': np.array(self._generate_coarse_axis_mesh(self.z_axis))
            }
        elif mesh_type == "cylindrical":
            if not all([self.r_axis, self.z_axis, self.theta_axis]):
                raise ValueError("Cylindrical mesh requires r_axis, z_axis, and theta_axis to be defined.")
            return {
                'r': np.array(self._generate_coarse_axis_mesh(self.r_axis)),
                'z': np.array(self._generate_coarse_axis_mesh(self.z_axis)),
                'theta': np.array(self._generate_coarse_axis_mesh(self.theta_axis))
            }
        elif mesh_type == "spherical":
            if not all([self.r_axis, self.theta_axis, self.phi_axis]):
                raise ValueError("Spherical mesh requires r_axis, theta_axis, and phi_axis to be defined.")
            return {
                'r': np.array(self._generate_coarse_axis_mesh(self.r_axis)),
                'theta': np.array(self._generate_coarse_axis_mesh(self.theta_axis)),
                'phi': np.array(self._generate_coarse_axis_mesh(self.phi_axis))
            }
        else:
            raise ValueError(f"Unsupported mesh type: {mesh_type}")

    @property
    def fine_mesh(self) -> Dict[str, np.ndarray]:
        mesh_type = self.header.type_of_mesh
        if mesh_type == "cartesian":
            if not all([self.x_axis, self.y_axis, self.z_axis]):
                raise ValueError("Cartesian mesh requires x_axis, y_axis, and z_axis to be defined.")
            return {
                'x': np.array(self._generate_fine_axis_mesh(self.x_axis)),
                'y': np.array(self._generate_fine_axis_mesh(self.y_axis)),
                'z': np.array(self._generate_fine_axis_mesh(self.z_axis))
            }
        elif mesh_type == "cylindrical":
            if not all([self.r_axis, self.z_axis, self.theta_axis]):
                raise ValueError("Cylindrical mesh requires r_axis, z_axis, and theta_axis to be defined.")
            return {
                'r': np.array(self._generate_fine_axis_mesh(self.r_axis)),
                'z': np.array(self._generate_fine_axis_mesh(self.z_axis)),
                'theta': np.array(self._generate_fine_axis_mesh(self.theta_axis))
            }
        elif mesh_type == "spherical":
            if not all([self.r_axis, self.theta_axis, self.phi_axis]):
                raise ValueError("Spherical mesh requires r_axis, theta_axis, and phi_axis to be defined.")
            return {
                'r': np.array(self._generate_fine_axis_mesh(self.r_axis)),
                'theta': np.array(self._generate_fine_axis_mesh(self.theta_axis)),
                'phi': np.array(self._generate_fine_axis_mesh(self.phi_axis))
            }
        else:
            raise ValueError(f"Unsupported mesh type: {mesh_type}")

    @property
    def indices(self) -> np.ndarray:
        """
        Generates a 3D array of geometry indices based on the dimensions
        (nfx, nfy, nfz) defined in the header.

        Returns:
            np.ndarray: A 3D array where each element corresponds to the geometry index
            calculated as z * (nfx * nfy) + y * nfx + x.
        """
        # Convert dimensions to integers to avoid TypeError
        nfx = int(self.header.nfx)
        nfy = int(self.header.nfy)
        nfz = int(self.header.nfz)
        
        # Create a 3D array of indices using the formula
        geom_indices = np.arange(nfx * nfy * nfz).reshape(nfz, nfy, nfx)
        return geom_indices



@dataclass
class Mesh:
    """
    Encapsulates geometry mesh, time mesh, and energy mesh.
    """
    header: Header
    geometry: GeometryData
    time_mesh: dict[int, np.ndarray] = field(default_factory=lambda: np.array([]))    # Shape: (nt,)
    energy_mesh: dict[int, np.ndarray] = field(default_factory=lambda: np.array([]))  # Shape: (ne,)

    @property
    def coarse_geometry_mesh(self) -> Dict[str, np.ndarray]:
        return self.geometry.coarse_mesh

    @property
    def fine_geometry_mesh(self) -> Dict[str, np.ndarray]:
        return self.geometry.fine_mesh

    @property
    def geometry_indices(self) -> np.ndarray:
        return self.geometry.indices

    @property
    def type_of_geometry_mesh(self) -> Optional[str]:
        """
        Returns the type of geometry mesh using the type_of_mesh property from the Header class.
        """
        return self.header.type_of_mesh


@dataclass
class ParticleBlock:
    """
    Represents a block of particles with weight window values.
    """
    ww_values: np.ndarray = field(default_factory=lambda: np.array([]))  # Shape: (nt, ne, geom_cells)


@dataclass
class QueryResult:
    """Stores the results of a weight window query."""
    header: Header
    particle_types: List[int]
    ww_values: List[np.ndarray]
    energy_intervals: List[Tuple[np.ndarray, np.ndarray]]
    time_intervals: List[Tuple[np.ndarray, np.ndarray]]
    x_intervals: Tuple[np.ndarray, np.ndarray]
    y_intervals: Tuple[np.ndarray, np.ndarray]
    z_intervals: Tuple[np.ndarray, np.ndarray]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert query results to a pandas DataFrame.
        """
        data_rows = []
        
        for p_idx, p_type in enumerate(self.particle_types):
            e_starts, e_ends = self.energy_intervals[p_idx]
            t_starts, t_ends = self.time_intervals[p_idx]
            x_starts, x_ends = self.x_intervals
            y_starts, y_ends = self.y_intervals
            z_starts, z_ends = self.z_intervals
            ww_vals = self.ww_values[p_idx]
            
            if len(t_starts) == 0 and len(t_ends) == 0:
                t_starts = np.array([0.0])
                t_ends = np.array([np.inf])
                
            # Get array shape
            time_dim = len(t_starts)
            for t_idx, (t_start, t_end) in enumerate(zip(t_starts, t_ends)):
                for e_idx, (e_start, e_end) in enumerate(zip(e_starts, e_ends)):
                    for z_idx, (z_start, z_end) in enumerate(zip(z_starts, z_ends)):
                        for y_idx, (y_start, y_end) in enumerate(zip(y_starts, y_ends)):
                            for x_idx, (x_start, x_end) in enumerate(zip(x_starts, x_ends)):
                                t_index = t_idx if time_dim > 1 else 0
                                data_rows.append({
                                    'particle_type': p_type,
                                    'time_start': t_start,
                                    'time_end': t_end,
                                    'energy_start': e_start,
                                    'energy_end': e_end,
                                    'x_start': x_start,
                                    'x_end': x_end,
                                    'y_start': y_start,
                                    'y_end': y_end,
                                    'z_start': z_start,
                                    'z_end': z_end,
                                    'ww_value': float(ww_vals[t_index, e_idx, z_idx, y_idx, x_idx])
                                })
        
        return pd.DataFrame(data_rows)


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
        z: Optional[Union[float, Tuple[float, float]]] = None,
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


@dataclass
class WWINPData:
    """
    Top-level container combining everything, now with Mesh class.
    """
    header: Header
    mesh: Mesh
    values: WeightWindowValues


