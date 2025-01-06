# wwpy/models.py

from wwpy.utils import get_closest_indices, get_range_indices, get_bin_intervals_from_indices
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple
import pandas as pd
import numpy as np

@dataclass
class Header:
    # Example fields derived from the specification in Table A.1
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
    """
    Encapsulates the result of a weight window query and provides methods to manipulate it.
    """
    particle_types: np.ndarray
    time_bin_start: np.ndarray
    time_bin_end: np.ndarray
    energy_bin_start: np.ndarray
    energy_bin_end: np.ndarray
    x_start: np.ndarray
    x_end: np.ndarray
    y_start: np.ndarray
    y_end: np.ndarray
    z_start: np.ndarray
    z_end: np.ndarray
    geom_index: np.ndarray
    ww_value: np.ndarray


    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the query result to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the specified columns.
        """
        data = {
            "particle_type": self.particle_types,
            "time_bin_start": self.time_bin_start,
            "time_bin_end": self.time_bin_end,
            "energy_bin_start": self.energy_bin_start,
            "energy_bin_end": self.energy_bin_end,
            "x_start": self.x_start,
            "x_end": self.x_end,
            "y_start": self.y_start,
            "y_end": self.y_end,
            "z_start": self.z_start,
            "z_end": self.z_end,
            "geom_index": self.geom_index,
            "ww_value": self.ww_value
        }
        return pd.DataFrame(data)


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
        Efficiently query ww_values and return a QueryResult object.

        Args:
            particle_type (Optional[int]): Specific particle type to query.
            time (Optional[Union[float, Tuple[float, float]]]): Time value or range.
            energy (Optional[Union[float, Tuple[float, float]]]): Energy value or range.
            x (Optional[Union[float, Tuple[float, float]]]): X coordinate or range.
            y (Optional[Union[float, Tuple[float, float]]]): Y coordinate or range.
            z (Optional[Union[float, Tuple[float, float]]]): Z coordinate or range.
            geom_idx (Optional[Union[int, Tuple[int, int]]]): Geometry index or range.

        Returns:
            QueryResult: An object containing the query results.
        """
        # Enforce mutual exclusivity between spatial queries and geom_idx
        spatial_queries = any(param is not None for param in [x, y, z])
        if spatial_queries and geom_idx is not None:
            raise ValueError("Cannot query with x, y, or z simultaneously with geom_idx.")

        if self.header is None or self.mesh is None:
            raise ValueError("Header and Mesh must be provided for querying.")

        # Handle particle_type filtering
        if particle_type is not None:
            if particle_type < 0 or particle_type >= len(self.particles):
                raise IndexError(f"particle_type {particle_type} is out of range.")
            particles = [self.particles[particle_type]]
            particle_indices = [particle_type]
        else:
            particles = self.particles
            particle_indices = list(range(len(self.particles)))

        # Initialize lists to collect query results
        particle_types, time_bin_start, time_bin_end = [], [], []
        energy_bin_start, energy_bin_end = [], []
        x_start, x_end, y_start, y_end, z_start, z_end = [], [], [], [], [], []
        geom_index, ww_values = [], []

        # Precompute allowed geom_indices if spatial queries are used
        if spatial_queries and geom_idx is None:
            # Determine x indices
            if x is not None:
                if isinstance(x, tuple):
                    x_bounds = get_range_indices(self.mesh.x_grid, x)
                else:
                    x_bounds = get_closest_indices(self.mesh.x_grid, x)
                x_start_val, x_end_val = get_bin_intervals_from_indices(self.mesh.x_grid, x_bounds)
                x_indices = np.arange(x_bounds[0], x_bounds[1] + 1)
            else:
                x_start_val, x_end_val = self.mesh.x_grid[0], self.mesh.x_grid[-1]
                x_indices = np.arange(len(self.mesh.x_grid))

            # Determine y indices
            if y is not None:
                if isinstance(y, tuple):
                    y_bounds = get_range_indices(self.mesh.y_grid, y)
                else:
                    y_bounds = get_closest_indices(self.mesh.y_grid, y)
                y_start_val, y_end_val = get_bin_intervals_from_indices(self.mesh.y_grid, y_bounds)
                y_indices = np.arange(y_bounds[0], y_bounds[1] + 1)
            else:
                y_start_val, y_end_val = self.mesh.y_grid[0], self.mesh.y_grid[-1]
                y_indices = np.arange(len(self.mesh.y_grid))

            # Determine z indices
            if z is not None:
                if isinstance(z, tuple):
                    z_bounds = get_range_indices(self.mesh.z_grid, z)
                else:
                    z_bounds = get_closest_indices(self.mesh.z_grid, z)
                z_start_val, z_end_val = get_bin_intervals_from_indices(self.mesh.z_grid, z_bounds)
                z_indices = np.arange(z_bounds[0], z_bounds[1] + 1)
            else:
                z_start_val, z_end_val = self.mesh.z_grid[0], self.mesh.z_grid[-1]
                z_indices = np.arange(len(self.mesh.z_grid))

            # Compute all possible geom_indices from the specified x, y, z ranges
            # Using meshgrid to create all combinations
            zz, yy, xx = np.meshgrid(z_indices, y_indices, x_indices, indexing='ij')
            allowed_geom_indices = (zz * (self.header.nfx * self.header.nfy) + yy * self.header.nfx + xx).flatten()
            allowed_geom_indices_set = set(allowed_geom_indices)
        else:
            # If geom_idx is specified or no spatial queries, we don't precompute allowed_geom_indices
            allowed_geom_indices = None

        for p_idx, particle in zip(particle_indices, particles):
            # Determine geom_indices based on geom_idx or spatial queries
            if geom_idx is not None:
                # Query based on geom_idx
                if isinstance(geom_idx, int):
                    geom_mask = (particle.geom_indices == geom_idx)
                else:
                    start_idx, end_idx = geom_idx
                    geom_mask = (particle.geom_indices >= start_idx) & (particle.geom_indices <= end_idx)
            elif spatial_queries:
                # Query based on spatial coordinates
                # Use the precomputed allowed_geom_indices to filter
                geom_mask = np.isin(particle.geom_indices, allowed_geom_indices)
            else:
                # No geom filtering
                geom_mask = np.ones_like(particle.geom_indices, dtype=bool)

            # Apply geom_mask to filter geom_indices and ww_values
            filtered_indices = np.where(geom_mask)[0]
            if filtered_indices.size == 0:
                continue  # No matching entries for this particle

            selected_geom_indices = particle.geom_indices[filtered_indices]
            selected_ww_values = particle.ww_values[filtered_indices]

            # Handle time filtering
            if time is not None:
                if isinstance(time, tuple):
                    time_bounds = get_range_indices(self.header.time_grid, time)
                else:
                    time_bounds = get_closest_indices(self.header.time_grid, time)
                time_start, time_end = get_bin_intervals_from_indices(self.header.time_grid, time_bounds)
            else:
                time_start, time_end = self.header.time_grid[0], self.header.time_grid[-1]

            # Handle energy filtering
            if energy is not None:
                if isinstance(energy, tuple):
                    energy_bounds = get_range_indices(self.header.energy_grid, energy)
                else:
                    energy_bounds = get_closest_indices(self.header.energy_grid, energy)
                energy_start, energy_end = get_bin_intervals_from_indices(self.header.energy_grid, energy_bounds)
            else:
                energy_start, energy_end = self.header.energy_grid[0], self.header.energy_grid[-1]

            # If querying based on geom_idx, set spatial bin starts/ends to entire range
            if geom_idx is not None or not spatial_queries:
                current_x_start, current_x_end = self.mesh.x_grid[0], self.mesh.x_grid[-1]
                current_y_start, current_y_end = self.mesh.y_grid[0], self.mesh.y_grid[-1]
                current_z_start, current_z_end = self.mesh.z_grid[0], self.mesh.z_grid[-1]
            else:
                # Spatial queries are used; use the computed bin starts/ends
                current_x_start, current_x_end = x_start_val, x_end_val
                current_y_start, current_y_end = y_start_val, y_end_val
                current_z_start, current_z_end = z_start_val, z_end_val

            # Append data to result lists
            particle_types.append(np.full(selected_ww_values.shape, p_idx, dtype=int))
            time_bin_start.append(np.full(selected_ww_values.shape, time_start, dtype=float))
            time_bin_end.append(np.full(selected_ww_values.shape, time_end, dtype=float))
            energy_bin_start.append(np.full(selected_ww_values.shape, energy_start, dtype=float))
            energy_bin_end.append(np.full(selected_ww_values.shape, energy_end, dtype=float))
            x_start.append(np.full(selected_ww_values.shape, current_x_start, dtype=float))
            x_end.append(np.full(selected_ww_values.shape, current_x_end, dtype=float))
            y_start.append(np.full(selected_ww_values.shape, current_y_start, dtype=float))
            y_end.append(np.full(selected_ww_values.shape, current_y_end, dtype=float))
            z_start.append(np.full(selected_ww_values.shape, current_z_start, dtype=float))
            z_end.append(np.full(selected_ww_values.shape, current_z_end, dtype=float))
            geom_index.append(selected_geom_indices)
            ww_values.append(selected_ww_values)

        # Finalize results
        if ww_values:
            particle_types = np.concatenate(particle_types)
            time_bin_start = np.concatenate(time_bin_start)
            time_bin_end = np.concatenate(time_bin_end)
            energy_bin_start = np.concatenate(energy_bin_start)
            energy_bin_end = np.concatenate(energy_bin_end)
            x_start = np.concatenate(x_start)
            x_end = np.concatenate(x_end)
            y_start = np.concatenate(y_start)
            y_end = np.concatenate(y_end)
            z_start = np.concatenate(z_start)
            z_end = np.concatenate(z_end)
            geom_index = np.concatenate(geom_index)
            ww_value = np.concatenate(ww_values)
        else:
            # Initialize empty arrays with correct dtype
            particle_types = np.array([], dtype=int)
            time_bin_start = np.array([], dtype=float)
            time_bin_end = np.array([], dtype=float)
            energy_bin_start = np.array([], dtype=float)
            energy_bin_end = np.array([], dtype=float)
            x_start = np.array([], dtype=float)
            x_end = np.array([], dtype=float)
            y_start = np.array([], dtype=float)
            y_end = np.array([], dtype=float)
            z_start = np.array([], dtype=float)
            z_end = np.array([], dtype=float)
            geom_index = np.array([], dtype=int)
            ww_value = np.array([], dtype=float)

        return QueryResult(
            particle_types=particle_types,
            time_bin_start=time_bin_start,
            time_bin_end=time_bin_end,
            energy_bin_start=energy_bin_start,
            energy_bin_end=energy_bin_end,
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end,
            z_start=z_start,
            z_end=z_end,
            geom_index=geom_index,
            ww_value=ww_value
        )


@dataclass
class WWINPData:
    """
    Top-level container combining everything, now with Mesh class.
    """
    header: Header
    mesh: Mesh
    values: WeightWindowValues


