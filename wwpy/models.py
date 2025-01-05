# wwpy/models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple
import pandas as pd
import numpy as np
 
@dataclass
class Header:
    # Example fields derived from the specification in Table A.1
    if_: int
    iv: int
    ni: int
    nr: int
    probid: str = ""  # Made optional with default empty string

    # Optional arrays that might appear depending on 'iv' or 'nr'
    nt: List[int] = field(default_factory=list)
    ne: List[int] = field(default_factory=list)

    # Additional geometry specs
    nfx: Optional[float] = None
    nfy: Optional[float] = None
    nfz: Optional[float] = None
    x0:  Optional[float] = None
    y0:  Optional[float] = None
    z0:  Optional[float] = None

    ncx: Optional[float] = None
    ncy: Optional[float] = None
    ncz: Optional[float] = None
    nwg: Optional[float] = None
    
    # Optional for nr = 16
    x1:  Optional[float] = None
    y1:  Optional[float] = None
    z1:  Optional[float] = None
    x2:  Optional[float] = None
    y2:  Optional[float] = None
    z2:  Optional[float] = None

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
        return mesh_types.get(int(self.nwg), "unknown")
    
    @property
    def number_of_time_bins(self) -> int:
        """
        Returns the number of time bins.
        """
        return len(self.nt)
    
    @property
    def number_of_energy_bins(self) -> int:
        """
        Returns the number of energy bins.
        """
        return len(self.ne)
    
    @property
    def number_of_particle_types(self) -> int:
        """
        Returns the number of particles.
        """
        return self.ni


@dataclass
class CoarseMeshSegment:
    """
    Represents a single coarse-mesh segment along one axis,
    storing q, p, and s for that segment.
    """
    q: float  # Fine mesh ratio (presently = 1 always)
    p: float  # Coarse mesh coordinate
    s: float  # Number of fine meshes in each coarse mesh

    def __post_init__(self):
        if not isinstance(self.s, float):
            raise TypeError(f"'s' must be a float, got {type(self.s).__name__}")
        if not self.s.is_integer():
            raise ValueError(f"'s' must be a whole number (float), got {self.s}")
        self.s = int(self.s)  # Convert to integer for internal use


@dataclass
class GeometryAxis:
    """
    Represents all geometry data along a single axis (x, y, or z).
    """
    origin: float  # e.g., x0, y0, or z0
    segments: List[CoarseMeshSegment] = field(default_factory=list)


@dataclass
class GeometryData:
    """
    Holds geometry info for x, y, z axes using structured data.
    """
    x_axis: GeometryAxis
    y_axis: GeometryAxis
    z_axis: GeometryAxis


@dataclass
class ParticleBlock:
    """
    Time/Energy/Weight data for one particle.
    """
    time_bins: Optional[np.ndarray] = None  # Shape: (nt,)
    energy_bins: Optional[np.ndarray] = None  # Shape: (ne,)
    # w_values: shape (nt, ne, geom_cells)
    w_values: Optional[np.ndarray] = None  # dtype: float32 or float64

@dataclass
class WeightWindowValues:
    """
    Stores the data from Block 3 for all particles.
    """
    particles: List[ParticleBlock] = field(default_factory=list)


@dataclass
class WWINPData:
    """
    Top-level container combining everything.
    """
    header: Header
    geometry: GeometryData
    values: WeightWindowValues

    @property
    def energy_mesh(self) -> Dict[int, List[float]]:
        """
        Returns a dictionary mapping each particle type to its energy mesh.
        """
        energy_mesh_dict = {
            index: particle.energy_bins 
            for index, particle in enumerate(self.values.particles)
        }
        return energy_mesh_dict

    @property
    def time_mesh(self) -> Dict[int, List[float]]:
        """
        Returns a dictionary mapping each particle type to its time mesh.
        """
        time_mesh_dict = {
            index: particle.time_bins 
            for index, particle in enumerate(self.values.particles)
        }
        return time_mesh_dict

    @property
    def geometry_coarse_mesh(self) -> Dict[str, List[float]]:
        mesh_type = self.header.type_of_mesh
        if mesh_type == "cartesian":
            if not all([self.geometry.x_axis, self.geometry.y_axis, self.geometry.z_axis]):
                raise ValueError("Cartesian mesh requires x_axis, y_axis, and z_axis to be defined.")
            return {
                'x': self._generate_coarse_axis_mesh(self.geometry.x_axis),
                'y': self._generate_coarse_axis_mesh(self.geometry.y_axis),
                'z': self._generate_coarse_axis_mesh(self.geometry.z_axis)
            }
        elif mesh_type == "cylindrical":
            if not all([self.geometry.r_axis, self.geometry.z_axis, self.geometry.theta_axis]):
                raise ValueError("Cylindrical mesh requires r_axis, z_axis, and theta_axis to be defined.")
            return {
                'r': self._generate_coarse_axis_mesh(self.geometry.r_axis),
                'z': self._generate_coarse_axis_mesh(self.geometry.z_axis),
                'theta': self._generate_coarse_axis_mesh(self.geometry.theta_axis)
            }
        elif mesh_type == "spherical":
            if not all([self.geometry.r_axis, self.geometry.theta_axis, self.geometry.phi_axis]):
                raise ValueError("Spherical mesh requires r_axis, theta_axis, and phi_axis to be defined.")
            return {
                'r': self._generate_coarse_axis_mesh(self.geometry.r_axis),
                'theta': self._generate_coarse_axis_mesh(self.geometry.theta_axis),
                'phi': self._generate_coarse_axis_mesh(self.geometry.phi_axis)
            }
        else:
            raise ValueError(f"Unsupported mesh type: {mesh_type}")

    @property
    def geometry_fine_mesh(self) -> Dict[str, List[float]]:
        mesh_type = self.header.type_of_mesh
        if mesh_type == "cartesian":
            return {
                'x': self._generate_fine_axis_mesh(self.geometry.x_axis),
                'y': self._generate_fine_axis_mesh(self.geometry.y_axis),
                'z': self._generate_fine_axis_mesh(self.geometry.z_axis)
            }
        elif mesh_type == "cylindrical":
            return {
                'r': self._generate_fine_axis_mesh(self.geometry.r_axis),
                'z': self._generate_fine_axis_mesh(self.geometry.z_axis),
                'theta': self._generate_fine_axis_mesh(self.geometry.theta_axis)
            }
        elif mesh_type == "spherical":
            return {
                'r': self._generate_fine_axis_mesh(self.geometry.r_axis),
                'theta': self._generate_fine_axis_mesh(self.geometry.theta_axis),
                'phi': self._generate_fine_axis_mesh(self.geometry.phi_axis)
            }
        else:
            raise ValueError(f"Unsupported mesh type: {mesh_type}")

    def _generate_coarse_axis_mesh(self, axis: GeometryAxis) -> List[float]:
        mesh = [axis.origin]
        for segment in axis.segments:
            mesh.append(segment.p)
        return mesh


    def _generate_fine_axis_mesh(self, axis: GeometryAxis) -> List[float]:
        fine_mesh = [axis.origin]
        current = axis.origin
        for segment in axis.segments:
            target = segment.p
            step = (target - current) / segment.s
            for _ in range(segment.s):
                current += step
                fine_mesh.append(current)
        return fine_mesh

    @property
    def _w_values_dataframe(self) -> pd.DataFrame:
        if not hasattr(self, '_cached_w_values_df'):
            data = []
            
            # Determine mesh type and get corresponding fine meshes
            mesh_type = self.header.type_of_mesh
            if mesh_type == "cartesian":
                x_mesh = self.geometry_fine_mesh['x']
                y_mesh = self.geometry_fine_mesh['y']
                z_mesh = self.geometry_fine_mesh['z']
            elif mesh_type == "cylindrical":
                r_mesh = self.geometry_fine_mesh['r']
                z_mesh = self.geometry_fine_mesh['z']
                theta_mesh = self.geometry_fine_mesh['theta']
                # Depending on application, you might need to handle cylindrical coordinates differently
                # For simplicity, we'll proceed similarly to cartesian
                x_mesh = r_mesh  # Placeholder
                y_mesh = z_mesh  # Placeholder
                z_mesh = theta_mesh  # Placeholder
            elif mesh_type == "spherical":
                r_mesh = self.geometry_fine_mesh['r']
                theta_mesh = self.geometry_fine_mesh['theta']
                phi_mesh = self.geometry_fine_mesh['phi']
                # Similarly, handle spherical coordinates as needed
                x_mesh = r_mesh  # Placeholder
                y_mesh = theta_mesh  # Placeholder
                z_mesh = phi_mesh  # Placeholder
            else:
                raise ValueError(f"Unsupported mesh type: {mesh_type}")

            # Get dimensions
            nx = len(x_mesh) - 1
            ny = len(y_mesh) - 1
            nz = len(z_mesh) - 1

            for p_idx, particle in enumerate(self.values.particles):
                time_indices = range(len(particle.w_values))
                
                for t_idx in time_indices:
                    time = particle.time_bins[t_idx] if particle.time_bins else 0.0
                    
                    for e_idx, energy in enumerate(particle.energy_bins):
                        w_values_slice = particle.w_values[t_idx][e_idx]
                        
                        # Loop through flattened array
                        for flat_idx in range(len(w_values_slice)):
                            # Convert flat index back to 3D coordinates
                            ix = flat_idx // (ny * nz)
                            iy = (flat_idx % (ny * nz)) // nz
                            iz = flat_idx % nz
                            
                            data.append({
                                'particle_type': p_idx,
                                'time': time,
                                'energy': energy,
                                'x': x_mesh[ix],
                                'y': y_mesh[iy],
                                'z': z_mesh[iz],
                                'x_next': x_mesh[ix + 1],
                                'y_next': y_mesh[iy + 1],
                                'z_next': z_mesh[iz + 1],
                                'geom_index': flat_idx,
                                'w_value': w_values_slice[flat_idx]
                            })

            self._cached_w_values_df = pd.DataFrame(data)
        
        return self._cached_w_values_df

    def _find_bounds(
        self, 
        mesh: List[float], 
        value: Union[float, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Find the lower and upper bounds in the mesh based on the input value or range.

        Parameters:
            mesh (List[float]): Sorted list of mesh points.
            value (Union[float, Tuple[float, float]]): Single value or (min, max) tuple.

        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        if isinstance(value, tuple):
            min_val, max_val = value
            # Find the closest mesh point <= min_val
            lower = max([m for m in mesh if m <= min_val], default=mesh[0])
            # Find the closest mesh point >= max_val
            upper = min([m for m in mesh if m >= max_val], default=mesh[-1])
        else:
            if value in mesh:
                lower = upper = value
            else:
                # Find the closest mesh point <= value
                lower_candidates = [m for m in mesh if m <= value]
                lower = max(lower_candidates, default=mesh[0])
                # Find the closest mesh point > value
                upper_candidates = [m for m in mesh if m > value]
                upper = min(upper_candidates, default=mesh[-1])
        return (lower, upper)

    def _get_fine_axis_mesh(self, coord: str) -> List[float]:
        """
        Retrieve the fine mesh for a given spatial coordinate.

        Parameters:
            coord (str): One of 'x', 'y', or 'z'.

        Returns:
            List[float]: The fine mesh list for the specified coordinate.
        """
        if coord not in self.geometry_fine_mesh:
            raise ValueError(f"Invalid coordinate: {coord}")
        return self.geometry_fine_mesh[coord]

    def query_w_values(
        self, 
        particle_type: Optional[int] = None,
        time: Optional[Union[float, Tuple[float, float]]] = None,
        energy: Optional[Union[float, Tuple[float, float]]] = None,
        x: Optional[Union[float, Tuple[float, float]]] = None,
        y: Optional[Union[float, Tuple[float, float]]] = None,
        z: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> pd.DataFrame:
        """
        Query the w_values based on provided criteria. Supports both single values and ranges.
        For single values, it will find the interval containing that value.

        Parameters:
            particle_type (Optional[int]): Index of the particle type.
            time (Optional[float | tuple]): Single time value or (min_time, max_time) range.
            energy (Optional[float | tuple]): Single energy value or (min_energy, max_energy) range.
            x (Optional[float | tuple]): Single x value or (min_x, max_x) range.
            y (Optional[float | tuple]): Single y value or (min_y, max_y) range.
            z (Optional[float | tuple]): Single z value or (min_z, max_z) range.

        Returns:
            pd.DataFrame: Filtered DataFrame containing the matching w_values.
        """
        df = self._w_values_dataframe.copy()

        # Filter by particle_type
        if particle_type is not None:
            df = df[df['particle_type'] == particle_type]
            if df.empty:
                return df  # Early exit if no matching particle_type

        # Retrieve mesh points for time and energy based on particle_type
        if particle_type is not None:
            time_mesh = self.time_mesh.get(particle_type, [])
            energy_mesh = self.energy_mesh.get(particle_type, [])
        else:
            # If particle_type is not specified, gather all unique time and energy mesh points
            time_mesh = sorted(set(df['time']))
            energy_mesh = sorted(set(df['energy']))

        # Apply time filter
        if time is not None:
            lower, upper = self._find_bounds(time_mesh, time)
            if lower == upper:
                df = df[df['time'] == lower]
            else:
                df = df[(df['time'] >= lower) & (df['time'] <= upper)]
        
        # Apply energy filter
        if energy is not None:
            lower, upper = self._find_bounds(energy_mesh, energy)
            if lower == upper:
                df = df[df['energy'] == lower]
            else:
                df = df[(df['energy'] >= lower) & (df['energy'] <= upper)]

        # Apply spatial filters for x, y, z
        for coord, value in zip(['x', 'y', 'z'], [x, y, z]):
            if value is not None:
                # Extract the corresponding mesh for the spatial axis
                axis_mesh = self._get_fine_axis_mesh(coord)
                lower, upper = self._find_bounds(axis_mesh, value)
                if lower == upper:
                    # Exact match: filter rows where axis <= value < axis_next
                    df = df[
                        (df[coord] <= lower) & 
                        (df[f'{coord}_next'] > lower)
                    ]
                else:
                    # Range: filter rows where axis >= lower and axis <= upper
                    df = df[
                        (df[coord] >= lower) & 
                        (df[coord] <= upper)
                    ]

        return df

    def get_w_values_dataframe(self) -> pd.DataFrame:
        """
        Returns the entire w_values as a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all w_values with columns:
                          ['particle_type', 'time', 'energy', 'x', 'y', 'z', 'x_next', 'y_next', 'z_next', 'geometry_index', 'w_value']
        """
        return self._w_values_dataframe.copy()