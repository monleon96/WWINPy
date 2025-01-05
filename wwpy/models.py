# wwpy/models.py

from wwpy.utils import get_closest_indices, get_range_indices
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
    q: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    p: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    s: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

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



@dataclass
class Mesh:
    """
    Encapsulates geometry mesh, time mesh, and energy mesh.
    """
    header: Header
    geometry: GeometryData
    time_mesh: dict[int, np.ndarray] = field(default_factory=lambda: np.array([], dtype=np.float32))    # Shape: (nt,)
    energy_mesh: dict[int, np.ndarray] = field(default_factory=lambda: np.array([], dtype=np.float32))  # Shape: (ne,)

    @property
    def coarse_geometry_mesh(self) -> Dict[str, np.ndarray]:
        return self.geometry.coarse_mesh

    @property
    def fine_geometry_mesh(self) -> Dict[str, np.ndarray]:
        return self.geometry.fine_mesh
    
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
    ww_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))  # Shape: (nt, ne, geom_cells)


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
    ) -> np.ndarray:
        """
        Efficiently query ww_values using NumPy's advanced indexing.

        Returns:
            np.ndarray: Array of ww_values that match the query criteria.
        """
        if self.header is None or self.mesh is None:
            raise ValueError("Header and Mesh must be provided for querying.")

        if particle_type is not None:
            if particle_type < 0 or particle_type >= len(self.particles):
                raise IndexError(f"particle_type {particle_type} is out of range.")
            particles = [self.particles[particle_type]]
        else:
            particles = self.particles

        results = []
        for p_idx, particle in enumerate(particles):
            # Time filter
            if time is not None and self.mesh.time_mesh is not None:
                time_bins = self.mesh.time_mesh.get(p_idx)
                if time_bins is None:
                    continue
                if isinstance(time, tuple):
                    time_indices = get_range_indices(time_bins, time)
                else:
                    time_indices = get_closest_indices(time_bins, time)
            else:
                time_indices = np.arange(particle.ww_values.shape[0])

            if len(time_indices) == 0:
                continue  # No matching time bins

            # Energy filter
            if energy is not None:
                energy_bins = self.mesh.energy_mesh.get(p_idx)
                if energy_bins is None:
                    continue
                if isinstance(energy, tuple):
                    energy_indices = get_range_indices(energy_bins, energy)
                else:
                    energy_indices = get_closest_indices(energy_bins, energy)
            else:
                energy_indices = np.arange(particle.ww_values.shape[1])

            if len(energy_indices) == 0:
                continue  # No matching energy bins

            # Spatial filters
            geom_indices = np.arange(particle.ww_values.shape[2])
            if any([x is not None, y is not None, z is not None]):
                # Assuming Cartesian coordinates; adjust based on mesh type
                mesh_type = self.header.type_of_mesh
                if mesh_type == "cartesian":
                    x_mesh = self.mesh.fine_geometry_mesh['x']
                    y_mesh = self.mesh.fine_geometry_mesh['y']
                    z_mesh = self.mesh.fine_geometry_mesh['z']
                elif mesh_type == "cylindrical":
                    r_mesh = self.mesh.fine_geometry_mesh['r']
                    theta_mesh = self.mesh.fine_geometry_mesh['theta']
                    z_mesh = self.mesh.fine_geometry_mesh['z']
                elif mesh_type == "spherical":
                    r_mesh = self.mesh.fine_geometry_mesh['r']
                    theta_mesh = self.mesh.fine_geometry_mesh['theta']
                    phi_mesh = self.mesh.fine_geometry_mesh['phi']
                else:
                    raise ValueError(f"Unsupported mesh type: {mesh_type}")

                # Placeholder for actual spatial filtering logic
                # Implement based on specific requirements
                pass  # To be implemented as per application needs

            # Extract the relevant subset of ww_values
            subset = particle.ww_values[np.ix_(time_indices, energy_indices, geom_indices)]
            results.append(subset)

        if results:
            return np.concatenate(results, axis=0)
        else:
            return np.array([], dtype=np.float32)



@dataclass
class WWINPData:
    """
    Top-level container combining everything, now with Mesh class.
    """
    header: Header
    mesh: Mesh
    values: WeightWindowValues


