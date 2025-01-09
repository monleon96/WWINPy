"""
This module handles geometric data structures and operations for Cartesian, Cylindrical, and Spherical coordinate systems.
Provides classes for managing geometry axes and mesh generation in various coordinate systems.
"""

from dataclasses import dataclass, field
from wwinpy.header import Header
from typing import List, Optional, Dict
import numpy as np


@dataclass
class GeometryAxis:
    """A class representing a geometric axis with coarse and fine mesh information.

    :param origin: The starting coordinate of the axis
    :type origin: float
    :param q: Array of fine mesh ratios for each segment
    :type q: np.ndarray
    :param p: Array of coarse mesh coordinates
    :type p: np.ndarray
    :param s: Array of number of fine meshes per segment
    :type s: np.ndarray
    """

    origin: float
    q: np.ndarray = field(default_factory=lambda: np.array([]))
    p: np.ndarray = field(default_factory=lambda: np.array([]))
    s: np.ndarray = field(default_factory=lambda: np.array([]))

    def add_segment(self, q: float, p: float, s: float):
        """
        Add a new coarse mesh segment to the axis.

        :param q: Fine mesh ratio for the segment
        :type q: float
        :param p: Coarse mesh coordinate endpoint
        :type p: float
        :param s: Number of fine meshes in the segment
        :type s: float
        """
        self.q = np.append(self.q, np.float32(q))  # Explicitly cast to float32
        self.p = np.append(self.p, np.float32(p))  # Explicitly cast to float32
        self.s = np.append(self.s, np.int32(s))  # Explicitly cast to int32


@dataclass
class GeometryData:
    """A class managing geometric mesh data for different coordinate systems.

    :param header: Configuration header containing mesh type and dimensions
    :type header: Header
    :param x_axis: X-axis data for Cartesian coordinates
    :type x_axis: Optional[GeometryAxis]
    :param y_axis: Y-axis data for Cartesian coordinates
    :type y_axis: Optional[GeometryAxis]
    :param z_axis: Z-axis data for Cartesian/Cylindrical coordinates
    :type z_axis: Optional[GeometryAxis]
    :param r_axis: Radial axis data for Cylindrical/Spherical coordinates
    :type r_axis: Optional[GeometryAxis]
    :param theta_axis: Angular axis data for Cylindrical/Spherical coordinates
    :type theta_axis: Optional[GeometryAxis]
    :param phi_axis: Azimuthal axis data for Spherical coordinates
    :type phi_axis: Optional[GeometryAxis]
    """

    header: Header
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
        Generate coarse mesh points for a given axis.

        :param axis: The geometry axis to generate coarse mesh for
        :type axis: GeometryAxis
        :return: List of coarse mesh coordinates
        :rtype: List[float]
        """
        mesh = [float(axis.origin)]  # Ensure origin is a Python float
        for p in axis.p:
            mesh.append(float(p))  # Convert each point to Python float
        return mesh

    def _generate_fine_axis_mesh(self, axis: GeometryAxis) -> List[float]:
        """
        Generate fine mesh points for a given axis.

        :param axis: The geometry axis to generate fine mesh for
        :type axis: GeometryAxis
        :return: List of fine mesh coordinates
        :rtype: List[float]
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
        """
        Get the coarse mesh for the current geometry type.

        :return: Dictionary containing mesh coordinates for each dimension
        :rtype: Dict[str, np.ndarray]
        :raises ValueError: If required axes for the geometry type are not defined
        """
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
        """
        Get the fine mesh for the current geometry type.

        :return: Dictionary containing mesh coordinates for each dimension
        :rtype: Dict[str, np.ndarray]
        :raises ValueError: If required axes for the geometry type are not defined
        """
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
        Generate a 3D array of geometry indices.

        Creates indices based on the dimensions (nfx, nfy, nfz) defined in the header.
        The index at position (x,y,z) is calculated as z*(nfx*nfy) + y*nfx + x.

        :return: 3D array of geometry indices
        :rtype: np.ndarray
        """
        # Convert dimensions to integers to avoid TypeError
        nfx = int(self.header.nfx)
        nfy = int(self.header.nfy)
        nfz = int(self.header.nfz)
        
        # Create a 3D array of indices using the formula
        geom_indices = np.arange(nfx * nfy * nfz).reshape(nfz, nfy, nfx)
        return geom_indices
