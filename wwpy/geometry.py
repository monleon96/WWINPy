from dataclasses import dataclass, field
from wwpy.models import Header
from typing import List, Optional, Dict
import pandas as pd
import numpy as np


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
