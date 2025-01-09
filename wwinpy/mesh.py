"""
This module provides the Mesh class for handling geometric, time, and energy meshes in Monte Carlo simulations.
The module implements data structures for managing multi-dimensional mesh configurations and their properties.
"""

# wwinpy/mesh.py

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from wwinpy.geometry import GeometryData
from wwinpy.header import Header

@dataclass
class Mesh:
    """A class that encapsulates geometry, time, and energy mesh data for Monte Carlo simulations.

    This class serves as a container for all mesh-related data, including geometric meshes
    (both coarse and fine), time meshes, and energy meshes. It provides properties to access
    different aspects of the mesh configuration.

    :param header: Configuration header containing mesh specifications
    :type header: Header
    :param geometry: Geometric data containing mesh coordinates and properties
    :type geometry: GeometryData
    :param time_mesh: Dictionary mapping particle types to time mesh arrays
    :type time_mesh: dict[int, np.ndarray]
    :param energy_mesh: Dictionary mapping particle types to energy mesh arrays
    :type energy_mesh: dict[int, np.ndarray]
    """
    header: Header
    geometry: GeometryData
    time_mesh: dict[int, np.ndarray] = field(default_factory=lambda: np.array([]))    # Shape: (nt,)
    energy_mesh: dict[int, np.ndarray] = field(default_factory=lambda: np.array([]))  # Shape: (ne,)

    @property
    def coarse_geometry_mesh(self) -> Dict[str, np.ndarray]:
        """
        Get the coarse geometry mesh coordinates.

        :return: Dictionary containing coarse mesh coordinates for each dimension
        :rtype: Dict[str, np.ndarray]
        """
        return self.geometry.coarse_mesh

    @property
    def fine_geometry_mesh(self) -> Dict[str, np.ndarray]:
        """
        Get the fine geometry mesh coordinates.

        :return: Dictionary containing fine mesh coordinates for each dimension
        :rtype: Dict[str, np.ndarray]
        """
        return self.geometry.fine_mesh

    @property
    def geometry_indices(self) -> np.ndarray:
        """
        Get the 3D array of geometry indices.

        :return: 3D array of geometry indices
        :rtype: np.ndarray
        """
        return self.geometry.indices

    @property
    def type_of_geometry_mesh(self) -> Optional[str]:
        """
        Get the type of geometry mesh (cartesian, cylindrical, or spherical).

        :return: String indicating the geometry mesh type, or None if not defined
        :rtype: Optional[str]
        """
        return self.header.type_of_mesh
