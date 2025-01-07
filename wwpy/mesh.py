# wwpy/mesh.py

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from wwpy.geometry import GeometryData
from wwpy.header import Header

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
