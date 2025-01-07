# wwpy/models.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
from wwpy.geometry import GeometryData, GeometryAxis
from wwpy.weight_windows import WeightWindowValues

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
class WWINPData:
    """
    Top-level container combining everything, now with Mesh class.
    """
    header: Header
    mesh: Mesh
    values: WeightWindowValues

    def multiply(self, factor: float) -> None:
        """
        Multiply all weight window values by a given factor.

        Parameters:
            factor (float): The multiplication factor to apply to all weight window values.
        """
        for particle in self.values.particles:
            particle.ww_values *= factor

    def soften(self, power: float) -> None:
        """
        Raise all weight window values to a given power.
        This can be used to "soften" or "harden" the weight window boundaries.
        
        Parameters:
            power (float): The exponent to apply to all weight window values.
                          Values < 1 will soften the boundaries
                          Values > 1 will harden the boundaries
        """
        for particle in self.values.particles:
            particle.ww_values = np.power(particle.ww_values, power)

    def write_file(self, filename: str) -> None:
        """
        Write the WWINP data to a file using FORTRAN-style formatting.
        
        Parameters:
            filename (str): Path to the output file
        """
        with open(filename, 'w') as f:
            # First line: if iv ni nr probid (4i10, 20x, a19)
            f.write(f"{self.header.if_:10d}{self.header.iv:10d}{self.header.ni:10d}{self.header.nr:10d}" + 
                   " " * 20 + f"{self.header.probid:19s}\n")
            
            # Time bins if iv == 2 (7i10)
            if self.header.has_time_dependency:
                line = ""
                for nt_val in self.header.nt:
                    line += f"{nt_val:10d}"
                f.write(line + "\n")
            
            # Energy bins (7i10)
            line = ""
            for ne_val in self.header.ne:
                line += f"{ne_val:10d}"
            f.write(line + "\n")
            
            # Mesh dimensions and origins (6g13.5)
            f.write(f"{self.header.nfx:13.5e}{self.header.nfy:13.5e}{self.header.nfz:13.5e}"
                   f"{self.header.x0:13.5e}{self.header.y0:13.5e}{self.header.z0:13.5e}\n")
            
            # Geometry-specific parameters (6g13.5)
            if self.header.nr == 10:  # Rectangular
                f.write(f"{self.header.ncx:13.5e}{self.header.ncy:13.5e}{self.header.ncz:13.5e}"
                       f"{self.header.nwg:13.5e}\n")
            elif self.header.nr == 16:  # Cylindrical/Spherical
                f.write(f"{self.header.ncx:13.5e}{self.header.ncy:13.5e}{self.header.ncz:13.5e}"
                       f"{self.header.x1:13.5e}{self.header.y1:13.5e}{self.header.z1:13.5e}\n")
                f.write(f"{self.header.x2:13.5e}{self.header.y2:13.5e}{self.header.z2:13.5e}"
                       f"{self.header.nwg:13.5e}\n")
            
            # Write mesh data for each axis (6g13.5)
            mesh_type = self.header.type_of_mesh
            if mesh_type == "cartesian":
                self._write_axis_data(f, self.mesh.geometry.x_axis)
                self._write_axis_data(f, self.mesh.geometry.y_axis)
                self._write_axis_data(f, self.mesh.geometry.z_axis)
            elif mesh_type == "cylindrical":
                self._write_axis_data(f, self.mesh.geometry.r_axis)
                self._write_axis_data(f, self.mesh.geometry.z_axis)
                self._write_axis_data(f, self.mesh.geometry.theta_axis)
            elif mesh_type == "spherical":
                self._write_axis_data(f, self.mesh.geometry.r_axis)
                self._write_axis_data(f, self.mesh.geometry.theta_axis)
                self._write_axis_data(f, self.mesh.geometry.phi_axis)
            
            # Write time and energy meshes for each particle
            for i in range(self.header.ni):
                # Write time mesh if time-dependent (6g13.5)
                if self.header.has_time_dependency and len(self.mesh.time_mesh[i]) > 1:
                    self._write_array(f, self.mesh.time_mesh[i])
                # Write energy mesh (6g13.5)
                self._write_array(f, self.mesh.energy_mesh[i])
                
            # Write weight window values (6g13.5)
            for particle in self.values.particles:
                ww = particle.ww_values
                if self.header.has_time_dependency:
                    for t in range(ww.shape[0]):  # For each time bin
                        for e in range(ww.shape[1]):  # For each energy bin
                            values = ww[t, e, :].flatten()
                            self._write_ww_block(f, values, e < ww.shape[1]-1 or t < ww.shape[0]-1)
                else:
                    for e in range(ww.shape[1]):  # For each energy bin
                        values = ww[0, e, :].flatten()
                        self._write_ww_block(f, values, e < ww.shape[1]-1)

    def _write_ww_block(self, f, values: np.ndarray, add_newline: bool) -> None:
        """Helper method to write a block of weight window values."""
        for i in range(0, len(values), 6):
            chunk = values[i:i+6]
            line = "".join(f"{value:13.5e}" for value in chunk)
            f.write(line + "\n")

    def _write_array(self, f, array: np.ndarray) -> None:
        """Helper method to write non-WW arrays in 6g13.5 format."""
        values = array.flatten()
        for i in range(0, len(values), 6):
            chunk = values[i:i+6]
            line = "".join(f"{value:13.5e}" for value in chunk)
            f.write(line + "\n")

    def _write_axis_data(self, f, axis: GeometryAxis) -> None:
        """Helper method to write axis data in 6g13.5 format."""
        values = [axis.origin]  # Start with origin
        
        for q, p, s in zip(axis.q, axis.p, axis.s):
            values.extend([q, p, s])  # Add the triplet to values list
            
        # Write in chunks of 6
        for i in range(0, len(values), 6):
            chunk = values[i:i+6]
            line = "".join(f"{value:13.5e}" for value in chunk)
            f.write(line + "\n")


