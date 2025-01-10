"""
Main interface module for WWINP file manipulation.
Provides the WWData class for high-level operations on weight window data.
"""

# wwinpy/ww_data.py

from dataclasses import dataclass
from typing import Union, List
from copy import deepcopy
import numpy as np
from wwinpy.header import Header
from wwinpy.mesh import Mesh
from wwinpy.weight_windows import WeightWindowValues
from wwinpy.geometry import GeometryAxis
from wwinpy.query import QueryResult


@dataclass
class WWData:
    """Main interface for WWINP file manipulation.

    High-level interface for manipulating weight window data from WWINP files.

    :param header: Weight window file header information
    :type header: Header
    :param mesh: Mesh geometry and binning information
    :type mesh: Mesh
    :param values: Weight window values container
    :type values: WeightWindowValues
    """
    header: Header
    mesh: Mesh
    values: WeightWindowValues

    def copy(self) -> 'WWData':
        """Create a deep copy of the WWData object.
        
        :return: A new WWData instance with copied data
        :rtype: WWData
        
        :Example:
        
            >>> import wwinpy
            >>> ww = wwinpy.from_file("path/to/wwinp_file")
            >>> ww_copy = ww.copy()
        """
        return deepcopy(self)

    def multiply(self, factor: float = 2.0,
                particle_types: Union[int, List[int]] = -1) -> None:
        """Multiply weight window values by a specified factor.

        :param factor: Multiplication factor to apply
        :type factor: float
        :param particle_types: Particle type(s) to process. Use -1 for all types
        :type particle_types: Union[int, List[int]]
        :return: None

        :Example:

            >>> import wwinpy
            >>> ww = wwinpy.from_file("path/to/wwinp_file")
            >>> ww.multiply(2.0)  # Apply to all particle types
            >>> ww.multiply(2.0, particle_types=0)  # Apply only to particle type 0
        """
        self.values.multiply(factor, particle_types)

    def soften(self, power: float = 0.6,
            particle_types: Union[int, List[int]] = -1) -> None:
        """Adjust weight window values by applying a power transformation.

        :param power: Exponent to apply to values (< 1 softens, > 1 hardens)
        :type power: float
        :param particle_types: Particle type(s) to process. Use -1 for all types
        :type particle_types: Union[int, List[int]]
        :return: None

        :Example:

            >>> import wwinpy
            >>> ww = wwinpy.from_file("path/to/wwinp_file")
            >>> ww.soften(0.6)  # Apply to all particle types
            >>> ww.soften(0.6, particle_types=0)  # Apply only to particle type 0
        """
        self.values.soften(power, particle_types)

    def apply_ratio_threshold(self, threshold: float = 10.0,
                         particle_types: Union[int, List[int]] = -1,
                         verbose: bool = False) -> None:
        """Apply a ratio threshold to weight window values.

        :param threshold: Maximum allowed ratio between neighboring cells
        :type threshold: float
        :param particle_types: Particle type(s) to process. Use -1 for all types
        :type particle_types: Union[int, List[int]]
        :param verbose: If True, prints detailed information
        :type verbose: bool
        :return: None

        :Example:

            >>> import wwinpy
            >>> ww = wwinpy.from_file("path/to/wwinp_file")
            >>> ww.apply_ratio_threshold(10.0)  # Apply to all particle types
            >>> ww.apply_ratio_threshold(threshold=10.0, particle_types=0)  # Apply only to particle type 0
            >>> ww.apply_ratio_threshold(threshold=10.0, particle_types=[0, 1]) 
        """
        return self.values.apply_ratio_threshold(
            threshold=threshold,
            particle_types=particle_types,
            verbose=verbose
        )

    def query_ww(self, **kwargs) -> QueryResult:
        """Query weight window values based on specified criteria.

        :param kwargs: Query parameters for filtering weight windows
                     Valid keys: particle_type, time, energy, x, y, z
        :type kwargs: dict
        :return: Object containing queried values and metadata
        :rtype: QueryResult

        :Example:

            >>> import wwinpy
            >>> ww = wwinpy.from_file("path/to/wwinp_file")
            >>> result = wwinp.query_ww(
            ...     particle_type=0,
            ...     energy=(1.0, 10.0),
            ...     x=(0, 10),
            ...     y=0,
            ...     z=(0, 10)
            ... )
        """
        return self.values.query_ww(**kwargs)

    
    def write_file(self, filename: str) -> None:
        """Write the WWINP data to a file in FORTRAN-compatible format.

        :param filename: Path where the output file will be written
        :type filename: str
        :return: None
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
            for i in range(self.header.ni):
                ww = self.values.ww_values[i]
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
        """Write a block of weight window values in FORTRAN-compatible format.

        :param f: File object to write to
        :type f: TextIO
        :param values: Array of weight window values to write
        :type values: np.ndarray
        :param add_newline: Whether to add a newline after the block
        :type add_newline: bool
        :return: None
        """
        for i in range(0, len(values), 6):
            chunk = values[i:i+6]
            line = "".join(f"{value:13.5e}" for value in chunk)
            f.write(line + "\n")

    def _write_array(self, f, array: np.ndarray) -> None:
        """Write a generic array in FORTRAN 6g13.5 format.

        :param f: File object to write to
        :type f: TextIO
        :param array: Array of values to write
        :type array: np.ndarray
        :return: None
        """
        values = array.flatten()
        for i in range(0, len(values), 6):
            chunk = values[i:i+6]
            line = "".join(f"{value:13.5e}" for value in chunk)
            f.write(line + "\n")

    def _write_axis_data(self, f, axis: GeometryAxis) -> None:
        """Write geometry axis data in FORTRAN 6g13.5 format.

        Writes the axis origin followed by triplets of q, p, and s values
        that define the axis geometry.

        :param f: File object to write to
        :type f: TextIO
        :param axis: Geometry axis object containing the data to write
        :type axis: GeometryAxis
        :return: None
        """
        values = [axis.origin]  # Start with origin
        
        for q, p, s in zip(axis.q, axis.p, axis.s):
            values.extend([q, p, s])  # Add the triplet to values list
            
        # Write in chunks of 6
        for i in range(0, len(values), 6):
            chunk = values[i:i+6]
            line = "".join(f"{value:13.5e}" for value in chunk)
            f.write(line + "\n")


