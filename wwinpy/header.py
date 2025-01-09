"""
This module contains the Header class for Window Weight (WW) file handling.
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Header:
    """A class representing the header information for a Window Weight (WW) file.

    This class stores and manages various parameters that define the geometric mesh
    and particle properties for radiation transport calculations.

    :param if_: File type indicator (currently only 1 is supported)
    :type if_: int
    :param iv: Time-dependent windows flag (1=no, 2=yes)
    :type iv: int
    :param ni: Number of particle types
    :type ni: int
    :param nr: Mesh type indicator (10=rectangular, 16=cylindrical/spherical)
    :type nr: int
    :param probid: Problem identification string
    :type probid: str, optional
    :param nt: Number of time bins per particle type
    :type nt: List[int]
    :param ne: Number of energy bins per particle type
    :type ne: List[int]
    :param nfx: Total number of fine mesh bins in x/r direction
    :type nfx: float, optional
    :param nfy: Total number of fine mesh bins in y/z/theta direction
    :type nfy: float, optional
    :param nfz: Total number of fine mesh bins in z/theta/phi direction
    :type nfz: float, optional
    :param x0: Origin x-coordinate or radial reference point
    :type x0: float, optional
    :param y0: Origin y-coordinate or axial reference point
    :type y0: float, optional
    :param z0: Origin z-coordinate or angular reference point
    :type z0: float, optional
    :param ncx: Number of coarse mesh bins in x/r direction
    :type ncx: float, optional
    :param ncy: Number of coarse mesh bins in y/z/theta direction
    :type ncy: float, optional
    :param ncz: Number of coarse mesh bins in z/theta/phi direction
    :type ncz: float, optional
    :param nwg: Geometry type (1=cartesian, 2=cylindrical, 3=spherical)
    :type nwg: float, optional
    :param x1: X-component of first reference vector
    :type x1: float, optional
    :param y1: Y-component of first reference vector
    :type y1: float, optional
    :param z1: Z-component of first reference vector
    :type z1: float, optional
    :param x2: X-component of second reference vector
    :type x2: float, optional
    :param y2: Y-component of second reference vector
    :type y2: float, optional
    :param z2: Z-component of second reference vector
    :type z2: float, optional
    """
    if_: int            # File type. Only 1 is supported.
    iv: int             # Time-dependent windows flag (1 / 2 = no / yes)
    ni: int             # Number of particle types   
    nr: int             # = 10 / 16 / 16 - rectangular / cylindrical / spherical
    probid: str = ""    # Made optional with default empty string

    # Optional arrays that might appear depending on 'iv' or 'nr'
    nt: List[int] = field(default_factory=list)     # Number of time bins per particle type in ni
    ne: List[int] = field(default_factory=list)     # Number of energy bins per particle type in ni

    # Additional geometry specs
    nfx: Optional[float] = None         # Total number of fine mesh bins in x(cartesian) or r(cylindrical) or r(spherical)
    nfy: Optional[float] = None         # Total number of fine mesh bins in y(cartesian) or z(cylindrical) or theta(spherical)
    nfz: Optional[float] = None         # Total number of fine mesh bins in z(cartesian) or theta(cylindrical) or phi(spherical)
    x0: Optional[float] = None          # x0,y0,z0: Corner of (x,y,z) in cartesian or 
    y0: Optional[float] = None          # bottom center of (r,z,theta) in cylindrical or 
    z0: Optional[float] = None          # center of (r,theta,phi) in spherical

    ncx: Optional[float] = None         # Number of coarse mesh bins in x(cartesian) or r(cylindrical) or r(spherical)
    ncy: Optional[float] = None         # Number of coarse mesh bins in y(cartesian) or z(cylindrical) or theta(spherical)
    ncz: Optional[float] = None         # Number of coarse mesh bins in z(cartesian) or theta(cylindrical) or phi(spherical)
    nwg: Optional[float] = None         # Geometry type (1=cartesian, 2=cylindrical, 3=spherical)

    # Optional for nr = 16
    x1: Optional[float] = None          # Vector from (x0,y0,z0) to (x1,y1,z1)
    y1: Optional[float] = None          # defines (r,z,theta) cylinder or
    z1: Optional[float] = None          # (r,theta,phi) polar axis
    x2: Optional[float] = None          # Vector from (x0,y0,z0) to (x2,y2,z2)
    y2: Optional[float] = None          # defines (r,z,theta) cylinder or
    z2: Optional[float] = None          # (r,theta,phi) azimuthal axis

    @property
    def has_time_dependency(self) -> bool:
        """Check if the window weights have time dependency.

        :return: True if window weights are time-dependent (iv=2), False otherwise
        :rtype: bool
        """
        return self.iv == 2

    @property
    def type_of_mesh(self) -> Optional[str]:
        """Get the type of mesh geometry being used.

        :return: String indicating mesh type ('cartesian', 'cylindrical', 'spherical'),
                'unknown' for invalid values, or None if nwg is not set
        :rtype: Optional[str]
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
    def number_of_time_bins(self) -> List[int]:
        """Get the list of time bins for each particle type.

        :return: List containing the number of time bins for each particle type
        :rtype: List[int]
        """
        return self.nt

    @property
    def number_of_energy_bins(self) -> List[int]:
        """Get the list of energy bins for each particle type.

        :return: List containing the number of energy bins for each particle type
        :rtype: List[int]
        """
        return self.ne

    @property
    def number_of_particle_types(self) -> int:
        """Get the total number of particle types.

        :return: Number of particle types defined in the window weights
        :rtype: int
        """
        return self.ni