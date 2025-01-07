# wwpy/header.py

from dataclasses import dataclass, field
from typing import List, Optional

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