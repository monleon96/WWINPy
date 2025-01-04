# wwpy/models.py
from dataclasses import dataclass, field
from typing import List, Optional

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


@dataclass
class CoarseMeshSegment:
    """
    Represents a single coarse-mesh segment along one axis,
    storing q, p, and s for that segment.
    """
    q: float  # Fine mesh ratio (presently = 1 always)
    p: float  # Coarse mesh coordinate
    s: float  # Number of fine meshes in each coarse mesh


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
    time_bins: List[float] = field(default_factory=list)
    energy_bins: List[float] = field(default_factory=list)
    # w_values[time_index][energy_index][geom_index] = lower weight-window bound
    w_values: List[List[List[float]]] = field(default_factory=list)


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
