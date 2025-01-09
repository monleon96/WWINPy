"""
WWINP file parser module.

This module provides functionality for reading and parsing MCNP weight window input (WWINP) files.
Contains optimized parsing functions for handling large data files efficiently.
"""

from typing import Iterator
import itertools
from wwinpy._utils import verify_and_correct
from wwinpy._exceptions import WWINPFormatError
from wwinpy.header import Header
from wwinpy.ww_data import WWData
from wwinpy.mesh import Mesh
from wwinpy.geometry import GeometryData, GeometryAxis
from wwinpy.weight_windows import WeightWindowValues
import numpy as np

def _tokenize_file(file_path: str) -> Iterator[str]:
    """Generate tokens from a WWINP file line by line.

    :param file_path: Path to the WWINP file
    :type file_path: str
    :return: Iterator over tokens in the file
    :rtype: Iterator[str]
    :raises WWINPFormatError: If an empty line is encountered
    """
    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                raise WWINPFormatError(f"Empty line detected at line {line_number}.")
            for token in line.split():
                yield token

def from_file(file_path: str, verbose: bool = False) -> WWData:
    """Parse a WWINP file and create a WWData object.

    :param file_path: Path to the WWINP file to parse
    :type file_path: str
    :param verbose: If True, print detailed parsing information
    :type verbose: bool
    :return: Parsed WWINP data object
    :rtype: WWData
    :raises WWINPFormatError: If the file format is invalid
    :raises WWINPParsingError: If there are errors during parsing
    """
    token_gen = _tokenize_file(file_path)

    # ---------------------------
    # Block 1: Parse Header
    # ---------------------------
    if verbose:
        print(f"Reading file: {file_path}")

    try:
        # Read first 4 tokens as integers
        header_tokens = list(itertools.islice(token_gen, 4))
        if len(header_tokens) < 4:
            raise WWINPFormatError("File ended unexpectedly while reading the header.")
        if_, iv, ni, nr = map(int, header_tokens)
        if verbose:
            print(f"Header values: if={if_}, iv={iv}, ni={ni}, nr={nr}")

        # Attempt to read probid
        try:
            next_token = next(token_gen)
            if next_token.isdigit():
                probid = ""
                token_gen = itertools.chain([next_token], token_gen)
            else:
                probid = next_token
        except StopIteration:
            probid = ""

        if verbose:
            print(f"probid='{probid}'")
    except StopIteration:
        raise WWINPFormatError("File ended unexpectedly while reading the header.")
    except ValueError:
        raise WWINPFormatError("Header contains non-integer values.")

    header = Header(if_=if_, iv=iv, ni=ni, nr=nr, probid=probid)

    # Parse nt if iv=2
    if iv == 2:
        if verbose:
            print("Parsing nt array (iv=2)...")
        try:
            header.nt = list(map(int, itertools.islice(token_gen, ni)))
            if len(header.nt) < ni:
                raise WWINPFormatError("File ended while reading nt array.")
            if verbose:
                for i, val in enumerate(header.nt):
                    print(f"  nt[{i}] = {val}")
        except StopIteration:
            raise WWINPFormatError("File ended while reading nt array.")

    # Parse ne array
    if verbose:
        print("Parsing ne array...")
    try:
        header.ne = list(map(int, itertools.islice(token_gen, ni)))
        if len(header.ne) < ni:
            raise WWINPFormatError("File ended while reading ne array.")
        if verbose:
            for i, val in enumerate(header.ne):
                print(f"  ne[{i}] = {val}")
    except StopIteration:
        raise WWINPFormatError("File ended while reading ne array.")

    # ---------------------------
    # Verification and Correction
    # ---------------------------
    if verbose:
        print("\n=== Verifying and Correcting Data ===")

    updated_ni, updated_nt, updated_ne = verify_and_correct(
        ni=header.ni,
        nt=header.nt if iv == 2 else None,
        ne=header.ne,
        iv=iv,
        verbose=verbose
    )

    header.ni = updated_ni
    if iv == 2:
        header.nt = updated_nt
    header.ne = updated_ne

    if verbose:
        print(f"  Updated ni: {header.ni}")
        for i in range(header.ni):
            if iv == 2:
                print(f"  Updated nt[{i}]: {header.nt[i]}")
            print(f"  Updated ne[{i}]: {header.ne[i]}")

    # Parse geometry parameters
    if verbose:
        print("\nParsing geometry parameters...")
    try:
        geom_tokens = list(itertools.islice(token_gen, 6))
        if len(geom_tokens) < 6:
            raise WWINPFormatError("Not enough tokens for geometry parameters.")
        header.nfx, header.nfy, header.nfz, header.x0, header.y0, header.z0 = map(float, geom_tokens)
        if verbose:
            print(f"  nfx={header.nfx}, nfy={header.nfy}, nfz={header.nfz}")
            print(f"  x0={header.x0}, y0={header.y0}, z0={header.z0}")
    except StopIteration:
        raise WWINPFormatError("File ended while reading geometry parameters.")

    # Parse nr-dependent values
    if verbose:
        print(f"Parsing nr={nr} specific values...")
    try:
        if nr == 10:
            nr_tokens = list(itertools.islice(token_gen, 4))
            if len(nr_tokens) < 4:
                raise WWINPFormatError("Not enough tokens for [nr=10] line.")
            header.ncx, header.ncy, header.ncz, header.nwg = map(float, nr_tokens)
            if verbose:
                print(f"  ncx={header.ncx}, ncy={header.ncy}, ncz={header.ncz}, nwg={header.nwg}")
        elif nr == 16:
            nr_tokens1 = list(itertools.islice(token_gen, 6))
            nr_tokens2 = list(itertools.islice(token_gen, 4))
            if len(nr_tokens1) < 6 or len(nr_tokens2) < 4:
                raise WWINPFormatError("Not enough tokens for [nr=16] lines.")
            header.ncx, header.ncy, header.ncz, header.x1, header.y1, header.z1 = map(float, nr_tokens1)
            header.x2, header.y2, header.z2, header.nwg = map(float, nr_tokens2)
            if verbose:
                print(f"  ncx={header.ncx}, ncy={header.ncy}, ncz={header.ncz}")
                print(f"  x1={header.x1}, y1={header.y1}, z1={header.z1}")
                print(f"  x2={header.x2}, y2={header.y2}, z2={header.z2}, nwg={header.nwg}")
        else:
            raise WWINPFormatError(f"Unsupported nr value: {nr}")
    except StopIteration:
        raise WWINPFormatError("File ended while reading nr-dependent values.")

    # ---------------------------
    # Block 2: Geometry
    # ---------------------------
    if verbose:
        print("\n=== Parsing Geometry Block ===")
        
    ncx, ncy, ncz = int(header.ncx), int(header.ncy), int(header.ncz)

    if verbose:
        print(f"Mesh dimensions: ncx={ncx}, ncy={ncy}, ncz={ncz}")

    # Function to parse axis data using GeometryAxis with NumPy arrays
    def parse_axis(axis_name: str, n_segments: int, verbose: bool) -> GeometryAxis:
        axis = GeometryAxis(origin=0.0)  # Initialize with default origin
        try:
            origin = float(next(token_gen))
            axis.origin = origin
            if verbose:
                print(f"{axis_name}-axis origin: {origin}")
        except StopIteration:
            raise WWINPFormatError(f"File ended while reading {axis_name}-axis origin.")
        
        for i in range(n_segments):
            try:
                q, p, s = map(float, itertools.islice(token_gen, 3))
                axis.add_segment(q, p, s)
                if verbose:
                    print(f"  {axis_name}-segment[{i}]: q={q}, p={p}, s={s}")
            except StopIteration:
                raise WWINPFormatError(f"File ended while reading {axis_name}_segments.")
        return axis

    x_axis = parse_axis("X", ncx, verbose)
    y_axis = parse_axis("Y", ncy, verbose)
    z_axis = parse_axis("Z", ncz, verbose)

    geometry = GeometryData(
        header=header,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis
    )

    # ---------------------------
    # Block 3: Values (Times, Energies, WW-Values)
    # ---------------------------
    if verbose:
        print("\n=== Parsing Values Block ===")

    # 1) Read time bins (if iv=2)
    if verbose:
        print("Reading time bins...")
    if iv == 2:
        time_bins_all = []
        for i in range(header.ni):
            try:
                t_bins = np.fromiter(
                    (float(token) for token in itertools.islice(token_gen, header.nt[i])),
                    dtype=np.float64,
                    count=header.nt[i]
                )
                if t_bins.size < header.nt[i]:
                    raise WWINPFormatError(f"File ended while reading time bins for particle {i}.")
                time_bins_all.append(t_bins)
                if verbose:
                    for j, val in enumerate(t_bins):
                        print(f"  t[{i}][{j}] = {val}")
            except StopIteration:
                raise WWINPFormatError(f"File ended while reading time bins for particle {i}.")
    else:
        # No time dependency; assign a default single-element array for each particle
        default_time = 0.0
        time_bins_all = [np.array([default_time]) for _ in range(header.ni)]
        if verbose:
            for i in range(header.ni):
                print(f"  t[{i}] = []")

    # 2) Read energy bins
    if verbose:
        print("Reading energy bins...")
    energy_bins_all = []
    for i in range(header.ni):
        try:
            e_bins = np.fromiter(
                (float(token) for token in itertools.islice(token_gen, header.ne[i])),
                dtype=np.float64,
                count=header.ne[i]
            )
            if e_bins.size < header.ne[i]:
                raise WWINPFormatError(f"File ended while reading energy bins for particle {i}.")
            energy_bins_all.append(e_bins)
            if verbose:
                for j, val in enumerate(e_bins):
                    print(f"  e[{i}][{j}] = {val}")
        except StopIteration:
            raise WWINPFormatError(f"File ended while reading energy bins for particle {i}.")

    # 3) Read ww-values with enhanced verbosity
    if verbose:
        print("Reading ww-values...")

    # Precompute number of geometry cells
    num_geom_cells = int(header.nfx * header.nfy * header.nfz)  # Assuming ncx, ncy, ncz are integers
    
    if verbose:
        print(f"Total geometry cells: {num_geom_cells}")

        # Initialize counters for verbose logging
        total_expected_ww_values = 0
        for i in range(header.ni):
            time_max_range = header.nt[i] if iv == 2 else 1
            total_expected_ww_values += time_max_range * header.ne[i] * num_geom_cells

        print(f"Total expected ww-values: {total_expected_ww_values}")
    

    # Create dictionaries for time and energy bins
    time_mesh_dict = {}
    energy_mesh_dict = {}
    ww_values_dict = {}

    for i in range(header.ni):
        time_mesh_dict[i] = time_bins_all[i]
        energy_mesh_dict[i] = energy_bins_all[i]

    try:
        for i in range(header.ni):
            time_bins = time_bins_all[i]
            energy_bins = energy_bins_all[i]

            ww_all = np.empty((len(time_bins), len(energy_bins), num_geom_cells))

            for t in range(len(time_bins)):
                for e in range(len(energy_bins)):
                    ww_all[t, e, :] = np.fromiter(
                        (float(token) for token in itertools.islice(token_gen, num_geom_cells)),
                        dtype=np.float64,
                        count=num_geom_cells
                    )
            ww_values_dict[i] = ww_all
    except StopIteration:
        raise WWINPFormatError("File ended unexpectedly while reading ww-values.")

    if verbose:
        total_ww_values_read = sum(p.size for p in ww_values_dict.values())
        print(f"Total ww-values read: {total_ww_values_read}")
        if total_ww_values_read < total_expected_ww_values:
            print(f"Warning: Expected {total_expected_ww_values} ww-values, but only {total_ww_values_read} were read.")

    # Create the mesh with time and energy data
    mesh = Mesh(
        header=header,
        geometry=geometry,
        time_mesh=time_mesh_dict,
        energy_mesh=energy_mesh_dict,
    )
    
    values = WeightWindowValues(
        header=header,
        mesh=mesh,
        ww_values=ww_values_dict
    )

    # Finally construct the WWData
    wwinp_data = WWData(
        header=header,
        mesh=mesh,
        values=values
    )

    return wwinp_data
