from typing import List, Iterator
import re
import itertools
import sys

from wwpy.utils import verify_and_correct
from wwpy.exceptions import WWINPFormatError, WWINPParsingError
from wwpy.models import (
    WWINPData,
    Header,
    CoarseMeshSegment,
    GeometryAxis,
    GeometryData,
    ParticleBlock,
    WeightWindowValues    
)
import numpy as np

def _tokenize_file(file_path: str) -> Iterator[str]:
    """
    Efficient generator that yields tokens from the file line by line.
    """
    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                raise WWINPFormatError(f"Empty line detected at line {line_number}.")
            for token in line.split():
                yield token

def parse_wwinp_file(file_path: str, verbose: bool = False) -> WWINPData:
    """
    Optimized parser function for WWINP files with enhanced verbosity for w_values.
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
        iv=iv
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

    # Function to parse axis data
    def parse_axis(axis_name: str, n_segments: int, verbose: bool):
        try:
            origin = float(next(token_gen))
            if verbose:
                print(f"{axis_name}-axis origin: {origin}")
        except StopIteration:
            raise WWINPFormatError(f"File ended while reading {axis_name}-axis origin.")
        
        segments = []
        for i in range(n_segments):
            try:
                q, p, s = map(float, itertools.islice(token_gen, 3))
                if len([q, p, s]) < 3:
                    raise WWINPFormatError(f"Not enough tokens for {axis_name}-segment[{i}].")
                segments.append(CoarseMeshSegment(q=q, p=p, s=s))
                if verbose:
                    print(f"  {axis_name}-segment[{i}]: q={q}, p={p}, s={s}")
            except StopIteration:
                raise WWINPFormatError(f"File ended while reading {axis_name}_segments.")
        return GeometryAxis(origin=origin, segments=segments)

    x_axis = parse_axis("X", ncx, verbose)
    y_axis = parse_axis("Y", ncy, verbose)
    z_axis = parse_axis("Z", ncz, verbose)

    geometry = GeometryData(
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis
    )

    # ---------------------------
    # Block 3: Values (Times, Energies, W-Values)
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
                t_bins = list(map(float, itertools.islice(token_gen, header.nt[i])))
                if len(t_bins) < header.nt[i]:
                    raise WWINPFormatError(f"File ended while reading time bins for particle {i}.")
                time_bins_all.append(t_bins)
                if verbose:
                    for j, val in enumerate(t_bins):
                        print(f"  t[{i}][{j}] = {val}")
            except StopIteration:
                raise WWINPFormatError(f"File ended while reading time bins for particle {i}.")
    else:
        # No time dependency; assign an empty list or a default single-element list for each particle
        time_bins_all = [[] for _ in range(header.ni)]
        if verbose:
            for i in range(header.ni):
                print(f"  t[{i}] = []")

    # 2) Read energy bins
    if verbose:
        print("Reading energy bins...")
    energy_bins_all = []
    for i in range(header.ni):
        try:
            e_bins = list(map(float, itertools.islice(token_gen, header.ne[i])))
            if len(e_bins) < header.ne[i]:
                raise WWINPFormatError(f"File ended while reading energy bins for particle {i}.")
            energy_bins_all.append(e_bins)
            if verbose:
                for j, val in enumerate(e_bins):
                    print(f"  e[{i}][{j}] = {val}")
        except StopIteration:
            raise WWINPFormatError(f"File ended while reading energy bins for particle {i}.")

    # 3) Read w-values with enhanced verbosity
    if verbose:
        print("Reading w-values...")

    # Precompute number of geometry cells
    total_x_fine = sum(seg.s for seg in x_axis.segments)
    total_y_fine = sum(seg.s for seg in y_axis.segments)
    total_z_fine = sum(seg.s for seg in z_axis.segments)

    if any(val <= 0 for val in [total_x_fine, total_y_fine, total_z_fine]):
        raise WWINPFormatError(f"Invalid mesh dimensions: x={total_x_fine}, y={total_y_fine}, z={total_z_fine}")

    num_geom_cells = int(total_x_fine * total_y_fine * total_z_fine)

    if verbose:
        print(f"Total geometry cells: {num_geom_cells}")

    # Initialize counters for verbose logging
    total_expected_w_values = 0
    for i in range(header.ni):
        time_max_range = header.nt[i] if iv == 2 else 1
        total_expected_w_values += time_max_range * header.ne[i] * num_geom_cells

    if verbose:
        print(f"Total expected w-values: {total_expected_w_values}")

    # Initialize storage for w_values
    w_all = []
    w_values_read = 0  # Counter for w-values read

    try:
        for i in range(header.ni):
            w_for_i = []
            time_max_range = header.nt[i] if iv == 2 else 1
            for time_index in range(time_max_range):
                w_for_time = []
                for energy_index in range(header.ne[i]):
                    # Read w_geom for each geometry cell
                    try:
                        w_geom = list(itertools.islice(token_gen, num_geom_cells))
                        if len(w_geom) < num_geom_cells:
                            raise WWINPFormatError(
                                f"Not enough w-values for particle {i}, time {time_index}, energy {energy_index}. "
                                f"Expected {num_geom_cells}, got {len(w_geom)}."
                            )
                        # Convert to float
                        w_geom = list(map(float, w_geom))
                        w_for_time.append(w_geom)
                        w_values_read += num_geom_cells

                        # Verbose logging every 1,000,000 w-values
                        if verbose and w_values_read % 1_000_000 == 0:
                            print(f"  Progress: {w_values_read}/{total_expected_w_values} w-values read ({(w_values_read/total_expected_w_values)*100:.2f}%)")

                    except StopIteration:
                        raise WWINPFormatError(
                            f"File ended while reading w_values for particle {i}, time {time_index}, energy {energy_index}."
                        )
                w_for_i.append(w_for_time)
            w_all.append(w_for_i)
    except WWINPFormatError as e:
        print(f"\nError parsing w-values: {e}", file=sys.stderr)
        raise e

    if verbose:
        print(f"Total w-values read: {w_values_read}")
        if w_values_read < total_expected_w_values:
            print(f"Warning: Expected {total_expected_w_values} w-values, but only {w_values_read} were read.")

    # Construct ParticleBlock objects
    particle_blocks = []
    for i in range(header.ni):
        block = ParticleBlock(
            time_bins=np.array(time_bins_all[i], dtype=np.float32) if time_bins_all[i] else None,
            energy_bins=np.array(energy_bins_all[i], dtype=np.float32),
            w_values=np.array(w_all[i], dtype=np.float32)
        )
        particle_blocks.append(block)
        
    values = WeightWindowValues(particles=particle_blocks)

    # Finally construct the WWINPData
    wwinp_data = WWINPData(
        header=header,
        geometry=geometry,
        values=values
    )

    return wwinp_data
