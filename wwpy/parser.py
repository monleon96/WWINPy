# wwpy/parser.py

from typing import List, Iterator
import re

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


def _tokenize_file_lines(lines: List[str]) -> Iterator[str]:
    """
    Generator that yields tokens from each line.
    Each line can have up to 6 columns (tokens). 
    We'll split on whitespace, but also handle continuation if needed.
    """
    for line_number, line in enumerate(lines, start=1):
        # Strip trailing newline
        line = line.strip()
        if not line:
            # According to specification: No empty lines permitted, so we can raise an error
            raise WWINPFormatError(f"Empty line detected at line {line_number}.")
        # Split by whitespace
        tokens = re.split(r"\s+", line)
        for token in tokens:
            yield token


def parse_wwinp_file(file_path: str, verbose: bool = False) -> WWINPData:
    """
    Main parser function for WWINP files.
    Args:
        file_path: Path to the WWINP file
        verbose: If True, print detailed parsing information
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    if verbose:
        print(f"Reading file: {file_path}")
        print(f"Found {len(lines)} lines")

    token_gen = _tokenize_file_lines(lines)

    # ---------------------------
    # Block 1: Parse Header
    # ---------------------------
    if verbose:
        print("\n=== Parsing Header Block ===")
    
    try:
        if_ = int(next(token_gen))
        iv = int(next(token_gen))
        ni = int(next(token_gen))
        nr = int(next(token_gen))
        
        # Try to get the next token and check if it's numeric
        next_token = next(token_gen)
        try:
            int(next_token)  # If this succeeds, it's a number, not probid
            probid = ""  # probid was missing
            # Put the token back into consideration since it's part of the next data
            token_gen = iter([next_token] + list(token_gen))
        except ValueError:
            # If conversion to float fails, this is the probid
            probid = next_token
            
        if verbose:
            print(f"Header values: if={if_}, iv={iv}, ni={ni}, nr={nr}, probid='{probid}'")
    except StopIteration:
        raise WWINPFormatError("File ended unexpectedly while reading the header.")

    header = Header(if_=if_, iv=iv, ni=ni, nr=nr, probid=probid)

    # Parse nt if iv=2
    if iv == 2:
        if verbose:
            print("Parsing nt array (iv=2)...")
        nt_list = []
        for i in range(ni):
            try:
                val = int(next(token_gen))
                nt_list.append(val)
                if verbose:
                    print(f"  nt[{i}] = {val}")
            except StopIteration:
                raise WWINPFormatError("File ended while reading nt array.")
        header.nt = nt_list

    # Parse ne array
    if verbose:
        print("Parsing ne array...")
    ne_list = []
    for i in range(ni):
        try:
            val = int(next(token_gen))
            ne_list.append(val)
            if verbose:
                print(f"  ne[{i}] = {val}")
        except StopIteration:
            raise WWINPFormatError("File ended while reading ne array.")
    header.ne = ne_list

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
        header.nfx = float(next(token_gen))
        header.nfy = float(next(token_gen))
        header.nfz = float(next(token_gen))
        header.x0  = float(next(token_gen))
        header.y0  = float(next(token_gen))
        header.z0  = float(next(token_gen))
        if verbose:
            print(f"  nfx={header.nfx}, nfy={header.nfy}, nfz={header.nfz}")
            print(f"  x0={header.x0}, y0={header.y0}, z0={header.z0}")
    except StopIteration:
        raise WWINPFormatError("File ended while reading nfx, nfy, nfz, x0, y0, z0.")

    # Parse nr-dependent values
    if verbose:
        print(f"Parsing nr={nr} specific values...")
    if nr == 10:
        try:
            header.ncx = float(next(token_gen))
            header.ncy = float(next(token_gen))
            header.ncz = float(next(token_gen))
            header.nwg = float(next(token_gen))
            if verbose:
                print(f"  ncx={header.ncx}, ncy={header.ncy}, ncz={header.ncz}, nwg={header.nwg}")
        except StopIteration:
            raise WWINPFormatError("File ended while reading [nr=10] line.")
    elif nr == 16:
        try:
            header.ncx = float(next(token_gen))
            header.ncy = float(next(token_gen))
            header.ncz = float(next(token_gen))
            header.x1  = float(next(token_gen))
            header.y1  = float(next(token_gen))
            header.z1  = float(next(token_gen))
            if verbose:
                print(f"  ncx={header.ncx}, ncy={header.ncy}, ncz={header.ncz}")
                print(f"  x1={header.x1}, y1={header.y1}, z1={header.z1}")
        except StopIteration:
            raise WWINPFormatError("File ended while reading [nr=16] line #1.")
        try:
            header.x2  = float(next(token_gen))
            header.y2  = float(next(token_gen))
            header.z2  = float(next(token_gen))
            header.nwg = float(next(token_gen))
            if verbose:
                print(f"  x2={header.x2}, y2={header.y2}, z2={header.z2}, nwg={header.nwg}")
        except StopIteration:
            raise WWINPFormatError("File ended while reading [nr=16] line #2.")

    # ---------------------------
    # Block 2: Geometry
    # ---------------------------
    if verbose:
        print("\n=== Parsing Geometry Block ===")
        
    ncx = int(header.ncx) if header.ncx else 0
    ncy = int(header.ncy) if header.ncy else 0
    ncz = int(header.ncz) if header.ncz else 0

    if verbose:
        print(f"Mesh dimensions: ncx={ncx}, ncy={ncy}, ncz={ncz}")

    # Parse X-axis
    try:
        x_origin = float(next(token_gen))
        if verbose:
            print(f"X-axis origin: {x_origin}")
    except StopIteration:
        raise WWINPFormatError("File ended while reading x-axis origin.")

    x_segments = []
    for i in range(ncx):
        try:
            q_val = float(next(token_gen))
            p_val = float(next(token_gen))
            s_val = float(next(token_gen))
            if verbose:
                print(f"  X-segment[{i}]: q={q_val}, p={p_val}, s={s_val}")
        except StopIteration:
            raise WWINPFormatError("File ended while reading x_segments.")
        x_segments.append(CoarseMeshSegment(q=q_val, p=p_val, s=s_val))

    x_axis = GeometryAxis(origin=x_origin, segments=x_segments)

    # Parse Y-axis
    try:
        y_origin = float(next(token_gen))
        if verbose:
            print(f"Y-axis origin: {y_origin}")
    except StopIteration:
        raise WWINPFormatError("File ended while reading y-axis origin.")

    y_segments = []
    for i in range(ncy):
        try:
            q_val = float(next(token_gen))
            p_val = float(next(token_gen))
            s_val = float(next(token_gen))
            if verbose:
                print(f"  Y-segment[{i}]: q={q_val}, p={p_val}, s={s_val}")
        except StopIteration:
            raise WWINPFormatError("File ended while reading y_segments.")
        y_segments.append(CoarseMeshSegment(q=q_val, p=p_val, s=s_val))

    y_axis = GeometryAxis(origin=y_origin, segments=y_segments)

    # Parse Z-axis
    try:
        z_origin = float(next(token_gen))
        if verbose:
            print(f"Z-axis origin: {z_origin}")
    except StopIteration:
        raise WWINPFormatError("File ended while reading z-axis origin.")

    z_segments = []
    for i in range(ncz):
        try:
            q_val = float(next(token_gen))
            p_val = float(next(token_gen))
            s_val = float(next(token_gen))
            if verbose:
                print(f"  Z-segment[{i}]: q={q_val}, p={p_val}, s={s_val}")
        except StopIteration:
            raise WWINPFormatError("File ended while reading z_segments.")
        z_segments.append(CoarseMeshSegment(q=q_val, p=p_val, s=s_val))

    z_axis = GeometryAxis(origin=z_origin, segments=z_segments)

    geometry = GeometryData(
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis
    )

    # ---------------------------
    # Block 3: Values (Times, Energies, W-Values)
    #   - t(i,1) ... t(i,nt(i)) [if nt(i) > 1]
    #   - e(i,1) ... e(i,ne(i))
    #   - ((w(i,j,k,l), l=1,nft, k=1,ne(i), j=1,nt(i))
    # ---------------------------
    if verbose:
        print("\n=== Parsing Values Block ===")
    
    # Build a ParticleBlock for each particle i in [0..ni-1].
    particle_blocks = []

    # 1) Read time bins (if nt(i) > 1)
    if verbose:
        print("Reading time bins...")
    time_bins_all = []
    if header.iv == 2:
        for i in range(header.ni):
            t_bin = []
            for j in range(header.nt[i]):
                try:
                    val = float(next(token_gen))
                    t_bin.append(val)
                    if verbose:
                        print(f"  t[{i}][{j}] = {val}")
                except StopIteration:
                    raise WWINPFormatError("File ended while reading time bins.")
            time_bins_all.append(t_bin)
    else:
        # No time dependency; assign an empty list or a default single-element list for each particle
        for i in range(header.ni):
            time_bins_all.append([])  
            if verbose:
                print(f"  t[{i}] = []")  

    # 2) Read energy bins
    if verbose:
        print("Reading energy bins...")
    energy_bins_all = []
    for i in range(header.ni):
        e_bin = []
        for j in range(header.ne[i]):
            try:
                val = float(next(token_gen))
                e_bin.append(val)
                if verbose:
                    print(f"  e[{i}][{j}] = {val}")
            except StopIteration:
                raise WWINPFormatError("File ended while reading energy bins.")
        energy_bins_all.append(e_bin)

    # 3) Read w-values in a nested loop:
    #    w_values[i][time_index][energy_index][geometry_index]
    #    We'll flatten geometry_index to a single dimension.
    #    If you need a 3D shape, adapt accordingly.
    if verbose:
        print("Reading w-values...")
    w_all = []
    for i in range(header.ni):
        w_for_i = []
        # If we have no explicit time bins, interpret nt[i] as 1
        time_max_range = 1 if header.iv != 2 else header.nt[i]
        for time_index in range(time_max_range):
            w_for_time = []
            for energy_index in range(header.ne[i]):
                # Calculate the total number of fine segments in each dimension
                total_x_fine = sum(seg.s for seg in x_segments)
                total_y_fine = sum(seg.s for seg in y_segments)
                total_z_fine = sum(seg.s for seg in z_segments)

                if any(val <= 0 for val in [total_x_fine, total_y_fine, total_z_fine]):
                    raise WWINPFormatError(f"Invalid mesh dimensions: x={total_x_fine}, y={total_y_fine}, z={total_z_fine}")

                num_geom_cells = int(total_x_fine * total_y_fine * total_z_fine)
                w_geom = []
                try:
                    for geom_index in range(num_geom_cells):
                        val = next(token_gen)
                        if val is None:
                            raise WWINPFormatError(f"Not enough w-values. Expected {num_geom_cells} values.")
                        w_geom.append(float(val))
                        if verbose:
                            print(f"  w[particle={i}/{header.ni-1}]"
                                  f"[time={time_index}/{time_max_range-1}]"
                                  f"[energy={energy_index}/{header.ne[i]-1}]"
                                  f"[geom={geom_index}/{num_geom_cells-1}] = {val}")
                except StopIteration:
                    raise WWINPFormatError(f"File ended while reading w_values. Expected {num_geom_cells} values, got {len(w_geom)}")
                
                w_for_time.append(w_geom)
            w_for_i.append(w_for_time)
        w_all.append(w_for_i)

    # Construct ParticleBlock objects
    for i in range(header.ni):
        block = ParticleBlock(
            time_bins=time_bins_all[i],
            energy_bins=energy_bins_all[i],
            w_values=w_all[i]
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
