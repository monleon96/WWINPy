"""Utility functions for WWINP file processing.

Provides helper functions for data verification and grid operations.
"""

# utils.py

from typing import List, Optional, Tuple
import numpy as np

def verify_and_correct(ni: int, nt: Optional[List[int]], ne: List[int],
                      iv: int, verbose: bool = False) -> Tuple[int, Optional[List[int]], List[int]]:
    """Verify and correct weight window input parameters.

    :param ni: Number of particle types
    :type ni: int
    :param nt: Time groups per particle type (None if iv != 2)
    :type nt: Optional[List[int]]
    :param ne: Energy groups per particle type
    :type ne: List[int]
    :param iv: Time dependency flag (2 for time-dependent)
    :type iv: int
    :param verbose: Enable detailed output
    :type verbose: bool
    :return: Tuple of (corrected ni, corrected nt, corrected ne)
    :rtype: Tuple[int, Optional[List[int]], List[int]]
    :raises ValueError: If input parameters are inconsistent
    """
    changes_made = False

    # Step 1: Verify lengths
    if iv == 2 and nt is not None:
        if len(nt) != len(ne):
            min_length = min(len(nt), len(ne))
            if len(nt) != min_length or len(ne) != min_length:
                if verbose:
                    print(
                        f"Warning: Length of nt ({len(nt)}) and ne ({len(ne)}) do not match. Truncating to {min_length}."
                    )
                nt = nt[:min_length]
                ne = ne[:min_length]
                ni = min_length
                changes_made = True

    # Step 2: Verify lengths match ni
    if iv == 2 and nt is not None:
        if len(ne) != ni or len(nt) != ni:
            if verbose:
                print(
                    f"Warning: Length of ne ({len(ne)}) or nt ({len(nt)}) does not match ni ({ni}). Adjusting ni to {min(len(ne), len(nt))}."
                )
            ni = min(len(ne), len(nt))
            ne = ne[:ni]
            nt = nt[:ni]
            changes_made = True
    else:
        if len(ne) != ni:
            if verbose:
                print(
                    f"Warning: Length of ne ({len(ne)}) does not match ni ({ni}). Adjusting ni to {len(ne)}."
                )
            ni = len(ne)
            ne = ne[:ni]
            changes_made = True

    # Step 3: Identify indices with ne == 0
    zero_ne_indices = {i for i, val in enumerate(ne) if val == 0}

    if zero_ne_indices:
        changes_made = True
        for i in sorted(zero_ne_indices, reverse=True):
            if iv == 2 and nt is not None:
                if nt[i] == 0:
                    if verbose:
                        print(
                            f"Warning: Particle type {i} has 0 energy and 0 time groups. It has been deleted and ni updated."
                        )
                else:
                    if verbose:
                        print(
                            f"Warning: Particle type {i} has 0 energy groups. It has been deleted and ni updated."
                        )
            else:
                if verbose:
                    print(
                        f"Warning: Particle type {i} has 0 energy groups. It has been deleted and ni updated."
                    )
            # Remove the particle type
            del ne[i]
            if iv == 2 and nt is not None:
                del nt[i]
            ni -= 1

    # Step 4: If iv == 2, check for 0's in nt
    if iv == 2 and nt is not None:
        zero_nt_indices = {i for i, val in enumerate(nt) if val == 0}
        if zero_nt_indices:
            changes_made = True
            for i in sorted(zero_nt_indices, reverse=True):
                if verbose:
                    print(
                        f"Warning: Particle type {i} has 0 time groups. It has been deleted and ni updated."
                    )
                del nt[i]
                del ne[i]
                ni -= 1

    # Step 5: Final length checks
    if iv == 2 and nt is not None:
        if len(ne) != ni or len(nt) != ni:
            min_length = min(len(ne), len(nt), ni)
            if len(ne) != min_length or len(nt) != min_length:
                if verbose:
                    print(
                        f"Warning: After corrections, lengths of ne ({len(ne)}) or nt ({len(nt)}) do not match ni ({ni}). Truncating lists to {min_length}."
                    )
                ne = ne[:min_length]
                nt = nt[:min_length]
                ni = min_length
    else:
        if len(ne) != ni:
            if verbose:
                print(
                    f"Warning: After corrections, length of ne ({len(ne)}) does not match ni ({ni}). Truncating ne to {ni}."
                )
            ne = ne[:ni]

    if changes_made:
        return ni, nt, ne
    else:
        if verbose:
            print("Header verification complete. No changes made.")
        return ni, nt, ne
    

def get_closest_indices(grid: np.ndarray, value: float, atol: float = 1e-9) -> np.ndarray:
    """Find indices bounding a value in a grid.

    :param grid: Sorted grid points array
    :type grid: np.ndarray
    :param value: Target value
    :type value: float
    :param atol: Absolute tolerance for float comparison
    :type atol: float, optional
    :return: Array of two bounding indices
    :rtype: np.ndarray
    """
    if len(grid) == 1:
        return np.array([0, np.inf])
    
    if value < grid[0]:
        return np.array([0, 1])
    if value > grid[-1]:
        return np.array([len(grid) - 2, len(grid) - 1])
    
    idx = np.searchsorted(grid, value)
    
    if idx == 0:
        return np.array([0, 1])
    if idx == len(grid):
        return np.array([len(grid) - 2, len(grid) - 1])

    if np.isclose(grid[idx], value, atol=atol):
        if idx == 0:
            return np.array([0, 1])
        elif idx == len(grid) - 1:
            return np.array([len(grid) - 2, len(grid) - 1])
        else:
            return np.array([idx - 1, idx + 1])
        
    return np.array([idx - 1, idx])


def get_range_indices(grid: np.ndarray, range_tuple: Tuple[float, float]) -> np.ndarray:
    """Find grid indices within a range.

    :param grid: Sorted grid points array
    :type grid: np.ndarray
    :param range_tuple: (min, max) range values
    :type range_tuple: Tuple[float, float]
    :return: Array of indices within range
    :rtype: np.ndarray
    :raises ValueError: If range_tuple[0] > range_tuple[1]
    """
    v_min, v_max = range_tuple

    # Handle grids with 0 or 1 elements
    if grid.size <= 1:
        return np.array([0, np.inf])

    if v_min > v_max:
        raise ValueError(f"Invalid range: min {v_min} is greater than max {v_max}.")

    # Initialize lower and upper indices
    lower_idx = None
    upper_idx = None

    # Check if v_min is exactly on the grid
    exact_min = np.isclose(grid, v_min, atol=1e-9)
    if exact_min.any():
        lower_idx = np.argmax(exact_min)
    elif v_min < grid[0]:
        print(f"Warning: Lower bound {v_min} is below the grid range. Using first grid point {grid[0]:.4e} as the lower limit.")
        lower_idx = 0
    else:
        # Find the closest lower grid point
        lower_idx = np.searchsorted(grid, v_min, side='right') - 1
        lower_idx = max(lower_idx, 0)  # Ensure non-negative

    # Check if v_max is exactly on the grid
    exact_max = np.isclose(grid, v_max, atol=1e-9)
    if exact_max.any():
        upper_idx = np.argmax(exact_max)
    elif v_max > grid[-1]:
        print(f"Warning: Upper bound {v_max} is above the grid range. Using last grid point {grid[-1]:.4e} as the upper limit.")
        upper_idx = grid.size - 1
    else:
        # Find the closest higher grid point
        upper_idx = np.searchsorted(grid, v_max, side='left')
        if upper_idx >= grid.size:
            upper_idx = grid.size - 1  # Ensure within bounds

    # Ensure upper_idx is not less than lower_idx
    upper_idx = max(upper_idx, lower_idx)

    # Collect all indices within [lower_idx, upper_idx]
    indices = np.arange(lower_idx, upper_idx + 1)

    return indices


def get_bin_intervals_from_indices(bins: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get bin interval boundaries for given indices.

    :param bins: Array of bin boundaries
    :type bins: np.ndarray
    :param indices: Array of [start, end] indices
    :type indices: np.ndarray
    :return: Tuple of (bin_starts, bin_ends) arrays
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: If bins is None/empty or indices invalid
    """
    if bins is None:
        raise ValueError("Bins must have at least one value.")
    
    if indices[0] == 0 and (indices[1] == np.inf or indices[1] == 0):
        return 0.0, float('inf')
    
    start_idx = indices[0]
    end_idx = indices[-1]
    if start_idx > end_idx or start_idx < 0 or end_idx > len(bins):
        raise ValueError("Indices must define a valid range within the bins.")

    bin_starts = bins[start_idx:end_idx]
    bin_ends = bins[start_idx + 1:end_idx + 1]

    return bin_starts, bin_ends