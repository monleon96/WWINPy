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
    

def get_closest_energy_indices(energy_grid: np.ndarray, energy_value: float, atol: float = 1e-9) -> np.ndarray:
    """Find energy indices bounding a value in an energy grid.

    :param energy_grid: Sorted energy grid points array
    :type energy_grid: np.ndarray
    :param energy_value: Target energy value
    :type energy_value: float
    :param atol: Absolute tolerance for float comparison
    :type atol: float, optional
    :return: Array of one or two indices. Two indices only when value exactly matches a grid point
    :rtype: np.ndarray
    """
    if len(energy_grid) == 0:
        raise ValueError("Energy grid cannot be empty")

    if energy_value > energy_grid[-1]:
        return np.array([len(energy_grid)-1])
    
    # Use side='left' to get the index where the value would be inserted
    idx = np.searchsorted(energy_grid, energy_value, side='left')
    
    # Check if the value exactly matches a grid point
    if idx < len(energy_grid) and np.isclose(energy_grid[idx], energy_value, atol=atol):
        if idx == len(energy_grid) - 1:  # Last point in grid
            return np.array([idx])
        return np.array([idx, idx + 1])
    
    # If we're at the start of the array, return first index
    if idx == 0:
        return np.array([0])
        
    # Otherwise return the index to the left of where the value would be inserted
    return np.array([idx])

def get_range_energy_indices(grid: np.ndarray, range_tuple: Tuple[float, float], atol: float = 1e-9) -> np.ndarray:
    """Find energy indices for a range of values in an energy grid.

    :param grid: Sorted energy grid points array
    :type grid: np.ndarray
    :param range_tuple: (min, max) energy range values
    :type range_tuple: Tuple[float, float]
    :return: Array of indices covering the range
    :rtype: np.ndarray
    :raises ValueError: If grid is empty or range is invalid
    """
    if not grid.size:  # More robust empty check
        raise ValueError("Energy grid cannot be empty")

    v_min, v_max = range_tuple
    if v_min > v_max:
        raise ValueError(f"Invalid range: min {v_min} is greater than max {v_max}")

    # Find the starting index
    start_idx = np.searchsorted(grid, v_min)
    if start_idx == len(grid) or not np.isclose(grid[start_idx], v_min, atol) and grid[start_idx] < v_min:
        if start_idx == len(grid):
            return np.array([], dtype=int)
        start_idx += 1

    # Find the ending index (Corrected Logic)
    end_idx = np.searchsorted(grid, v_max)
    if end_idx > 0 and not np.isclose(grid[end_idx-1], v_max, atol) and grid[end_idx-1] > v_max:
        end_idx -= 1

    if start_idx > end_idx:
        return np.array([], dtype=int)

    return np.arange(start_idx, end_idx+1)

def get_energy_intervals_from_indices(bins: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get energy interval boundaries for given indices.
    
    :param bins: Energy grid boundaries array
    :type bins: np.ndarray
    :param indices: Array of indices from get_closest_energy_indices
    :type indices: np.ndarray
    :return: Tuple of (starts, ends) arrays for energy intervals
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    if len(indices) == 0:
        return np.array([]), np.array([])
    
    starts = []
    ends = []
    
    for idx in indices:
        if idx == 0:
            # For index 0, use 0.0 as lower bound
            starts.append(0.0)
            ends.append(bins[0])
        else:
            # For other indices, use previous bin value and current bin value
            starts.append(bins[idx-1])
            ends.append(bins[idx])
            
    return np.array(starts), np.array(ends)

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