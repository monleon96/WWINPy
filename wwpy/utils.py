# utils.py

from typing import List, Optional, Tuple
import numpy as np

def safe_int(value, default=None):
    """
    Attempt to convert `value` to int, return `default` if conversion fails.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value, default=None):
    """
    Attempt to convert `value` to float, return `default` if conversion fails.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
    

def verify_and_correct(
    ni: int,
    nt: Optional[List[int]],
    ne: List[int],
    iv: int,
) -> Tuple[int, Optional[List[int]], List[int]]:
    """
    Verifies and corrects the ni, nt, and ne parameters based on specified rules.
    
    Args:
        ni: Number of initial particles.
        nt: List of time groups per particle type (only if iv == 2).
        ne: List of energy groups per particle type.
        iv: Indicator if nt exists (iv=2 means nt exists).
    
    Returns:
        A tuple containing the updated ni, nt, and ne.
    """
    changes_made = False

    # Step 1: Verify lengths
    if iv == 2 and nt is not None:
        if len(nt) != len(ne):
            min_length = min(len(nt), len(ne))
            if len(nt) != min_length or len(ne) != min_length:
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
            print(
                f"Warning: Length of ne ({len(ne)}) or nt ({len(nt)}) does not match ni ({ni}). Adjusting ni to {min(len(ne), len(nt))}."
            )
            ni = min(len(ne), len(nt))
            ne = ne[:ni]
            nt = nt[:ni]
            changes_made = True
    else:
        if len(ne) != ni:
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
                    print(
                        f"Warning: Particle type {i} has 0 energy and 0 time groups. It has been deleted and ni updated."
                    )
                else:
                    print(
                        f"Warning: Particle type {i} has 0 energy groups. It has been deleted and ni updated."
                    )
            else:
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
                print(
                    f"Warning: After corrections, lengths of ne ({len(ne)}) or nt ({len(nt)}) do not match ni ({ni}). Truncating lists to {min_length}."
                )
                ne = ne[:min_length]
                nt = nt[:min_length]
                ni = min_length
    else:
        if len(ne) != ni:
            print(
                f"Warning: After corrections, length of ne ({len(ne)}) does not match ni ({ni}). Truncating ne to {ni}."
            )
            ne = ne[:ni]

    if changes_made:
        return ni, nt, ne
    else:
        print("Header verification complete. No changes made.")
        return ni, nt, ne
    

def get_closest_indices(grid, value):
    """Helper function to find closest lower and upper indices for a value."""
    idx = np.searchsorted(grid, value, side="left")
    lower_idx = max(0, idx - 1)
    upper_idx = min(len(grid) - 1, idx)
    if grid[lower_idx] == value:
        return [lower_idx]
    return [lower_idx, upper_idx]

def get_range_indices(grid, range_tuple):
    """Helper function to find range indices for a tuple."""
    start_idx = np.searchsorted(grid, range_tuple[0], side="left")
    end_idx = np.searchsorted(grid, range_tuple[1], side="right") - 1
    return np.arange(max(0, start_idx), min(len(grid), end_idx + 1))