"""Module providing optimized ratio calculations for weight window meshes.

Uses Numba-accelerated functions for efficient ratio calculations between neighboring cells.
"""

import numpy as np
from numba import njit

__all__ = ['calculate_max_ratio_array', 'calculate_ratios_stats']

@njit(cache=True)
def calculate_max_ratio_array(array: np.ndarray) -> np.ndarray:
    ratios = np.ones_like(array)
    
    # Loop over every cell
    for z in range(array.shape[0]):
        for y in range(array.shape[1]):
            for x in range(array.shape[2]):
                center_value = array[z, y, x]
                
                # If center_value is zero, skip ratio
                if center_value == 0:
                    ratios[z, y, x] = 1.0
                    continue

                # Collect only valid neighbors
                neighbors = []
                if z > 0:
                    neighbors.append(array[z - 1, y, x])
                if z < array.shape[0] - 1:
                    neighbors.append(array[z + 1, y, x])
                if y > 0:
                    neighbors.append(array[z, y - 1, x])
                if y < array.shape[1] - 1:
                    neighbors.append(array[z, y + 1, x])
                if x > 0:
                    neighbors.append(array[z, y, x - 1])
                if x < array.shape[2] - 1:
                    neighbors.append(array[z, y, x + 1])
                
                # If there are no neighbors (very rare), set ratio = 1
                if not neighbors:
                    ratios[z, y, x] = 1.0
                else:
                    max_neighbor = max(neighbors)
                    # Compute ratio
                    ratios[z, y, x] = max_neighbor / center_value

    return ratios

@njit(cache=True)
def calculate_ratios_stats(array: np.ndarray) -> tuple[float, float]:
    """
    Calculate direction-invariant average and maximum ratio between neighbors,
    counting each pair only once (forward neighbors only).
    Ratio is defined as max(center, neighbor) / min(center, neighbor),
    so it is always >= 1 and does not depend on whether the data is
    ascending or descending.

    :param array: 3D array of values, shape [nz, ny, nx].
    :return: (average_ratio, max_ratio).
    """

    nz, ny, nx = array.shape

    # We have up to 3 "forward" neighbors (x+1, y+1, z+1) per cell,
    # so we create an array big enough for all possible pairs.
    max_pairs = 3 * array.size  
    ratios = np.zeros(max_pairs, dtype=np.float64)
    n_ratios = 0

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                center_value = array[z, y, x]

                # Skip if center is zero or negative (avoid division by zero).
                if center_value <= 0:
                    continue

                # (x+1) neighbor
                if x < nx - 1:
                    neighbor = array[z, y, x+1]
                    if neighbor > 0:
                        hi = max(center_value, neighbor)
                        lo = min(center_value, neighbor)
                        ratios[n_ratios] = hi / lo
                        n_ratios += 1

                # (y+1) neighbor
                if y < ny - 1:
                    neighbor = array[z, y+1, x]
                    if neighbor > 0:
                        hi = max(center_value, neighbor)
                        lo = min(center_value, neighbor)
                        ratios[n_ratios] = hi / lo
                        n_ratios += 1

                # (z+1) neighbor
                if z < nz - 1:
                    neighbor = array[z+1, y, x]
                    if neighbor > 0:
                        hi = max(center_value, neighbor)
                        lo = min(center_value, neighbor)
                        ratios[n_ratios] = hi / lo
                        n_ratios += 1

    # Calculate average and maximum ratios
    if n_ratios > 0:
        average_ratio = np.sum(ratios[:n_ratios]) / n_ratios
        max_ratio = np.max(ratios[:n_ratios])
    else:
        # If no valid ratios were computed, return NaNs
        average_ratio = float('nan')
        max_ratio = float('nan')

    return average_ratio, max_ratio