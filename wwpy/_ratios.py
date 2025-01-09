# wwpy/ratios.py

"""
Performance-optimized functions for ratio calculations in weight window meshes.

This module uses Numba for improved computational performance when calculating
ratios between neighboring cells in 3D arrays.
"""

import numpy as np
from numba import njit

@njit(cache=True)
def calculate_max_ratio_array(array: np.ndarray) -> np.ndarray:
    """
    Calculate the maximum ratio between each cell and its neighbors in a 3D array.

    Uses Numba acceleration to efficiently compute ratios between neighboring cells
    in a three-dimensional mesh. Only considers the six direct neighbors (faces)
    of each cell.

    :param array: 3D input array of weight window values
    :type array: np.ndarray
    :return: 3D array containing maximum neighbor ratios for each cell
    :rtype: np.ndarray
    :note: Border cells are assigned a ratio of 1.0
    :note: If a cell has value 0, its ratio is set to 1.0
    :note: Uses @njit for performance optimization
    """
    # Initialize the ratios array with ones
    ratios = np.ones_like(array)
    
    # Iterate over the array to calculate ratios
    for z in range(1, array.shape[0] - 1):
        for y in range(1, array.shape[1] - 1):
            for x in range(1, array.shape[2] - 1):
                center_value = array[z, y, x]
                neighbors = [
                    array[z-1, y, x], array[z+1, y, x],
                    array[z, y-1, x], array[z, y+1, x],
                    array[z, y, x-1], array[z, y, x+1]
                ]
                max_neighbor = max(neighbors)
                if center_value != 0:
                    ratios[z, y, x] = max_neighbor / center_value
    
    return ratios