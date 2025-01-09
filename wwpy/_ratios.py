"""Module providing optimized ratio calculations for weight window meshes.

Uses Numba-accelerated functions for efficient ratio calculations between neighboring cells.
"""

import numpy as np
from numba import njit

@njit(cache=True)
def calculate_max_ratio_array(array: np.ndarray) -> np.ndarray:
    """Calculate maximum ratios between each cell and its neighbors.

    :param array: 3D input array of weight window values
    :type array: np.ndarray
    :return: 3D array of maximum neighbor ratios
    :rtype: np.ndarray
    :note: Border cells get ratio 1.0
    :note: Cells with value 0 get ratio 1.0
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