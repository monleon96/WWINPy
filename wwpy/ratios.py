# wwpy/ratios.py

"""
Includes some numba functions for improved performance of the ratios calculation.
"""

import numpy as np


@njit(cache=True)
def calculate_max_ratio_array(array: np.ndarray) -> np.ndarray:
    """
    Calculate the maximum ratio between each cell and its neighbors in a 3D array.

    Parameters
    ----------
    array : np.ndarray
        The input 3D array.

    Returns
    -------
    np.ndarray
        A 3D array of the same shape as the input, where each value is the maximum
        ratio of the corresponding value in the input array to its neighbours.
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