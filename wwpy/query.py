"""
Query module for weight window data retrieval.
Provides the QueryResult class for structured access to weight window query results.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from wwinpy.header import Header

@dataclass
class QueryResult:
    """Store the results of a weight window query.

    :param header: Weight window file header information
    :type header: Header
    :param particle_types: List of particle type identifiers
    :type particle_types: List[int]
    :param ww_values: List of weight window values arrays, one per particle type
    :type ww_values: List[np.ndarray]
    :param energy_intervals: List of energy interval pairs (starts, ends) for each particle type
    :type energy_intervals: List[Tuple[np.ndarray, np.ndarray]]
    :param time_intervals: List of time interval pairs (starts, ends) for each particle type
    :type time_intervals: List[Tuple[np.ndarray, np.ndarray]]
    :param x_intervals: Spatial interval pairs (starts, ends) for x-direction
    :type x_intervals: Tuple[np.ndarray, np.ndarray]
    :param y_intervals: Spatial interval pairs (starts, ends) for y-direction
    :type y_intervals: Tuple[np.ndarray, np.ndarray]
    :param z_intervals: Spatial interval pairs (starts, ends) for z-direction
    :type z_intervals: Tuple[np.ndarray, np.ndarray]
    """

    header: Header
    particle_types: List[int]
    ww_values: List[np.ndarray]
    energy_intervals: List[Tuple[np.ndarray, np.ndarray]]
    time_intervals: List[Tuple[np.ndarray, np.ndarray]]
    x_intervals: Tuple[np.ndarray, np.ndarray]
    y_intervals: Tuple[np.ndarray, np.ndarray]
    z_intervals: Tuple[np.ndarray, np.ndarray]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert query results to a pandas DataFrame.

        :return: DataFrame containing weight window data with particle type, spatial,
                energy, and time information
        :rtype: pd.DataFrame
        """
        data_rows = []
        
        for p_idx, p_type in enumerate(self.particle_types):
            e_starts, e_ends = self.energy_intervals[p_idx]
            t_starts, t_ends = self.time_intervals[p_idx]
            x_starts, x_ends = self.x_intervals
            y_starts, y_ends = self.y_intervals
            z_starts, z_ends = self.z_intervals
            ww_vals = self.ww_values[p_idx]
            
            if len(t_starts) == 0 and len(t_ends) == 0:
                t_starts = np.array([0.0])
                t_ends = np.array([np.inf])
                
            # Get array shape
            time_dim = len(t_starts)
            for t_idx, (t_start, t_end) in enumerate(zip(t_starts, t_ends)):
                for e_idx, (e_start, e_end) in enumerate(zip(e_starts, e_ends)):
                    for z_idx, (z_start, z_end) in enumerate(zip(z_starts, z_ends)):
                        for y_idx, (y_start, y_end) in enumerate(zip(y_starts, y_ends)):
                            for x_idx, (x_start, x_end) in enumerate(zip(x_starts, x_ends)):
                                t_index = t_idx if time_dim > 1 else 0
                                data_rows.append({
                                    'particle_type': p_type,
                                    'time_start': t_start,
                                    'time_end': t_end,
                                    'energy_start': e_start,
                                    'energy_end': e_end,
                                    'x_start': x_start,
                                    'x_end': x_end,
                                    'y_start': y_start,
                                    'y_end': y_end,
                                    'z_start': z_start,
                                    'z_end': z_end,
                                    'ww_value': float(ww_vals[t_index, e_idx, z_idx, y_idx, x_idx])
                                })
        
        return pd.DataFrame(data_rows)
