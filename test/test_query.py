import pytest
import numpy as np
from wwpy.utils import get_closest_indices, get_range_indices, get_bin_intervals_from_indices

@pytest.fixture
def grid1():
    return np.array([-2.50000e+01, 1.50000e+00, 2.80000e+01, 5.45000e+01, 8.10000e+01,
                      1.07500e+02, 1.09167e+02, 1.10833e+02, 1.12500e+02])

@pytest.fixture
def grid2():
    return np.array([1.0000e-08, 3.0000e-08, 5.0000e-08, 1.0000e-07, 2.2500e-07,
                     3.2500e-07, 4.1399e-07, 8.0000e-07, 1.0000e-06, 1.1253e-06,
                     1.3000e-06, 1.8554e-06, 3.0590e-06, 1.0677e-05, 2.9023e-05,
                     1.0130e-04, 5.8295e-04, 3.0354e-03, 1.5034e-02, 1.1109e-01,
                     4.0762e-01, 9.0718e-01, 1.4227e+00, 1.8268e+00, 3.0119e+00,
                     6.3763e+00, 2.0000e+01])

@pytest.fixture
def grid3():
    return np.array([0.])


# -----------------------------------
# Test get_closest_indices
# -----------------------------------
def test_exact_match(grid1):
    assert np.array_equal(get_closest_indices(grid1, 28.0), [1, 3])
    assert np.array_equal(get_closest_indices(grid1, -25.0), [0, 1])
    assert np.array_equal(get_closest_indices(grid1, 112.5), [7, 8])

def test_between_values(grid1):
    assert np.array_equal(get_closest_indices(grid1, 15.0), [1, 2])
    assert np.array_equal(get_closest_indices(grid1, 108.0), [5, 6])

def test_out_of_bounds(grid1):
    assert np.array_equal(get_closest_indices(grid1, -30.0), [0, 1])
    assert np.array_equal(get_closest_indices(grid1, 200.0), [7, 8])

def test_small_values(grid2):
    assert np.array_equal(get_closest_indices(grid2, 1e-8), [0, 1])
    assert np.array_equal(get_closest_indices(grid2, 4e-7), [5, 6])
    assert np.array_equal(get_closest_indices(grid2, 1e-6), [7, 9])

def test_large_values(grid2):
    assert np.array_equal(get_closest_indices(grid2, 1.0), [21, 22])
    assert np.array_equal(get_closest_indices(grid2, 20.0), [25, 26])

def test_single_value_grid(grid3):
    assert np.array_equal(get_closest_indices(grid3, 0.0), [0, np.inf])
    assert np.array_equal(get_closest_indices(grid3, -1.0), [0, np.inf])
    assert np.array_equal(get_closest_indices(grid3, 1.0), [0, np.inf])


# -----------------------------------
# Test get_range_indices
# -----------------------------------
# Test cases for grid1
def test_get_range_indices_grid1_exact_match_min(grid1):
    # Exact match for min
    range_tuple = (-25.0, 50.0)
    expected_indices = np.arange(0, 4)  
    assert np.array_equal(get_range_indices(grid1, range_tuple), expected_indices)

def test_get_range_indices_grid1_exact_match_max(grid1):
    # Exact match for max
    range_tuple = (40.0, 112.5)
    expected_indices = np.arange(2, 9) 
    assert np.array_equal(get_range_indices(grid1, range_tuple), expected_indices)

def test_get_range_indices_grid1_exact_match_both(grid1):
    # Exact match for both min and max
    range_tuple = (1.5, 107.5)
    expected_indices = np.arange(1, 6)  
    assert np.array_equal(get_range_indices(grid1, range_tuple), expected_indices)

def test_get_range_indices_grid1_between(grid1):
    # Range between grid points
    range_tuple = (2.0, 54.0)
    expected_indices = np.arange(1, 4)  
    assert np.array_equal(get_range_indices(grid1, range_tuple), expected_indices)

def test_get_range_indices_grid1_out_of_bounds_low(grid1):
    # Lower bound below grid
    range_tuple = (-30.0, 10.0)
    expected_indices = np.arange(0, 3) 
    assert np.array_equal(get_range_indices(grid1, range_tuple), expected_indices)

def test_get_range_indices_grid1_out_of_bounds_high(grid1):
    # Upper bound above grid
    range_tuple = (50.0, 200.0)
    expected_indices = np.arange(2, 9) 
    assert np.array_equal(get_range_indices(grid1, range_tuple), expected_indices)

def test_get_range_indices_grid1_out_of_bounds_both(grid1):
    # Both bounds out of grid
    range_tuple = (-100.0, 200.0)
    expected_indices = np.arange(0, 9) 
    assert np.array_equal(get_range_indices(grid1, range_tuple), expected_indices)

# Test cases for grid2
def test_get_range_indices_grid2_exact_match_min(grid2):
    # Exact match for min
    range_tuple = (1.0000e-08, 4.0000e-08)
    expected_indices = np.arange(0, 3)  
    assert np.array_equal(get_range_indices(grid2, range_tuple), expected_indices)

def test_get_range_indices_grid2_exact_match_max(grid2):
    # Exact match for max
    range_tuple = (1.5000e-06, 2.9023e-05)
    expected_indices = np.arange(10, 15)  
    assert np.array_equal(get_range_indices(grid2, range_tuple), expected_indices)

def test_get_range_indices_grid2_exact_match_both(grid2):
    # Exact match for both min and max
    range_tuple = (3.0e-08, 1.0e-06)
    expected_indices = np.arange(1, 9)  
    assert np.array_equal(get_range_indices(grid2, range_tuple), expected_indices)

def test_get_range_indices_grid2_between(grid2):
    # Range between grid points
    range_tuple = (4.5000e-08, 1.4000e-07)
    expected_indices = np.arange(1, 5)  
    assert np.array_equal(get_range_indices(grid2, range_tuple), expected_indices)

def test_get_range_indices_grid2_out_of_bounds_low(grid2):
    # Lower bound below grid
    range_tuple = (0.0, 1.0e-07)
    expected_indices = np.arange(0, 4)  
    assert np.array_equal(get_range_indices(grid2, range_tuple), expected_indices)

def test_get_range_indices_grid2_out_of_bounds_high(grid2):
    # Upper bound above grid
    range_tuple = (1.0e-06, 3.0e+01)
    expected_indices = np.arange(8, 27)  
    assert np.array_equal(get_range_indices(grid2, range_tuple), expected_indices)

def test_get_range_indices_grid2_out_of_bounds_both(grid2):
    # Both bounds out of grid
    range_tuple = (-1.0e-05, 5.0e+01)
    expected_indices = np.arange(0, 27)  
    assert np.array_equal(get_range_indices(grid2, range_tuple), expected_indices)

# Test cases for grid3
def test_get_range_indices_grid3(grid3):
    # Exact match for min and max
    range_tuple = (0.0, np.inf)
    expected_indices = np.array([0, np.inf])
    assert np.array_equal(get_range_indices(grid3, range_tuple), expected_indices)

def test_get_range_indices_grid3_between(grid3):
    # Range around the single grid point
    range_tuple = (-1.0, 1.0)
    expected_indices = np.array([0, np.inf])
    assert np.array_equal(get_range_indices(grid3, range_tuple), expected_indices)

def test_get_range_indices_grid3_out_of_bounds_low(grid3):
    # Lower bound below grid (same as above since grid has one point)
    range_tuple = (-10.0, -1.0)
    expected_indices = np.array([0, np.inf])
    assert np.array_equal(get_range_indices(grid3, range_tuple), expected_indices)

def test_get_range_indices_grid3_out_of_bounds_high(grid3):
    # Upper bound above grid
    range_tuple = (1.0, 10.0)
    expected_indices = np.array([0, np.inf])
    assert np.array_equal(get_range_indices(grid3, range_tuple), expected_indices)

def test_get_range_indices_grid3_out_of_bounds_both(grid3):
    # Both bounds out of grid
    range_tuple = (-5.0, 5.0)
    expected_indices = np.array([0, np.inf])
    assert np.array_equal(get_range_indices(grid3, range_tuple), expected_indices)



# -----------------------------------
# Test get_bin_intervals_from_indices
# -----------------------------------

def test_get_bin_intervals_from_indices_grid2(grid2):
    # Test Case 1: indices = [0,1]
    indices = np.array([0, 1])
    expected_start = np.array([1.0000e-08])
    expected_end = np.array([3.0000e-08])
    bin_starts, bin_ends = get_bin_intervals_from_indices(grid2, indices)
    np.testing.assert_allclose(
        bin_starts, expected_start, equal_nan=True,
        err_msg="Bin starts do not match expected values for indices [0, 1] with grid2."
    )
    np.testing.assert_allclose(
        bin_ends, expected_end, equal_nan=True,
        err_msg="Bin ends do not match expected values for indices [0, 1] with grid2."
    )

    # Test Case 2: indices = [0, np.inf]
    indices = np.array([0, np.inf])
    expected_start = np.array([0.0])
    expected_end = np.array([np.inf])
    bin_starts, bin_ends = get_bin_intervals_from_indices(grid2, indices)
    np.testing.assert_allclose(
        bin_starts, expected_start, equal_nan=True,
        err_msg="Bin starts do not match expected values for indices [0, inf] with grid2."
    )
    np.testing.assert_allclose(
        bin_ends, expected_end, equal_nan=True,
        err_msg="Bin ends do not match expected values for indices [0, inf] with grid2."
    )

    # Test Case 3: indices = [2, 5]
    indices = np.array([2, 5])
    expected_start = np.array([5.0000e-08, 1.0000e-07, 2.2500e-07])
    expected_end = np.array([1.0000e-07, 2.2500e-07, 3.2500e-07])
    bin_starts, bin_ends = get_bin_intervals_from_indices(grid2, indices)
    np.testing.assert_allclose(
        bin_starts, expected_start, equal_nan=True,
        err_msg="Bin starts do not match expected values for indices [2, 5] with grid2."
    )
    np.testing.assert_allclose(
        bin_ends, expected_end, equal_nan=True,
        err_msg="Bin ends do not match expected values for indices [2, 5] with grid2."
    )

    # Test Case 4: indices = [25, 26]
    indices = np.array([25, 26])
    expected_starts = np.array([6.3763e+00])
    expected_ends = np.array([2.0000e+01])
    bin_starts, bin_ends = get_bin_intervals_from_indices(grid2, indices)
    np.testing.assert_allclose(
        bin_starts, expected_starts, equal_nan=True,
        err_msg="Bin starts do not match expected values for the last index with grid2."
    )
    np.testing.assert_allclose(
        bin_ends, expected_ends, equal_nan=True,
        err_msg="Bin ends do not match expected NaN for the last index with grid2."
    )

def test_get_bin_intervals_from_indices_grid3(grid3):
    indices = np.array([0, np.inf])
    expected_starts = np.array([0.0])
    expected_ends = np.array([np.inf])
    bin_starts, bin_ends = get_bin_intervals_from_indices(grid3, indices)
    np.testing.assert_allclose(
        bin_starts, expected_starts, equal_nan=True,
        err_msg="Bin starts do not match expected values for indices [0] with grid3."
    )
    np.testing.assert_allclose(
        bin_ends, expected_ends, equal_nan=True,
        err_msg="Bin ends do not match expected values for indices [0] with grid3."
    )
    