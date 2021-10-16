import pytest
from icecream import ic
import math
import numpy as np
import nncolor as nc
import nncolor.data 


def test_position_idx():
    """Tests coord to position conversion.

    3x4 example:
    (0, 0) (0, 1) (0, 2)     0  1  2  
    (1, 0) (1, 1) (1, 2)     3  4  5
    (2, 0) (3, 1) (2, 2)     6  7  8
    (3, 0) (3, 1) (3, 2)     9  10 11
    """
    shape = (4,3)
    assert nc.data.position_idx((0, 0), shape) == 0
    assert nc.data.position_idx((0, 1), shape) == 1
    assert nc.data.position_idx((0, 2), shape) == 2
    assert nc.data.position_idx((1, 0), shape) == 3
    assert nc.data.position_idx((2, 0), shape) == 6
    assert nc.data.position_idx((2, 1), shape) == 7


def test_grid_coords():
    """Tests position to coord conversion.

    4x3 example:
    0  1  2        (0, 1) (0, 2) (0, 3) 
    3  4  5    →   (1, 1) (1, 2) (1, 3) 
    6  7  8        (2, 1) (3, 2) (2, 3) 
    9  10 11       (3, 1) (3, 2) (3, 3) 
    """
    shape = (4, 3)
    assert np.array_equal(nc.data.grid_coords(0, shape), (0, 0))
    assert np.array_equal(nc.data.grid_coords(1, shape), (0, 1))
    assert np.array_equal(nc.data.grid_coords(2, shape), (0, 2))
    assert np.array_equal(nc.data.grid_coords(3, shape), (1, 0))
    assert np.array_equal(nc.data.grid_coords(6, shape), (2, 0))
    assert np.array_equal(nc.data.grid_coords(7, shape), (2, 1))


def test_radial_weighted_loss():
    # 1.
    # Setup
    grid_shape = (5, 5)
    pos_idx1 = nc.data.position_idx((2,2), grid_shape)
    max_dist = 2
    ease_fn = nc.data.ease_in_out_sine
    a1 = ease_fn(1/max_dist)
    a2 = ease_fn(math.sqrt(2)/max_dist)
    a3 = 16*1 + 4*a2 + 4*a1
    ans1 = np.array([
        [1, 1,  1,  1,  1],
        [1, a2, a1, a2, 1],
        [1, a1, a3, a1, 1],
        [1, a2, a1, a2, 1],
        [1, 1,  1,  1,  1]])

    # Test
    res = nc.data.radial_weighted_loss(grid_shape, pos_idx1, max_dist, ease_fn)
    assert np.allclose(res, ans1)

    # 2.
    pos_idx2 = nc.data.position_idx((2,0), grid_shape)
    a3 = 19*1 + 2*a2 + 3*a1
    ans2 = np.array([
        [1,  1,  1, 1, 1],
        [a1, a2, 1, 1, 1],
        [a3, a1, 1, 1, 1],
        [a1, a2, 1, 1, 1],
        [1,  1,  1, 1, 1]])

    # Test
    res = nc.data.radial_weight(grid_shape, pos_idx2, max_dist, ease_fn)
    assert np.allclose(res, ans2)

    # 3.
    res = nc.data.loss_for_every_pos(grid_shape, max_dist, ease_fn)
    assert np.allclose(res[pos_idx1], ans1)
    assert np.allclose(res[pos_idx2], ans2)





