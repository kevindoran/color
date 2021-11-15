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


def test_radial_weight():
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
    res = nc.data.radial_weight(grid_shape, pos_idx1, max_dist, ease_fn)
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
    res = nc.data.radial_weight_for_every_pos(grid_shape, max_dist, ease_fn)
    assert np.allclose(res[pos_idx1], ans1)
    assert np.allclose(res[pos_idx2], ans2)



def test_color_counts():
    # Setup
    colors = nc.data.exp_1_1_data_filtered
    # Precomputed counts:
    ans = np.array([39, 29, 0, 148])

    # Test
    res = nc.data.color_counts(colors)
    assert np.array_equal(ans, res)


#def test_datasets():
#    colors = nc.data.exp_1_1_data_filtered
#    grid_shape = (7,7)
#    num_elements = np.prod(grid_shape)
#    train_ds, test_ds, val_ds = nc.data.train_test_val_split(
#            colors, dot_radius=10, grid_shape=grid_shape)
#    dsl = [train_ds, test_ds, val_ds]
#    
#
#    # Test that the correct number of images for each color have been created.
#    for ds in dsl:
#        import pdb; pdb.set_trace();
#        color_tally = ds._labelled_colors['ans'].value_counts().to_dict()
#        img_label_tally = {k:0 for k in color_tally.keys()}
#        for sample in ds:
#            label = sample['label']
#            img_label_tally[label] = img_label_tally[label] + 1
#        for k,v in color_tally.items():
#            assert img_label_tally[k] == v*num_elements







