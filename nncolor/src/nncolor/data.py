import importlib
import cv2
import numpy as np
from enum import Enum
import colorsys
import moviepy.editor as mpe
import moviepy
from typing import *
import random
import pandas as pd
import json
import torch
from icecream import ic
from deprecated import deprecated
from bidict import bidict

DEFAULT_GRID_SHAPE = (4,4)
DEFAULT_NUM_POSITIONS = np.prod(DEFAULT_GRID_SHAPE)
assert DEFAULT_NUM_POSITIONS == 16  
DEFAULT_IMG_SHAPE = (224, 224, 3)
DEFAULT_ROW_LEN = DEFAULT_NUM_POSITIONS**0.5
DEFAULT_RADIUS = int(DEFAULT_IMG_SHAPE[0] / (2 * DEFAULT_ROW_LEN)) # 28
assert DEFAULT_ROW_LEN == 4
assert DEFAULT_RADIUS == 28
COLOR_ID_TO_LABEL = bidict({0: 'orange', 1: 'brown', 2: 'both', 3: 'neither'})
LABEL_TO_COLOR_ID = COLOR_ID_TO_LABEL.inverse
NUM_CLASSES = len(COLOR_ID_TO_LABEL)
COLOR_DIM = 3


# Load experiment data.
# Reference:
# https://github.com/wimglenn/resources-example/blob/master/myapp/example4.py
with importlib.resources.open_text("nncolor.pkdata", 'experiment_1_1_1.csv') as f:
    exp_1_1_data = pd.read_csv(f)


def filter_colors(table, include_colors):
    # TOD: how to properly filter and map pandas data?
    ans = []
    #table[0] = table[0].apply(lambda x : COLOR_ID_TO_LABEL[x])
    #filtered = table[table['ans'] in include_labels]
    for idx, row in table.iterrows():
        label = row['ans']
        label_str = COLOR_ID_TO_LABEL[label]
        if not label_str in include_colors:
            continue
        circle_rgb = json.loads(row['circle_rgb'])
        bg_rgb = json.loads(row['bg_rgb'])
        ans.append((label, circle_rgb, bg_rgb))
    return ans


exp_1_1_data_filtered = filter_colors(exp_1_1_data, 
        include_colors={'orange', 'brown', 'neither'})


class ColorOption:
    def __init__(self, circle_rgb, bg_rgb):
        self.circle_rgb = circle_rgb
        self.bg_rgb = bg_rgb
    
    
def _orange_rgbs():
    return exp_1_1_data.loc[lambda r:r['ans'] == 0, :]


def _brown_rgbs():
    return exp_1_1_data.loc[lambda r:r['ans'] == 1, :]


def color_id_to_rgbs(idx : int):
    return exp_1_1_data.loc[lambda r:r['ans'] == idx, :]


def _random_color(rgbs) -> ColorOption:
    """Chose a random color from the colors given.""" 
    idx = random.randrange(0, rgbs.shape[0])
    circle_rgb  = json.loads(rgbs.iloc[idx]['circle_rgb'])
    bg_rgb = json.loads(rgbs.iloc[idx]['bg_rgb'])
    circle_rgb = np.array(circle_rgb)
    bg_rgb = np.array(bg_rgb)
    return ColorOption(circle_rgb, bg_rgb)


def random_color() -> ColorOption:
    """Returns a (color, label) tuple, either for a brown or orange color."""
    oranges_or_browns = [_orange_rgbs, _brown_rgbs]
    idx = random.randrange(0, 2)
    label = "orange" if idx == 0 else "brown"
    colors = oranges_or_browns[idx]
    return _random_color(colors()), label


def cell_shape(grid_shape, img_shape):
    """Calculates the length in pixels of dimensions of a grid cell."""
    ans  = np.array([img_shape[0] / (grid_shape[0]), 
                     img_shape[1] / (grid_shape[1])])
    return ans


def position_idx(grid_coord, grid_shape):
    """Convert from coord to index.

    Origin is top left.

    Example:
    (0, 1) (0, 2) (0, 3)      0  1  2        
    (1, 1) (1, 2) (1, 3)  →   3  4  5       
    (2, 1) (3, 2) (2, 3)      6  7  8        
    (3, 1) (3, 2) (3, 3)      9  10 11       
          ⋮                      ⋮           
    """
    idx = grid_coord[0] * grid_shape[1] + grid_coord[1]
    return idx


def grid_coords(position_idx : int, grid_shape):
    """Calculates the (y, x) grid-coordinates of a position index.
    
    Origin is top left.

    Example:
    0  1  2        (0, 1) (0, 2) (0, 3) 
    3  4  5    →   (1, 1) (1, 2) (1, 3) 
    6  7  8        (2, 1) (3, 2) (2, 3) 
    9  10 11       (3, 1) (3, 2) (3, 3) 
       ⋮                 ⋮                
    """
    num_positions = grid_shape[0]*grid_shape[1]
    if not 0 <= position_idx < num_positions:
        raise Exception(f'Position index must be within 0 and {num_positions}. '
                        'Got: {position_idx}')
    grid_coord = np.array([position_idx // grid_shape[1],
                           position_idx %  grid_shape[1]])
    return grid_coord


def grid_to_img_coords(g_coords, grid_shape, img_shape):
    """Calculates the (y, x) cell center image coordinates of a grid coordinate.
    
    Origin is top left.
    """
    spacing = cell_shape(grid_shape, img_shape)
    coord = np.around(g_coords * spacing + spacing/2).astype(np.int)
    return coord


def img_coords(position_idx : int, grid_shape, img_shape):
    """Calculates the (y, x) cell center image coordinates of a position index.
    
    Origin is top left.
    """
    g_coords = grid_coords(position_idx, grid_shape)
    # Rounding? Do or don't?
    return grid_to_img_coords(g_coords, grid_shape, img_shape)


def to_cv2_coords(numpy_coords):
    """CV2 uses (x, y) order."""
    return (numpy_coords[1], numpy_coords[0])
    
    
def circle_img(circle_color, bg_color, radius, grid_shape, position_idx, 
        img_shape):
    """Create a circle-background image."""
    if np.shape(img_shape) != (3,):
        raise Exception(f'Expected img_shape to be 3 dimensions '
                        f'(got: {np.shape(img_shape)})')
    img = np.zeros(img_shape, np.float32)
    img[:] = bg_color
    center = to_cv2_coords(img_coords(position_idx, grid_shape, img_shape))
    img = cv2.circle(img, center, radius, circle_color, thickness=-1, 
            lineType=cv2.LINE_AA)
    return img


def label_grid(color_id, grid_shape, position_idx):
    """Creates an "answer" grid, assuming an orange/brown dot is within a cell.
    """
    res = np.full(grid_shape, COLOR_ID_TO_LABEL.inverse['neither'])
    g_coords = grid_coords(position_idx, grid_shape)
    res[tuple(g_coords)] = color_id
    return res


def mask_grid(grid_shape, position_idx):
    """Returns a mask; all values are zero except for the dot position (1)."""
    res = np.zeros(grid_shape)
    g_coords = grid_coords(position_idx, grid_shape)
    res[tuple(g_coords)] = 1
    return res


def linear(t: float) -> float:
    """Linear rate function."""
    return t


def ease_in_out_sine(t : float) -> float:
    """Slow start, slow end sine rate function."""
    t = np.clip(t, 0, 1)
    return -(np.cos(np.pi * t) - 1) / 2


def radial_weight(grid_shape, position_idx, max_dist, rate_fctn):
    """Returns a weight grid useful for training.

    Taking in a rate function, like the ones here:
    https://docs.manim.community/en/stable/reference/manim.utils.rate_functions.html
    """
    g_coord = grid_coords(position_idx, grid_shape)
    y_ticks = np.arange(grid_shape[0])
    x_ticks = np.arange(grid_shape[1])
    yv, xv, = np.meshgrid(y_ticks, x_ticks, indexing='ij')
    dist = np.sqrt(np.square(yv - g_coord[0]) + np.square(xv - g_coord[1]))
    dist /= max_dist
    weighting = rate_fctn(dist)
    total_weight = np.sum(weighting)
    weighting[tuple(g_coord)] = total_weight
    return weighting


def radial_weight_for_every_pos(grid_shape, max_dist, rate_fctn):
    """Repeat radial_weighted_loss, for every dot position."""
    num_positions = np.prod(grid_shape)
    res = np.zeros((num_positions, *grid_shape))
    for i in range(num_positions):
        res[i] = radial_weight(grid_shape, i, max_dist, rate_fctn)
    return res


def draw_overlay(img, labels):
    """Draw a grid overlay with labels."""
    color = (0, 0, 255)
    grid_shape = labels.shape
    # Spacing is HxW, but cv2 uses (x, y) point coordinates!
    spacing = cell_shape(grid_shape, img.shape)
    # Horizontal lines.
    h = spacing[0] 
    while h < img.shape[0]:
        # Sad that cv2 requires rounding. Would prefer an anti-aliased sample
        # from an arbitrary line.
        cv2.line(img, (0, round(h)), (img.shape[1], round(h)), 
                color, thickness=1)
        h += spacing[0]
    # Vertical lines
    v = spacing[1] 
    while v < img.shape[0]:
        # Sad that cv2 requires rounding. Would prefer an anti-aliased sample
        # from an arbitrary line.
        cv2.line(img, (round(v), 0), (round(v), img.shape[1]), 
                color, thickness=1)
        v += spacing[0]
    # Labels
    spacing = cell_shape(grid_shape, img.shape)
    for iy, ix in np.ndindex(labels.shape):
        # Why isn't flip correct? This is suggestive of a bug somewhere,
        # or my misunderstanding of the indexing for cv2.
        # padding_offset = 0.1
        #cell_lower_left = np.flip(cell_lower_left)
        #cell_lower_left = np.around(np.array(
        #    [iy + 1.0 - padding_offset, ix + padding_offset])
        #   * spacing).astype(np.int)
        padding_offset = 0.1
        cell_lower_left = np.around(np.array(
            [iy + padding_offset, ix +1.0 - padding_offset])
            * spacing).astype(np.int)
        color = (0, 0, 255)
        font_scale = 0.6
        thickness = 1
        cv2.putText(img, str(labels[iy,ix]), cell_lower_left, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness, cv2.LINE_AA) 

    return img

    
class DotImgGen:
    """Wraps cricle_img to reduce function parameters.

    We could use patials for this, but it's hard to figure out parameter 
    orders then.""" 
    def __init__(self, grid_shape, img_shape, dot_radius):
        self.grid_shape = grid_shape
        self.img_shape = img_shape
        self.dot_radius = dot_radius

    def gen(self, circle_color, bg_color, radius, pos):
        return circle_img(circle_color, bg_color, radius, self.grid_shape,
            pos, self.img_shape)


def create_samples(num_samples, radius=DEFAULT_RADIUS, 
        grid_shape=DEFAULT_GRID_SHAPE, img_shape=DEFAULT_IMG_SHAPE):
    frames = []
    labels = []
    for n in range(num_samples):
        color, label = random_color()
        num_positions = np.prod(grid_shape)
        pos = random.randrange(0, num_positions)
        frame = circle_img(color.circle_rgb, color.bg_rgb, radius, grid_shape,
            pos, img_shape)
        frames.append(frame)
        labels.append(label)
    return frames, labels


class ColorDotDataset(torch.utils.data.Dataset):
    "Colored circles on colored background, a dataset."
    
    def __init__(self, labelled_colors, dot_radius=DEFAULT_RADIUS, 
            grid_shape=DEFAULT_GRID_SHAPE, img_shape=DEFAULT_IMG_SHAPE, 
            transform=None):
        """Generate dataset from an array of (label, circle_rgb, bg_rgb) tuples.
        """
        self._labelled_colors = labelled_colors
        self.dot_radius = dot_radius
        self.grid_shape = grid_shape
        self.img_shape = img_shape
        self.transform = transform
        if len(grid_shape) != 2:
            raise Exception("Only 2D grids are supported.")
        self.grid_shape = grid_shape
        self._rng = np.random.default_rng(123)
        self._shuffled_idxs = np.arange(self.__len__())
        self._rng.shuffle(self._shuffled_idxs)
        self._img_gen = DotImgGen(self.grid_shape, 
                self.img_shape, self.dot_radius)
        
    def __len__(self):
        return len(self._labelled_colors) * self.num_positions()

    def num_positions(self):
        return self.grid_shape[0]*self.grid_shape[1]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self._shuffled_idxs[idx]    
        color_idx = idx // self.num_positions()
        pos = idx % self.num_positions()
        label, circle_rgb, bg_rgb = self._labelled_colors[color_idx]
        img = self._img_gen.gen(circle_rgb, bg_rgb, self.dot_radius, pos)
        label_g = label_grid(label, self.grid_shape, pos)
        mask_g = mask_grid(self.grid_shape, pos)
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'label': label, 'label_grid': label_g, 
                'position': pos, 'mask_grid': mask_g}
        #return (img, label)
    
    
def train_test_val_split(labelled_colors, 
        dot_radius=DEFAULT_RADIUS, grid_shape=DEFAULT_GRID_SHAPE, 
        img_shape=DEFAULT_IMG_SHAPE, split_ratio=[11, 6, 1]):
    divisions = np.sum(np.array(split_ratio))
    num_per_division = len(labelled_colors) // divisions
    remainder = len(labelled_colors) % divisions
    num_train = num_per_division * split_ratio[0] + remainder
    num_test = num_per_division * split_ratio[1]
    num_val = num_per_division * split_ratio[2]
    
    train_ds = ColorDotDataset(labelled_colors[0: num_train], 
            dot_radius, grid_shape, img_shape)
    test_ds = ColorDotDataset(labelled_colors[num_train: num_train + num_test],
            dot_radius, grid_shape, img_shape)
    val_ds = ColorDotDataset(labelled_colors[num_train + num_test:],
            dot_radius, grid_shape, img_shape)
    return (train_ds, test_ds, val_ds)


@deprecated
def load_datasets():
    """Just use the global variable exp_1_1_data_filtered instead."""
    colors = filter_colors(exp_1_1_data, 
                           include_colors={'orange', 'brown', 'neither'})
    datasets = train_test_val_split(colors)
    return datasets
    
