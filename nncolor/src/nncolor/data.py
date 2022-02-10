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
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn as sk
import sklearn.metrics
import sklearn.linear_model
import colorsys

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
DEFAULT_SPLIT = (10, 6, 2)


# Load experiment data.
# Reference:
# https://github.com/wimglenn/resources-example/blob/master/myapp/example4.py
"""
It's worth noting that the naming convention for the data files is not 
consistent with the experiment naming. I should have named the first
data file as experiment_1.csv. But I'll keep it as it is, as many notebooks
rely on it.
"""
with importlib.resources.open_text("nncolor.pkdata", 'experiment_1_1_1.csv') as f:
    exp_1_1_1_data = pd.read_csv(f)

"""
The original exp_1_1_1_data had two problems:
    - it recorded the initial starting stimulus (red circle, white background) 
      with a classification of "brown". This is a mistake.
    - it used a non-flat structure which make loading it a slight hassle in 
      comparison to a simple flat format.

With these issues in mind, the file was updated by removing the first
data entry, and by reformatting the file. This result is stored in 
experiment_1_1_1_v2.
"""
with importlib.resources.open_text("nncolor.pkdata", 'experiment_1_1_1_v2.csv') as f:
    exp_1_1_1_v2_data = pd.read_csv(f)

"""
Experiment 1.1.2 recorded ~1000 entries. It was collected on a separate day.
"""
with importlib.resources.open_text("nncolor.pkdata", 'experiment_1_1_2.csv') as f:
    exp_1_1_2_data = pd.read_csv(f)

"""
All experiment data are combined into the table below. This currently includes 
data from experiment_1_1_1_v2 and experiment_1_1_2.
"""
with importlib.resources.open_text("nncolor.pkdata", 'experiment_1_1_combined.csv') as f:
    exp_1_1_combined_data = pd.read_csv(f)


"""
The variable used for experiment 1.1.1 data was originally called exp_1_1_data.
For backward compatibility, we will allow this identifier to continue to refer 
to exp_1_1_1_data.
"""
exp_1_1_data = exp_1_1_1_data


def split(data, split_ratio=DEFAULT_SPLIT):
    """Splits the data into train, test and val."""
    divisions = np.sum(np.array(split_ratio)) 
    num_per_division = len(data) // divisions
    remainder = len(data) % divisions
    num_train = num_per_division * split_ratio[0] + remainder
    num_test = num_per_division * split_ratio[1]
    num_val = num_per_division * split_ratio[2]
    train_data = data[0:num_train]
    test_data = data[num_train:num_train + num_test]
    val_data = data[num_train + num_test:]
    assert len(val_data) + len(test_data) + len(train_data) == len(data)
    return train_data, test_data, val_data


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


def deserialize(table):
    """Convert Pandas data to a nested list: [ans, [r,g,b], [r,g,b]].

    The returned list is of the form:
        [
        (answer id, circle rgb, background rgb),
        (answer id, circle rgb, background rgb),
        ...
        ]
    """
    res = []
    for idx, row in table.iterrows():
        res.append((int(row['ans']), [row['circle_r'], row['circle_g'], 
            row['circle_b']], [row['bg_r'], row['bg_g'], row['bg_b']]))
    return res


def exp_data_in_hsv(data):
	"""Convert the Pandas data table to use HSV values instead of RGB."""
	data_hsv = pd.concat([pd.DataFrame([
		[row['ans'], *colorsys.rgb_to_hsv(*json.loads(row['circle_rgb'])), *colorsys.rgb_to_hsv(*json.loads(row['bg_rgb']))]],
		columns=['ans', 'circle hue', 'circle sat', 'circle val', 'bg hue', 'bg sat', 'bg val'])
	 for idx, row in data.iterrows()])
	return data_hsv


class UnwantedColorFilter:
	"""Creates a test to ignore color pairs that we are not interested in."""
	
	def __init__(self):
		# Choose a class weight that results in 100% recall. Trial and error
		# gives us 1:9.
		class_0_colors = (3,)
		class_1_colors = (0, 1, 2) 
		class_weight = {0:1, 1:9}	
		self.model = sk.linear_model.LogisticRegression(
			solver='liblinear',
			class_weight=class_weight)
		X, y = UnwantedColorFilter.data_as_Xy(class_0_colors, class_1_colors)
		self.model.fit(X, y)
		y_predict = self.model.predict(X)
		recall = sk.metrics.recall_score(y, y_predict)
		if recall != 1.0:
			raise Exception("The recall is expected to be zero. Maybe the " 
							"class weight needs to be updated. We don't want " 
							"any false negatives (false positives are fine).")

	@staticmethod
	def data_as_Xy(class_0, class_1):
		all_classes = class_0 +  class_1
		data_hsv = exp_data_in_hsv(exp_1_1_data)
		filtered = data_hsv[data_hsv['ans'].isin(all_classes)]
		X =  data_hsv[['circle hue', 'circle sat', 'circle val', 
   					   'bg val']].to_numpy()
		y = filtered['ans']
		y = y.apply(lambda i : 0 if i in class_0 else 1)
		return X, y

	def is_neither_with_high_confidence(self, circle_rgb, background_rgb):
		c_hsv = colorsys.rgb_to_hsv(*circle_rgb)
		bg_hsv = colorsys.rgb_to_hsv(*background_rgb)
		res = not self.model.predict(np.array([[*c_hsv, bg_hsv[0]]]))
		return res

_unwanted_color_filter = UnwantedColorFilter()


def is_neither_with_high_confidence(circle_rgb, background_rgb):
	return _unwanted_color_filter.is_neither_with_high_confidence(
		circle_rgb, background_rgb)


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


def color_counts(color_data):
    """Calculates the number of color codes of each color class.
    
    Args:
      color_data: pandas table, like exp_1_1_data.

    Returns a numpy array of shape (NUM_CLASSES,)
    """
    labels = np.array([c[0] for c in color_data])
    unique, counts = np.unique(labels, return_counts=True)
    res = np.zeros(NUM_CLASSES)
    res.put(unique, counts)
    return res


def plot_colors():
    orange_marker_color = '#ffa219'
    brown_marker_color = '#473d28'
    both_marker_color = '#9c7741'
    neither_marker_color = '#dddec9'
    ans = np.array([orange_marker_color, 
           brown_marker_color,
           both_marker_color,
           neither_marker_color])
    return ans


def color_legend():
    """Creates a Matplotlib figure showing the color codes used."""
    # orange, brown, both, neither
    colors_as_vec = np.array([[
        mpl.colors.to_rgb(c) for c in plot_colors()
        ]])
    fig, ax = plt.subplots()
    ax.imshow(colors_as_vec)
    ax.set_xticklabels(['orange', 'brown', 'both', 'neither'])
    plt.xticks(np.arange(0, 4, 1.0))
    ax.get_yaxis().set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor");
    ax.set_title("Figure color map");
    return fig


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
    

def circle_img(circle_color, bg_color, radius, grid_shape, position, 
        img_shape, pos_offset=None):
    """Create a circle-background image."""
    if np.shape(img_shape) != (3,):
        raise Exception(f'Expected img_shape to be 3 dimensions '
                        f'(got: {np.shape(img_shape)})')
    img = np.zeros(img_shape, np.float32)
    img[:] = bg_color
    if grid_shape != None:
        center = to_cv2_coords(img_coords(position, grid_shape, img_shape)) 
        center = (np.array(center) + np.array(pos_offset)).astype(np.uint8)
    else:
        center = position
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
    """Draw a grid overlay with labels.
    
    Args:
        img: image as numpy array, with 0-1 color values.
        labels: color labels for each grid cell.
    """
    color = [0.0, 0.0, 1.0]
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
        font_scale = 0.6
        thickness = 1
        cv2.putText(img, str(labels[iy,ix]), cell_lower_left, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness, cv2.LINE_AA) 
    # Urh. Why can't we have float colors with cv2.
    return img

    
class DotImgGen:
    """Wraps cricle_img to reduce function parameters.

    We could use patials for this, but it's hard to figure out parameter 
    orders then.""" 
    def __init__(self, grid_shape, img_shape, dot_radius, dot_offset):
        self.grid_shape = grid_shape
        self.img_shape = img_shape
        self.dot_radius = dot_radius
        self.dot_offset = dot_offset

    def gen(self, circle_color, bg_color, radius, pos):
        return circle_img(circle_color, bg_color, radius, self.grid_shape,
            pos, self.img_shape, self.dot_offset)


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
            transform=None, dot_offset=0.0):
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
        self.dot_offset = dot_offset
        self._rng = np.random.default_rng(123)
        self._shuffled_idxs = np.arange(self.__len__())
        self._rng.shuffle(self._shuffled_idxs)
        self._img_gen = DotImgGen(self.grid_shape, 
                self.img_shape, self.dot_radius, self.dot_offset)

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
        img_shape=DEFAULT_IMG_SHAPE, split_ratio=(10, 6, 2), dot_offset=(0,0)):
    train_data, test_data, val_data = split(labelled_colors, split_ratio)
    train_ds = ColorDotDataset(train_data, dot_radius, grid_shape, img_shape, 
            dot_offset=dot_offset)
    test_ds = ColorDotDataset(test_data, dot_radius, grid_shape, img_shape, 
            dot_offset=dot_offset)
    val_ds = ColorDotDataset(val_data, dot_radius, grid_shape, img_shape, 
                dot_offset=dot_offset)
    return (train_ds, test_ds, val_ds)


@deprecated
def load_datasets():
    """Just use the global variable exp_1_1_data_filtered instead."""
    colors = filter_colors(exp_1_1_data, 
                           include_colors={'orange', 'brown', 'neither'})
    datasets = train_test_val_split(colors)
    return datasets
    
