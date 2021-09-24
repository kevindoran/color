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


NUM_POSITIONS = 16  
IMG_SIZE = 224
ROW_LEN = NUM_POSITIONS**0.5
RADIUS = int(IMG_SIZE / (2 * ROW_LEN)) # 28
assert ROW_LEN == 4
assert RADIUS == 28
COLOR_ID_TO_LABEL = {0: 'orange', 1: 'brown', 2: 'both', 3: 'neither'}


# Load experiment data.
# Reference:
# https://github.com/wimglenn/resources-example/blob/master/myapp/example4.py
with importlib.resources.open_text("nncolor.pkdata", 'experiment_1_1_1.csv') as f:
    exp_1_1_data = pd.read_csv(f)


class ColorOption:
    def __init__(self, circle_rgb, bg_rgb):
        self.circle_rgb = circle_rgb
        self.bg_rgb = bg_rgb
    
    
def _orange_rgbs():
    return exp_1_1_data.loc[lambda r:r['ans'] == 0, :]


def _brown_rgbs():
    return exp_1_1_data.loc[lambda r:r['ans'] == 1, :]


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


def coords(position_idx : int) -> Tuple[int, int]:
    """Calculates the (y, x) coordinates of a position index.
    
    Origin is top left.
    """
    if not 0 <= position_idx < NUM_POSITIONS:
        raise Exception(f'Position index must be within 0 and {NUM_POSITIONS}. '
                        'Got: {position_idx}')
    row = position_idx // ROW_LEN
    col = position_idx % ROW_LEN
    y = int(RADIUS * (row * 2 + 1))
    x = int(RADIUS * (col * 2 + 1))
    return (y, x)
     
    
def to_cv2_coords(numpy_coords):
    """CV2 uses (x, y) order."""
    return (numpy_coords[1], numpy_coords[0])
    
    
def circle_img(position_idx, circle_color, bg_color):
    """Create a circle-background image."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32)
    img[:] = bg_color
    center = to_cv2_coords(coords(position_idx))
    img = cv2.circle(img, center, RADIUS, circle_color, thickness=-1, 
            lineType=cv2.LINE_AA)
    return img


def create_samples(num_samples):
    frames = []
    labels = []
    for n in range(num_samples):
        color, label = random_color()
        pos = random.randrange(0, NUM_POSITIONS)
        frame = circle_img(pos, color.circle_rgb, color.bg_rgb)
        frames.append(frame)
        labels.append(label)
    return frames, labels


class ColorDotDataset(torch.utils.data.Dataset):
    "Colored circles on colored background, a dataset."
    
    def __init__(self, labelled_colors):
        """Generate dataset from an array of (label, circle_rgb, bg_rgb) tuples.
        """
        self._labelled_colors = labelled_colors
        self.rng = np.random.default_rng(123)
        self.shuffled_idxs = self.rng.integers(low=0, 
                high=self.__len__(), size=self.__len__())
        
    def __len__(self):
        return len(self._labelled_colors) * NUM_POSITIONS
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.shuffled_idxs[idx]    
        color_idx = idx // NUM_POSITIONS
        pos = idx % NUM_POSITIONS
        label, circle_rgb, bg_rgb = self._labelled_colors[color_idx]
        ic((pos, label, circle_rgb, bg_rgb))
        return {'image': circle_img(pos, circle_rgb, bg_rgb), 'label': label}
    
    
def train_test_val_split(labelled_colors, split_ratio=[6, 1, 1]):
    divisions = np.sum(np.array(split_ratio))
    num_per_division = len(labelled_colors) // divisions
    remainder = len(labelled_colors) % divisions
    num_train = num_per_division * split_ratio[0] + remainder
    num_test = num_per_division * split_ratio[1]
    num_val = num_per_division * split_ratio[2]
    
    train_ds = ColorDotDataset(labelled_colors[0: num_train])
    test_ds = ColorDotDataset(labelled_colors[num_train: num_train + num_test])
    val_ds = ColorDotDataset(labelled_colors[num_train + num_test:])
    return (train_ds, test_ds, val_ds)


def filter_colors(table, include_colors):
    # TOD: how to properly filter and map pandas data?
    ans = []
    #table[0] = table[0].apply(lambda x : COLOR_ID_TO_LABEL[x])
    #filtered = table[table['ans'] in include_labels]
    for idx, row in table.iterrows():
        label = COLOR_ID_TO_LABEL[row['ans']]
        if not label in include_colors:
            continue
        circle_rgb = json.loads(row['circle_rgb'])
        bg_rgb = json.loads(row['bg_rgb'])
        ans.append((label, circle_rgb, bg_rgb))
    return ans


def load_datasets():
    colors = filter_colors(exp_1_1_data, 
                           include_colors={'orange', 'brown', 'neither'})
    datasets = train_test_val_split(colors)
    return datasets
    
