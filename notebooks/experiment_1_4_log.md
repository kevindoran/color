# Experiment 1.4
Carrying on from experiment 1.3. We will repeat the idea of 1.3, but chop off the output side of the network to expose the pre-pooled activations. This is a process of increasing the resolution that we are investigating.

This experiment shows that the Cx7x7 activations reaching the pool layer carry enough information for _poor but not terrible_ accuracy at a 7x7 resolution to detect the difference between orange and brown. The experiment setup might be contributing to poor accuracy, so it's not appropriate to conclude an accuracy upper bound.


```python
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
import nncolor as nc
import nncolor.data
import torchinfo
import torchvision as tv
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import torch.nn
import torch.optim
import time
import copy
import os
import torch.hub
from collections import namedtuple
import ipyplot
import matplotlib.pyplot as plt
import matplotlib as mpl
```


```python
presentation_mode = True
md_export_mode = True
if presentation_mode:
    import warnings
    warnings.filterwarnings('ignore')
    mpl.rcParams.update({'font.size': 20})
    mpl.rcParams.update({'axes.labelsize': 20})
    mpl.rcParams.update({'text.usetex': False})
```


```python
import IPython
def imshow(img):
    """Show image. 
    
    Image is a HWC numpy array with values in the range 0-1."""
    img = img*255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2 imencode takes images in HWC dimension order.
    _,ret = cv2.imencode('.jpg', img) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i) 
    
    
def imlist(images, labels=None, use_tabs=False):
    if md_export_mode:
        print("Skipping ipyplot image print for markdown export. The output"
              " produces HTML that either Jupyter Lab fails to export correctly,"
              " or Hugo fails to render correctly. Skipping for now.")
        return
    if use_tabs:
        ipyplot.plot_class_tabs(images, labels, max_imgs_per_tab=300)
    else:
        ipyplot.plot_images(images, labels)
```

## 1. Notebook constants
Variables used as constants throughout the notebook.


```python
# Choose CPU or GPU.
device = torch.device('cuda:0')
#device = "cpu"

# Choose small or large (standard) model variant
#model_name = "resnet18"
model_name = 'resnet50'
def model_fctn():
    if model_name == 'resnet18':
        return tv.models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        return tv.models.resnet50(pretrained=True)
resnet_model = model_fctn()

GRID_SHAPE = (7, 7)
NUM_CELLS = np.prod(GRID_SHAPE)
IMG_SHAPE = (224, 224, 3)
cell_shape = nc.data.cell_shape(GRID_SHAPE, IMG_SHAPE)
assert np.array_equal(cell_shape, (32, 32))
# Choosing a diameter less than cell width/height.
# Let's go with 20 (so radius is 10)
RADIUS = 10
BATCH_SIZE = 4
NUM_FC_CHANNELS = 512 if model_name == 'resnet18' else 2048
```

## 2. Model (resnet)
First, let's double check the model summary.

![image.png](attachment:60e76257-a823-40ca-a697-1138abcb5031.png)

Another model visualization can be seen  at: 
http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006


```python
torchinfo.summary(resnet_model, (1, 3, 224, 224))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    ResNet                                   --                        --
    ├─Conv2d: 1-1                            [1, 64, 112, 112]         9,408
    ├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
    ├─ReLU: 1-3                              [1, 64, 112, 112]         --
    ├─MaxPool2d: 1-4                         [1, 64, 56, 56]           --
    ├─Sequential: 1-5                        [1, 256, 56, 56]          --
    │    └─Bottleneck: 2-1                   [1, 256, 56, 56]          --
    │    │    └─Conv2d: 3-1                  [1, 64, 56, 56]           4,096
    │    │    └─BatchNorm2d: 3-2             [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-3                    [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-4                  [1, 64, 56, 56]           36,864
    │    │    └─BatchNorm2d: 3-5             [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-6                    [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-7                  [1, 256, 56, 56]          16,384
    │    │    └─BatchNorm2d: 3-8             [1, 256, 56, 56]          512
    │    │    └─Sequential: 3-9              [1, 256, 56, 56]          16,896
    │    │    └─ReLU: 3-10                   [1, 256, 56, 56]          --
    │    └─Bottleneck: 2-2                   [1, 256, 56, 56]          --
    │    │    └─Conv2d: 3-11                 [1, 64, 56, 56]           16,384
    │    │    └─BatchNorm2d: 3-12            [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-13                   [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-14                 [1, 64, 56, 56]           36,864
    │    │    └─BatchNorm2d: 3-15            [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-16                   [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-17                 [1, 256, 56, 56]          16,384
    │    │    └─BatchNorm2d: 3-18            [1, 256, 56, 56]          512
    │    │    └─ReLU: 3-19                   [1, 256, 56, 56]          --
    │    └─Bottleneck: 2-3                   [1, 256, 56, 56]          --
    │    │    └─Conv2d: 3-20                 [1, 64, 56, 56]           16,384
    │    │    └─BatchNorm2d: 3-21            [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-22                   [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-23                 [1, 64, 56, 56]           36,864
    │    │    └─BatchNorm2d: 3-24            [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-25                   [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-26                 [1, 256, 56, 56]          16,384
    │    │    └─BatchNorm2d: 3-27            [1, 256, 56, 56]          512
    │    │    └─ReLU: 3-28                   [1, 256, 56, 56]          --
    ├─Sequential: 1-6                        [1, 512, 28, 28]          --
    │    └─Bottleneck: 2-4                   [1, 512, 28, 28]          --
    │    │    └─Conv2d: 3-29                 [1, 128, 56, 56]          32,768
    │    │    └─BatchNorm2d: 3-30            [1, 128, 56, 56]          256
    │    │    └─ReLU: 3-31                   [1, 128, 56, 56]          --
    │    │    └─Conv2d: 3-32                 [1, 128, 28, 28]          147,456
    │    │    └─BatchNorm2d: 3-33            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-34                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-35                 [1, 512, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-36            [1, 512, 28, 28]          1,024
    │    │    └─Sequential: 3-37             [1, 512, 28, 28]          132,096
    │    │    └─ReLU: 3-38                   [1, 512, 28, 28]          --
    │    └─Bottleneck: 2-5                   [1, 512, 28, 28]          --
    │    │    └─Conv2d: 3-39                 [1, 128, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-40            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-41                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-42                 [1, 128, 28, 28]          147,456
    │    │    └─BatchNorm2d: 3-43            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-44                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-45                 [1, 512, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-46            [1, 512, 28, 28]          1,024
    │    │    └─ReLU: 3-47                   [1, 512, 28, 28]          --
    │    └─Bottleneck: 2-6                   [1, 512, 28, 28]          --
    │    │    └─Conv2d: 3-48                 [1, 128, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-49            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-50                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-51                 [1, 128, 28, 28]          147,456
    │    │    └─BatchNorm2d: 3-52            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-53                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-54                 [1, 512, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-55            [1, 512, 28, 28]          1,024
    │    │    └─ReLU: 3-56                   [1, 512, 28, 28]          --
    │    └─Bottleneck: 2-7                   [1, 512, 28, 28]          --
    │    │    └─Conv2d: 3-57                 [1, 128, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-58            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-59                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-60                 [1, 128, 28, 28]          147,456
    │    │    └─BatchNorm2d: 3-61            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-62                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-63                 [1, 512, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-64            [1, 512, 28, 28]          1,024
    │    │    └─ReLU: 3-65                   [1, 512, 28, 28]          --
    ├─Sequential: 1-7                        [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-8                   [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-66                 [1, 256, 28, 28]          131,072
    │    │    └─BatchNorm2d: 3-67            [1, 256, 28, 28]          512
    │    │    └─ReLU: 3-68                   [1, 256, 28, 28]          --
    │    │    └─Conv2d: 3-69                 [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-70            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-71                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-72                 [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-73            [1, 1024, 14, 14]         2,048
    │    │    └─Sequential: 3-74             [1, 1024, 14, 14]         526,336
    │    │    └─ReLU: 3-75                   [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-9                   [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-76                 [1, 256, 14, 14]          262,144
    │    │    └─BatchNorm2d: 3-77            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-78                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-79                 [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-80            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-81                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-82                 [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-83            [1, 1024, 14, 14]         2,048
    │    │    └─ReLU: 3-84                   [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-10                  [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-85                 [1, 256, 14, 14]          262,144
    │    │    └─BatchNorm2d: 3-86            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-87                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-88                 [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-89            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-90                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-91                 [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-92            [1, 1024, 14, 14]         2,048
    │    │    └─ReLU: 3-93                   [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-11                  [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-94                 [1, 256, 14, 14]          262,144
    │    │    └─BatchNorm2d: 3-95            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-96                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-97                 [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-98            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-99                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-100                [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-101           [1, 1024, 14, 14]         2,048
    │    │    └─ReLU: 3-102                  [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-12                  [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-103                [1, 256, 14, 14]          262,144
    │    │    └─BatchNorm2d: 3-104           [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-105                  [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-106                [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-107           [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-108                  [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-109                [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-110           [1, 1024, 14, 14]         2,048
    │    │    └─ReLU: 3-111                  [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-13                  [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-112                [1, 256, 14, 14]          262,144
    │    │    └─BatchNorm2d: 3-113           [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-114                  [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-115                [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-116           [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-117                  [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-118                [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-119           [1, 1024, 14, 14]         2,048
    │    │    └─ReLU: 3-120                  [1, 1024, 14, 14]         --
    ├─Sequential: 1-8                        [1, 2048, 7, 7]           --
    │    └─Bottleneck: 2-14                  [1, 2048, 7, 7]           --
    │    │    └─Conv2d: 3-121                [1, 512, 14, 14]          524,288
    │    │    └─BatchNorm2d: 3-122           [1, 512, 14, 14]          1,024
    │    │    └─ReLU: 3-123                  [1, 512, 14, 14]          --
    │    │    └─Conv2d: 3-124                [1, 512, 7, 7]            2,359,296
    │    │    └─BatchNorm2d: 3-125           [1, 512, 7, 7]            1,024
    │    │    └─ReLU: 3-126                  [1, 512, 7, 7]            --
    │    │    └─Conv2d: 3-127                [1, 2048, 7, 7]           1,048,576
    │    │    └─BatchNorm2d: 3-128           [1, 2048, 7, 7]           4,096
    │    │    └─Sequential: 3-129            [1, 2048, 7, 7]           2,101,248
    │    │    └─ReLU: 3-130                  [1, 2048, 7, 7]           --
    │    └─Bottleneck: 2-15                  [1, 2048, 7, 7]           --
    │    │    └─Conv2d: 3-131                [1, 512, 7, 7]            1,048,576
    │    │    └─BatchNorm2d: 3-132           [1, 512, 7, 7]            1,024
    │    │    └─ReLU: 3-133                  [1, 512, 7, 7]            --
    │    │    └─Conv2d: 3-134                [1, 512, 7, 7]            2,359,296
    │    │    └─BatchNorm2d: 3-135           [1, 512, 7, 7]            1,024
    │    │    └─ReLU: 3-136                  [1, 512, 7, 7]            --
    │    │    └─Conv2d: 3-137                [1, 2048, 7, 7]           1,048,576
    │    │    └─BatchNorm2d: 3-138           [1, 2048, 7, 7]           4,096
    │    │    └─ReLU: 3-139                  [1, 2048, 7, 7]           --
    │    └─Bottleneck: 2-16                  [1, 2048, 7, 7]           --
    │    │    └─Conv2d: 3-140                [1, 512, 7, 7]            1,048,576
    │    │    └─BatchNorm2d: 3-141           [1, 512, 7, 7]            1,024
    │    │    └─ReLU: 3-142                  [1, 512, 7, 7]            --
    │    │    └─Conv2d: 3-143                [1, 512, 7, 7]            2,359,296
    │    │    └─BatchNorm2d: 3-144           [1, 512, 7, 7]            1,024
    │    │    └─ReLU: 3-145                  [1, 512, 7, 7]            --
    │    │    └─Conv2d: 3-146                [1, 2048, 7, 7]           1,048,576
    │    │    └─BatchNorm2d: 3-147           [1, 2048, 7, 7]           4,096
    │    │    └─ReLU: 3-148                  [1, 2048, 7, 7]           --
    ├─AdaptiveAvgPool2d: 1-9                 [1, 2048, 1, 1]           --
    ├─Linear: 1-10                           [1, 1000]                 2,049,000
    ==========================================================================================
    Total params: 25,557,032
    Trainable params: 25,557,032
    Non-trainable params: 0
    Total mult-adds (G): 4.09
    ==========================================================================================
    Input size (MB): 0.60
    Forward/backward pass size (MB): 177.83
    Params size (MB): 102.23
    Estimated Total Size (MB): 280.66
    ==========================================================================================



## 3. Dataset
The dataset generation code has been moved out into a Python package, `nncolor`. Below, we 
take a moment to investigate the data.

First, recall the 4 color classes we are dealing with, and the numbers they are mapped to.


```python
nc.data.color_legend();
```


    
![png](experiment_1_4_log_files/experiment_1_4_log_10_0.png)
    


Below is a movie clip cycling through some of the image-label pairs that will be generated.


```python
def demo_data():
    FPS = 2
    frames, labels = nc.data.create_samples(30, radius=RADIUS, grid_shape=GRID_SHAPE, 
                                            img_shape=IMG_SHAPE)
    frames = [f*255 for f in frames]
    x_clip = mpe.ImageSequenceClip(frames, fps=2)
    y_clip = mpe.TextClip('WB-0', font='DejaVu-Sans')

    class FrameText(mpe.VideoClip):
        def __init__(self, text, fps):
            def make_frame(f):
               return mpe.TextClip(text[int(f)], font='DejaVu-Sans', color='white').get_frame(f)
            self.duration = 1.0 * len(text) / fps
            mpe.VideoClip.__init__(self, make_frame=make_frame, duration=self.duration)

    y_clip =   FrameText(labels, FPS)
    label_clip = mpe.CompositeVideoClip([mpe.ImageClip(np.zeros(nc.data.DEFAULT_IMG_SHAPE), duration=5), y_clip])
    comp_clip = mpe.clips_array([[y_clip],[x_clip]])
    return comp_clip
clip = demo_data() 
clip.ipython_display(rd_kwargs={'logger':None})
```




<div align=middle><video src='data:video/mp4;base64,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' controls>Sorry, seems like your browser doesn't support HTML5 audio/video</video></div>




```python
color_counts = [ (nc.data.COLOR_ID_TO_LABEL[idx], nc.data.exp_1_1_data.loc[lambda r:r['ans'] == idx, :].shape[0]) for idx in range(nc.data.NUM_CLASSES)]
ic(color_counts)
```

    ic| color_counts: [('orange', 39), ('brown', 29), ('both', 3), ('neither', 148)]





    [('orange', 39), ('brown', 29), ('both', 3), ('neither', 148)]




```python
def test_dataset():
    train, test, val = nc.data.train_test_val_split(nc.data.exp_1_1_data_filtered, 
                                      RADIUS, GRID_SHAPE, IMG_SHAPE)
    sample = val[7]
    imshow(sample['image'])
    ic(sample['label'])
    ic(sample['label_grid'])
    ic(sample['position'])
test_dataset()
```


    
![png](experiment_1_4_log_files/experiment_1_4_log_14_0.png)
    


    ic| sample['label']: 0
    ic| sample['label_grid']: array([[3, 3, 3, 3, 3, 3, 3],
                                     [3, 3, 3, 3, 3, 3, 0],
                                     [3, 3, 3, 3, 3, 3, 3],
                                     [3, 3, 3, 3, 3, 3, 3],
                                     [3, 3, 3, 3, 3, 3, 3],
                                     [3, 3, 3, 3, 3, 3, 3],
                                     [3, 3, 3, 3, 3, 3, 3]])
    ic| sample['position']: 13


## 4. Customize the model
We want to have a model with the fully connected layer removed, and for that, it seems easiest to subclass and override.


```python
class PoolHeadResNet(tv.models.ResNet):
    def __init__(self, train_all_params=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = nc.data.NUM_CLASSES
        # First, disable all other parameters.
        if not train_all_params:
            for param in self.parameters():
                param.requires_grad = False
        # A 1x1 convolution, representing effectively a fully connected, sub-batched at 7x7.
        self.fc2 = torch.nn.Conv2d(in_channels=NUM_FC_CHANNELS, out_channels=num_classes, kernel_size=1,stride=1)
        
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Stop above.
        #x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        x = self.fc2(x)
        return x
    
train_all_params = True
edited_resnet_model = None
if model_name == 'resnet18':
    edited_resnet_model = PoolHeadResNet(train_all_params, tv.models.resnet.BasicBlock, [2, 2, 2, 2])
elif model_name == 'resnet50':
    edited_resnet_model = PoolHeadResNet(train_all_params, tv.models.resnet.Bottleneck, [3, 4, 6, 3])
else:
    raise Exception('Not supported')
state = torch.hub.load_state_dict_from_url(tv.models.resnet.model_urls[model_name])
edited_resnet_model.load_state_dict(state, strict=False);
```

### 4.1 Modified model summary
The modified model can be seen to have a final layer with output shape [Batch, 4, 7, 7] representing 7x7 individual 4-class classification outputs.


```python
torchinfo.summary(edited_resnet_model, (1, 3, 224, 224))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    PoolHeadResNet                           --                        --
    ├─Conv2d: 1-1                            [1, 64, 112, 112]         9,408
    ├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
    ├─ReLU: 1-3                              [1, 64, 112, 112]         --
    ├─MaxPool2d: 1-4                         [1, 64, 56, 56]           --
    ├─Sequential: 1-5                        [1, 256, 56, 56]          --
    │    └─Bottleneck: 2-1                   [1, 256, 56, 56]          --
    │    │    └─Conv2d: 3-1                  [1, 64, 56, 56]           4,096
    │    │    └─BatchNorm2d: 3-2             [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-3                    [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-4                  [1, 64, 56, 56]           36,864
    │    │    └─BatchNorm2d: 3-5             [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-6                    [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-7                  [1, 256, 56, 56]          16,384
    │    │    └─BatchNorm2d: 3-8             [1, 256, 56, 56]          512
    │    │    └─Sequential: 3-9              [1, 256, 56, 56]          16,896
    │    │    └─ReLU: 3-10                   [1, 256, 56, 56]          --
    │    └─Bottleneck: 2-2                   [1, 256, 56, 56]          --
    │    │    └─Conv2d: 3-11                 [1, 64, 56, 56]           16,384
    │    │    └─BatchNorm2d: 3-12            [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-13                   [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-14                 [1, 64, 56, 56]           36,864
    │    │    └─BatchNorm2d: 3-15            [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-16                   [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-17                 [1, 256, 56, 56]          16,384
    │    │    └─BatchNorm2d: 3-18            [1, 256, 56, 56]          512
    │    │    └─ReLU: 3-19                   [1, 256, 56, 56]          --
    │    └─Bottleneck: 2-3                   [1, 256, 56, 56]          --
    │    │    └─Conv2d: 3-20                 [1, 64, 56, 56]           16,384
    │    │    └─BatchNorm2d: 3-21            [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-22                   [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-23                 [1, 64, 56, 56]           36,864
    │    │    └─BatchNorm2d: 3-24            [1, 64, 56, 56]           128
    │    │    └─ReLU: 3-25                   [1, 64, 56, 56]           --
    │    │    └─Conv2d: 3-26                 [1, 256, 56, 56]          16,384
    │    │    └─BatchNorm2d: 3-27            [1, 256, 56, 56]          512
    │    │    └─ReLU: 3-28                   [1, 256, 56, 56]          --
    ├─Sequential: 1-6                        [1, 512, 28, 28]          --
    │    └─Bottleneck: 2-4                   [1, 512, 28, 28]          --
    │    │    └─Conv2d: 3-29                 [1, 128, 56, 56]          32,768
    │    │    └─BatchNorm2d: 3-30            [1, 128, 56, 56]          256
    │    │    └─ReLU: 3-31                   [1, 128, 56, 56]          --
    │    │    └─Conv2d: 3-32                 [1, 128, 28, 28]          147,456
    │    │    └─BatchNorm2d: 3-33            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-34                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-35                 [1, 512, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-36            [1, 512, 28, 28]          1,024
    │    │    └─Sequential: 3-37             [1, 512, 28, 28]          132,096
    │    │    └─ReLU: 3-38                   [1, 512, 28, 28]          --
    │    └─Bottleneck: 2-5                   [1, 512, 28, 28]          --
    │    │    └─Conv2d: 3-39                 [1, 128, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-40            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-41                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-42                 [1, 128, 28, 28]          147,456
    │    │    └─BatchNorm2d: 3-43            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-44                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-45                 [1, 512, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-46            [1, 512, 28, 28]          1,024
    │    │    └─ReLU: 3-47                   [1, 512, 28, 28]          --
    │    └─Bottleneck: 2-6                   [1, 512, 28, 28]          --
    │    │    └─Conv2d: 3-48                 [1, 128, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-49            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-50                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-51                 [1, 128, 28, 28]          147,456
    │    │    └─BatchNorm2d: 3-52            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-53                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-54                 [1, 512, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-55            [1, 512, 28, 28]          1,024
    │    │    └─ReLU: 3-56                   [1, 512, 28, 28]          --
    │    └─Bottleneck: 2-7                   [1, 512, 28, 28]          --
    │    │    └─Conv2d: 3-57                 [1, 128, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-58            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-59                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-60                 [1, 128, 28, 28]          147,456
    │    │    └─BatchNorm2d: 3-61            [1, 128, 28, 28]          256
    │    │    └─ReLU: 3-62                   [1, 128, 28, 28]          --
    │    │    └─Conv2d: 3-63                 [1, 512, 28, 28]          65,536
    │    │    └─BatchNorm2d: 3-64            [1, 512, 28, 28]          1,024
    │    │    └─ReLU: 3-65                   [1, 512, 28, 28]          --
    ├─Sequential: 1-7                        [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-8                   [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-66                 [1, 256, 28, 28]          131,072
    │    │    └─BatchNorm2d: 3-67            [1, 256, 28, 28]          512
    │    │    └─ReLU: 3-68                   [1, 256, 28, 28]          --
    │    │    └─Conv2d: 3-69                 [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-70            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-71                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-72                 [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-73            [1, 1024, 14, 14]         2,048
    │    │    └─Sequential: 3-74             [1, 1024, 14, 14]         526,336
    │    │    └─ReLU: 3-75                   [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-9                   [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-76                 [1, 256, 14, 14]          262,144
    │    │    └─BatchNorm2d: 3-77            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-78                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-79                 [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-80            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-81                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-82                 [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-83            [1, 1024, 14, 14]         2,048
    │    │    └─ReLU: 3-84                   [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-10                  [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-85                 [1, 256, 14, 14]          262,144
    │    │    └─BatchNorm2d: 3-86            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-87                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-88                 [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-89            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-90                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-91                 [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-92            [1, 1024, 14, 14]         2,048
    │    │    └─ReLU: 3-93                   [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-11                  [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-94                 [1, 256, 14, 14]          262,144
    │    │    └─BatchNorm2d: 3-95            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-96                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-97                 [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-98            [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-99                   [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-100                [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-101           [1, 1024, 14, 14]         2,048
    │    │    └─ReLU: 3-102                  [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-12                  [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-103                [1, 256, 14, 14]          262,144
    │    │    └─BatchNorm2d: 3-104           [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-105                  [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-106                [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-107           [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-108                  [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-109                [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-110           [1, 1024, 14, 14]         2,048
    │    │    └─ReLU: 3-111                  [1, 1024, 14, 14]         --
    │    └─Bottleneck: 2-13                  [1, 1024, 14, 14]         --
    │    │    └─Conv2d: 3-112                [1, 256, 14, 14]          262,144
    │    │    └─BatchNorm2d: 3-113           [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-114                  [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-115                [1, 256, 14, 14]          589,824
    │    │    └─BatchNorm2d: 3-116           [1, 256, 14, 14]          512
    │    │    └─ReLU: 3-117                  [1, 256, 14, 14]          --
    │    │    └─Conv2d: 3-118                [1, 1024, 14, 14]         262,144
    │    │    └─BatchNorm2d: 3-119           [1, 1024, 14, 14]         2,048
    │    │    └─ReLU: 3-120                  [1, 1024, 14, 14]         --
    ├─Sequential: 1-8                        [1, 2048, 7, 7]           --
    │    └─Bottleneck: 2-14                  [1, 2048, 7, 7]           --
    │    │    └─Conv2d: 3-121                [1, 512, 14, 14]          524,288
    │    │    └─BatchNorm2d: 3-122           [1, 512, 14, 14]          1,024
    │    │    └─ReLU: 3-123                  [1, 512, 14, 14]          --
    │    │    └─Conv2d: 3-124                [1, 512, 7, 7]            2,359,296
    │    │    └─BatchNorm2d: 3-125           [1, 512, 7, 7]            1,024
    │    │    └─ReLU: 3-126                  [1, 512, 7, 7]            --
    │    │    └─Conv2d: 3-127                [1, 2048, 7, 7]           1,048,576
    │    │    └─BatchNorm2d: 3-128           [1, 2048, 7, 7]           4,096
    │    │    └─Sequential: 3-129            [1, 2048, 7, 7]           2,101,248
    │    │    └─ReLU: 3-130                  [1, 2048, 7, 7]           --
    │    └─Bottleneck: 2-15                  [1, 2048, 7, 7]           --
    │    │    └─Conv2d: 3-131                [1, 512, 7, 7]            1,048,576
    │    │    └─BatchNorm2d: 3-132           [1, 512, 7, 7]            1,024
    │    │    └─ReLU: 3-133                  [1, 512, 7, 7]            --
    │    │    └─Conv2d: 3-134                [1, 512, 7, 7]            2,359,296
    │    │    └─BatchNorm2d: 3-135           [1, 512, 7, 7]            1,024
    │    │    └─ReLU: 3-136                  [1, 512, 7, 7]            --
    │    │    └─Conv2d: 3-137                [1, 2048, 7, 7]           1,048,576
    │    │    └─BatchNorm2d: 3-138           [1, 2048, 7, 7]           4,096
    │    │    └─ReLU: 3-139                  [1, 2048, 7, 7]           --
    │    └─Bottleneck: 2-16                  [1, 2048, 7, 7]           --
    │    │    └─Conv2d: 3-140                [1, 512, 7, 7]            1,048,576
    │    │    └─BatchNorm2d: 3-141           [1, 512, 7, 7]            1,024
    │    │    └─ReLU: 3-142                  [1, 512, 7, 7]            --
    │    │    └─Conv2d: 3-143                [1, 512, 7, 7]            2,359,296
    │    │    └─BatchNorm2d: 3-144           [1, 512, 7, 7]            1,024
    │    │    └─ReLU: 3-145                  [1, 512, 7, 7]            --
    │    │    └─Conv2d: 3-146                [1, 2048, 7, 7]           1,048,576
    │    │    └─BatchNorm2d: 3-147           [1, 2048, 7, 7]           4,096
    │    │    └─ReLU: 3-148                  [1, 2048, 7, 7]           --
    ├─Conv2d: 1-9                            [1, 4, 7, 7]              8,196
    ==========================================================================================
    Total params: 23,516,228
    Trainable params: 23,516,228
    Non-trainable params: 0
    Total mult-adds (G): 4.09
    ==========================================================================================
    Input size (MB): 0.60
    Forward/backward pass size (MB): 177.83
    Params size (MB): 94.06
    Estimated Total Size (MB): 272.49
    ==========================================================================================



### 4.2 Check that initialization of the new layer
Double check that our new layer has been initialized


```python
edited_resnet_model.fc2.weight[0][0:10]
```




    tensor([[[ 0.0168]],
    
            [[ 0.0185]],
    
            [[-0.0094]],
    
            [[ 0.0166]],
    
            [[-0.0213]],
    
            [[-0.0178]],
    
            [[-0.0010]],
    
            [[-0.0024]],
    
            [[ 0.0076]],
    
            [[ 0.0081]]], device='cuda:0', grad_fn=<SliceBackward>)



All looks good! Now design the loss function.

## 5. Loss function
The best loss function to use is not obvious. On one extreme, we can weight all 49 outputs equally as 49 equal predictions to which we apply cross entropy loss. However, this will heavily weight the model caring only about the background, which is always a "3" for neither orange or brown. Alternatively, we could weight only the activation mapped to the dot position. A loss that is a balance of these extremes is also possible.

I try the two extreme losses. I also try two weighted losses that sit in the middle.

### 5.1 Equal weighting
We can extend the MLE/cross-entropy loss used in the previous experiment. If we treat each of the 49 (7x7) outputs as a classification output, then each forward pass now involves 49 different results that form a sort of "sub-batch". We need to take care that our use of the standard loss function is working correctly with this increased output dimension. Below is a brief demo of how the API can be used to achieve the correct result.
The `CrossEntropyLoss` function luckily supports multidimensional input/outputs in the way I need:


```python
# Tesing out a loss function
# 4x2x2 (smaller test version of our 7x7x4).
fc_in = torch.tensor([# batch
    [ # 4x2x2
        [[-0.3, 0.0], [0.3, 3.2] ], # 2x2
        [[ 4.0, 0.1], [-4., 1.1] ], # 2x2
        [[ 7.0, 0.4], [2.0, -.4] ], # 2x2
        [[ 2.1, 4.1], [2.2, 4.1] ], # 2x2
    ]
], dtype=torch.float)
target = torch.tensor([ # batch
    [[0, 1], [1, 1]] # 2x2
], dtype=torch.long)
# We can view all 4 classification losses:
loss = torch.nn.CrossEntropyLoss(reduction='none') 
output = loss(fc_in, target)
ic(output)
# Or we can reduce them to a single number.
loss = torch.nn.CrossEntropyLoss(reduction='mean') 
output = loss(fc_in, target)
ic(output)
```

    ic| output: tensor([[[7.3563, 4.0579],
                         [6.8782, 3.3835]]])
    ic| output: tensor(5.4190)





    tensor(5.4190)



As a loss function, the 49 sub-batch MLE loss is implemented like below. The extra `pos` and `mask` arguments as the later loss functions use these extra arguments.


```python
def mle(input_, target, pos, mask):
    """Loss function where each output unit is a classification."""
    return torch.nn.functional.cross_entropy(input_, target, reduction='mean')
```

### 5.2 MLE loss (background-foreground balance weighting)
This weighting gives the 48 units associated with the background a weighting of 1 and gives the
the single unit associated with the dot a weighting of 48.

When an equal weighting is used, the model can get an accuracy of about 48/49 if it always classifies all of the 49 outputs as "neither". This loss tries to correct for this issue. 


```python
def mle_balanced(input_, target, pos, mask):
    """48 weight for the dot-cell, 1 weight for each 48 background 48 cells.""" 
    loss = torch.nn.functional.cross_entropy(input_, target, reduction='none')
    loss_weight = torch.mul(mask, NUM_CELLS - 2) + torch.ones(mask.shape).to(device)
    loss = torch.mul(loss, loss_weight)
    loss = torch.mean(loss)
    return loss
```

### 5.3 Dot only
This loss is a MLE loss that only considers the output unit that is associated with the
dot position. All other outputs are ignored. 

An obvious issue with this loss is that there is no incentive for the model to correctly
classify the background color. This is a loss that should give the model great leniency 
towards maximizing accuracy of the dot color classification at the expense of the background
color classification.


```python
def mle_dot(input_, target, pos, mask):
    """MLE loss considering only the output unit associated with the dot position."""
    loss = torch.nn.functional.cross_entropy(input_, target, reduction='none')
    dot_loss = torch.mul(mask, loss) 
    dot_loss = torch.mean(loss)
    return dot_loss
```

### 5.3 MLE loss (radial weighted) 
MLE loss for each of the 49 outputs, with the loss weighted based on the distance to the dot. This is like the first loss above, except we have introduced a more complex weighting function.



```python
# Cache the computation of radial loss weighting. This is used
# in the below loss function.
radial_loss = nc.data.radial_weight_for_every_pos(GRID_SHAPE, max_dist=5, 
                                                rate_fctn=nc.data.ease_in_out_sine)
radial_loss[np.array([0,4])].shape
radial_loss = torch.from_numpy(radial_loss).to(device)

def mle_radial(input_, target, pos, mask):
    loss = torch.nn.functional.cross_entropy(input_, target, reduction='none')
    loss_weight = radial_loss[pos]
    dot_loss = torch.mul(loss_weight, loss) 
    dot_loss = torch.mean(loss)
    return dot_loss
```

## 6. Training
Training loop.


```python
# Copied from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# Data augmentation and normalization
normalize_transform =  tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# tv.transforms.ToTensor will swap the dimensions from HWC to CHW
to_CHW_tensor_transform = tv.transforms.ToTensor()
data_transform = tv.transforms.Compose([to_CHW_tensor_transform, normalize_transform])
train_ds, test_ds, val_ds = nc.data.train_test_val_split(nc.data.exp_1_1_data_filtered,
                                                        dot_radius=RADIUS,
                                                        grid_shape=GRID_SHAPE)
val_ds.transform = data_transform
train_ds.transform = data_transform
# Don't transform the test dataset. We need to visualize the images without them being normalized.
test_ds.transform = to_CHW_tensor_transform
ds = {'train': train_ds, 'val': val_ds, 'test': test_ds}
dataloaders = {x: torch.utils.data.DataLoader(ds[x], batch_size=BATCH_SIZE, num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(ds[x]) for x in ['train', 'val', 'test']}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        # For this experimentation, the validation isn't that useful, 
        # as the measure of accuracy we are interested is quite complex.
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_dot_corrects = 0

            # Iterate over data.
            #for inputs, labels in dataloaders[phase]:
            for batch in dataloaders[phase]:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                label_grids = batch['label_grid'].to(device)
                mask_grids = batch['mask_grid'].to(device)
                pos = batch['position'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, label_grids, pos, mask_grids)
                    # backward + optimize only if in training phase
                    #l2_reg = 0
                    #for param in model.fc2.parameters():
                    #    l2_reg += torch.norm(param)
                    #loss += 0.01 * l2_reg
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                pred_correct = preds == label_grids.data
                pred_dot_correct = torch.mul(pred_correct, mask_grids)
                running_corrects += torch.sum(pred_correct)
                running_dot_corrects += torch.sum(pred_dot_correct)
            if phase == 'train':
                scheduler.step()

            denom = dataset_sizes[phase] * NUM_CELLS
            epoch_loss = running_loss / denom
            epoch_acc = running_corrects.double() / denom
            epoch_dot_acc = running_dot_corrects / dataset_sizes[phase]

            print('{} Loss: {:.5f} Acc: {:.5f} Dot Acc: {:.5f}'.format(
                phase, epoch_loss, epoch_acc, epoch_dot_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # NO! should be model with least loss
    #model.load_state_dict(best_model_wts)
    return model
```

## 7. Investigate model
To explore the nature of the predictions, I want an array with dimensions (label, prediction) and values are tallies. Accuracy for each label is then the diagonal divided by the sum along the prediction dimension. Overall accuracy is the sum along the diagonal divided by the complete sum. 

## 7.1 Model accuracy 
Below is some code that was used to test out how to implement this. 


```python
def _test_code():
    acc = torch.zeros((4,4), dtype=torch.long)
    res, label = [torch.rand((2, 4, 7, 7)), 
                  torch.randint(0, 4, (2, 7, 7))]
    norml = torch.sum(res, dim=1)
    norml = norml.unsqueeze(1)
    res = res / norml
    _, p = torch.max(res, 1)
    label = torch.flatten(label)
    p = torch.flatten(p)
    # put_ requires linear indexes, which is a bit strange.
    # https://pytorch.org/docs/master/generated/torch.Tensor.put_.html#torch.Tensor.put_
    linear_indexes = label*4 + p 
    ones = torch.LongTensor([1]).expand_as(linear_indexes)
    return acc.put_(linear_indexes, ones, accumulate=True)
_test_code()
```




    tensor([[ 6,  4,  9,  3],
            [ 5,  3,  2,  8],
            [ 4, 10,  6,  9],
            [ 6, 11,  9,  3]])



The actual accuracy investigation code is implemented below.


```python
def test_model(model):
    model.eval()
    accuracy = None
    dot_accuracy = None
    dot_tally_grid = torch.zeros(GRID_SHAPE).to(device)
    accuracy_details = torch.zeros((nc.data.NUM_CLASSES, nc.data.NUM_CLASSES)).to(device)
    dot_accuracy_details = torch.zeros((nc.data.NUM_CLASSES, nc.data.NUM_CLASSES)).to(device)
    ones = torch.ones(BATCH_SIZE*NUM_CELLS, dtype=torch.float).to(device)
    dot_ones = torch.ones(BATCH_SIZE, dtype=torch.float).to(device)
    for batch in dataloaders['test']:
        images = batch['image']
        inputs = normalize_transform(images).to(device)
        labels = batch['label'].to(device)
        pos = batch['position'].to(device)
        mask_grids = batch['mask_grid'].to(device)
        label_grids = batch['label_grid'].to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        # Grid tallying correct dot predictions
        dot_tally_grid += torch.sum((preds == label_grids)*mask_grids, dim=0)
        
        # Accuracy breakdown for all units. 
        labelf = torch.flatten(label_grids)
        #labelf = torch.full((BATCH_SIZE, *GRID_SHAPE, nc.data.LABEL_TO_COLOR_ID['neither'])
        predf = torch.flatten(preds)
        linear_indexes = labelf * nc.data.NUM_CLASSES + predf
        accuracy_details.put_(linear_indexes, ones, accumulate=True)
        
        # Dot accuracy breakdown.
        dot_predf = torch.gather(torch.flatten(preds, 1, -1), 1,torch.unsqueeze(pos, -1))
        dot_predf = torch.squeeze(dot_predf)
        linear_indexes = labels * nc.data.NUM_CLASSES + dot_predf
        dot_accuracy_details.put_(linear_indexes, dot_ones, accumulate=True)
        
    tally = torch.sum(accuracy_details, dim=1)
    tally = tally.unsqueeze(1)
    accuracy_details_norm = accuracy_details / tally
    
    tally = torch.sum(dot_accuracy_details, dim=1)
    tally = tally.unsqueeze(1)
    dot_accuracy_details_norm = dot_accuracy_details / tally
        
    def print_acc_array(arr: np.ndarray, title=''):
        labels = nc.data.COLOR_ID_TO_LABEL.keys()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        cax = ax.imshow(arr, cmap='viridis')
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        ax.set_ylabel('actual color')
        ax.set_xlabel('predicted color')
        plt.show()
    print_acc_array(accuracy_details_norm.cpu().numpy(), "Test set confustion matrix")
    print_acc_array(dot_accuracy_details_norm.cpu().numpy(), "Test set confustion matrix\n(dot positions only)")
    print("Test set confusion matrix data:")
    print(accuracy_details_norm)
    print("Test set confusion matrix data (dot positions only):")
    print(dot_accuracy_details_norm)
    print("Test set unnormalized confusion matrix data:")
    print(accuracy_details)
    print("Test set unnormalized confusion matrix data (dot positions only):")
    print(dot_accuracy_details)
    print("Grid tallying correct dot predictions")
    print(dot_tally_grid)
```

## 7.2 Debug specific samples
The following methods are used to check how a model behaves for a specific image. 


```python
def debug_img(model, img):
        input_ = torch.unsqueeze(normalize_transform(img), 0).to(device)
        output = model(input_)
        ans = torch.argmax(output, 1)[0]
        ans = ans.cpu().detach().numpy()
        img = img.numpy().swapaxes(0, -1)
        # cv2 draw has some issues if numpy arrays are not continuous,
        # and to fix spurious issues, copying seems to work.
        im_print = img.copy()
        #im_print *= 255
        nc.data.draw_overlay(im_print, ans)
        return im_print

    
def showfilter(model):
    filter_len = 512 if model_name == 'resnet18' else 2048
    cut_size = 128
    cuts = int(filter_len / cut_size)
    for i in range(cuts):
        w = model.fc2.weight.cpu().detach().numpy()
        w = np.squeeze(w)
        plt.matshow(w[:,i*cut_size:(i+1)*cut_size])
        plt.title(f'Fully-connected layer filter. Block part {i}/{cuts}')
        
        
def debug_model(model):
    showfilter(model)
    # 3: hard
    # 9: blank
    # 10: orange
    # 14: easy, but machine performs poorly.
    interesting_samples = [test_ds[i] for i in (3, 8, 9, 10, 12, 13, 14, 17)]
    #ic(val_ds[0]['image'].shape)
    images = []
    labels = []
    print('Interesting examples:')
    for s in interesting_samples:
        im = debug_img(model, s['image'])
        images.append(im)
        labels.append(s['label'])
    imlist(images, labels)
    
    print('Other examples:')
    print("(Tab by correct label. The grids are the model outputs)")
    num_samples = 50
    images, labels = zip(*map(lambda s: (debug_img(model, s['image']), s['label']), 
                              (test_ds[s] for s in range(num_samples))))
    imlist(images, labels, use_tabs=True)
```

### 8. Run Experiment. 
Train one of two models (resnet18 or resnet50) with 4 types of losses with gradient descent. Test the accuracy of each model-loss pair.


```python
LossTest = namedtuple('LossTest', ['name', 'fctn', 'num_epochs'])
losses = [
    LossTest('mle',      mle,          20),
    LossTest('balanced', mle_balanced, 20),
    LossTest('dot',      mle_dot,      20),
    LossTest('radial',   mle_radial,   20)]
models = {}
def testall():
    pretrained = True
    save_path = './resources/exp_1_4/{model}_{loss}_loss_model_save'
    for loss in losses:
        model = copy.deepcopy(edited_resnet_model).to(device)
        if pretrained:
            model.load_state_dict(torch.load(save_path.format(model=model_name, loss=loss.name)))
        else:
            if not loss.num_epochs:
                continue
            print(f"Testing loss: {loss.name}")
            # Observe that all parameters are being optimized
            optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.9)
            model = train_model(model, loss.fctn, optimizer_ft, exp_lr_scheduler, 
                                num_epochs=loss.num_epochs)
            torch.save(model.state_dict(), save_path.format(model=model_name, loss=loss.name))
        models[loss.name] = model
        test_model(models[loss.name])
testall()
```


    
![png](experiment_1_4_log_files/experiment_1_4_log_41_0.png)
    



    
![png](experiment_1_4_log_files/experiment_1_4_log_41_1.png)
    


    Test set confusion matrix data:
    tensor([[7.6923e-01, 1.5385e-01, 0.0000e+00, 7.6923e-02],
            [1.6667e-01, 6.6667e-01, 0.0000e+00, 1.6667e-01],
            [       nan,        nan,        nan,        nan],
            [1.4249e-03, 8.7239e-04, 0.0000e+00, 9.9770e-01]], device='cuda:0')
    Test set confusion matrix data (dot positions only):
    tensor([[0.7692, 0.1538, 0.0000, 0.0769],
            [0.1667, 0.6667, 0.0000, 0.1667],
            [   nan,    nan,    nan,    nan],
            [0.0943, 0.0578, 0.0000, 0.8479]], device='cuda:0')
    Test set unnormalized confusion matrix data:
    tensor([[4.9000e+02, 9.8000e+01, 0.0000e+00, 4.9000e+01],
            [4.9000e+01, 1.9600e+02, 0.0000e+00, 4.9000e+01],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [2.4500e+02, 1.5000e+02, 0.0000e+00, 1.7155e+05]], device='cuda:0')
    Test set unnormalized confusion matrix data (dot positions only):
    tensor([[ 490.,   98.,    0.,   49.],
            [  49.,  196.,    0.,   49.],
            [   0.,    0.,    0.,    0.],
            [ 245.,  150.,    0., 2202.]], device='cuda:0')
    Grid tallying correct dot predictions
    tensor([[59., 59., 59., 59., 59., 59., 60.],
            [61., 59., 58., 58., 58., 58., 60.],
            [59., 60., 58., 58., 58., 59., 58.],
            [59., 60., 58., 58., 59., 59., 58.],
            [59., 60., 58., 58., 59., 59., 58.],
            [59., 60., 58., 59., 60., 59., 59.],
            [60., 61., 59., 59., 59., 59., 59.]], device='cuda:0')



    
![png](experiment_1_4_log_files/experiment_1_4_log_41_3.png)
    



    
![png](experiment_1_4_log_files/experiment_1_4_log_41_4.png)
    


    Test set confusion matrix data:
    tensor([[0.7692, 0.1538, 0.0000, 0.0769],
            [0.1667, 0.6667, 0.0000, 0.1667],
            [   nan,    nan,    nan,    nan],
            [0.0014, 0.0010, 0.0000, 0.9976]], device='cuda:0')
    Test set confusion matrix data (dot positions only):
    tensor([[0.7692, 0.1538, 0.0000, 0.0769],
            [0.1667, 0.6667, 0.0000, 0.1667],
            [   nan,    nan,    nan,    nan],
            [0.0943, 0.0662, 0.0000, 0.8394]], device='cuda:0')
    Test set unnormalized confusion matrix data:
    tensor([[4.9000e+02, 9.8000e+01, 0.0000e+00, 4.9000e+01],
            [4.9000e+01, 1.9600e+02, 0.0000e+00, 4.9000e+01],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [2.4500e+02, 1.7200e+02, 0.0000e+00, 1.7152e+05]], device='cuda:0')
    Test set unnormalized confusion matrix data (dot positions only):
    tensor([[ 490.,   98.,    0.,   49.],
            [  49.,  196.,    0.,   49.],
            [   0.,    0.,    0.,    0.],
            [ 245.,  172.,    0., 2180.]], device='cuda:0')
    Grid tallying correct dot predictions
    tensor([[59., 58., 58., 58., 58., 58., 57.],
            [59., 59., 59., 60., 58., 58., 58.],
            [59., 59., 59., 59., 59., 59., 58.],
            [59., 59., 58., 58., 58., 59., 57.],
            [59., 59., 58., 58., 58., 58., 57.],
            [59., 59., 59., 59., 59., 59., 58.],
            [59., 59., 59., 58., 58., 59., 58.]], device='cuda:0')



```python
# Investigate one of the models in more detail.
debug_model(models['radial'])
test_ds._labelled_colors.sort(key=lambda a:a[0])
```

## 9. Results
Low accuracy is some evidence to suggest that no enough information reaches the neurons in the target layer. However, there are a number of reasons why this evidence is quite weak.

## 9.1 Experiment issues
I have a number of concerns:

1. the dataset feels very small. This became especially obvious once the test set was shaved off and brown getting only XX different color codes.
2. I don't have any evidence that my mapping between dots and activations is effective. I need to check which image pixels feed into which activations, and to try place dots in the center of these calculated windows. A poor correspondence could result in the filter needing to locate orange/brown in a different position relative to its window depending on which activation it is calculating.
3. the distinction of orange/brown relies to relative brightnesses, and so, it seems likely that the size of the center dot is important. I have scaled down the dot without checking to see if the dataset classifications are affected.
4. The model uses MLE via softmax activations. Nothing inherently wrong with this choice; however, it doesn't take advantage of an important assumption: I think orange and brown can be linearly separable on a single dimension. Projecting the 2048 features (or 512 for resnet18) onto two dimensions (weight + bias) might constrain the model in a useful way. If high accuracy can be achieved under this constraint, then this might be used to reverse engineer the model to extract a color dimension (the dimension for "related" colors). 

## 10. Next steps
Some useful investigations to do next:

1. find a model for orange/brown such that we can generate the color pairs more freely, without being restricted to the dataset. This tackles issue #1 above.
2. Investigate the window of the input image that contributes to each activation. This tackles issue #2 above.
3. Another way of investigating issue #2 is to restrict the classification to only 1 of the 49 activations and then check what accuracy we get. Extend 1 to 3x3 then 4x4 etc.
we have 99% accuracy. Looking at the accuracy metric shows that it's not very useful: an accuracy of 48/49=98% can be achieved by just returning 0 for every square. To get a better idea of the performance, let's take a look at some specific examples.
4. Implement an alternative dataset, such as red-blue color distinction. This dataset will be useful to form a comparison and to help find any possible bugs.
5. Follow up of issue #4.
