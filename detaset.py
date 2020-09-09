# imports and stuff
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# Matplotlib
import matplotlib.pyplot as plt
# Torch imports
import torch
import torch.utils.data as data
import torch.optim.lr_scheduler
import torch.nn.init
import os
from utils import *

matches = [100, 200, 300, 400, 500, 600, 700, 800]

class DF_dataset(torch.utils.data.Dataset):
    def __init__(self, data_files, label_files, augmentation=True):
        super(DF_dataset, self).__init__()

        self.augmentation = augmentation

        # Sanity check : raise an error if some files do not exist
        # Initialize cache dicts
        img_name_list = os.listdir(data_files)
        img_name_list.sort()
        gt_name_list = os.listdir(label_files)
        gt_name_list.sort()
        self.data_cache_ = {}
        self.label_cache_ = {}
        self.train_images_file = [os.path.join(data_files,image_name) for image_name in img_name_list]

        self.train_labels_file = [os.path.join(label_files,gt_name) for gt_name in gt_name_list]

        self.total = len(gt_name_list)

    def __len__(self):
        # Default epoch size is 10 000 samples
        return self.total

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # find the image index

        # Data is normalized in [0, 1]

        data = 1 / 255 * np.asarray(io.imread(self.train_images_file[i]).transpose((2, 0, 1)), dtype='float32')
        label=io.imread(self.train_labels_file[i])
        label = convert_from_uint16(label,matches)


        label=label.astype(np.int64)
        # print(np.unique(label))
        # Data augmentation
        data_p, label_p = self.data_augmentation(data, label)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))


