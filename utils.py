# Utils
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
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from torch.backends import cudnn
import cv2
random.seed(4)
torch.manual_seed(4)
torch.cuda.manual_seed(4)
np.random.seed(4)
cudnn.deterministic = True


def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def get_pos(img, patch_idx, stride, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    row = int(img.shape[1] / stride) + 1
    x1 = random.randint(int((patch_idx % row - 1) * stride), int((patch_idx % row) * stride - 1))
    if x1 + window_shape[0] > img.shape[1]:
        x1 = img.shape[1] - window_shape[0]
    if x1 < 0:
        x1 = 0
    y1 = random.randint(int((patch_idx // row - 1) * stride), int((patch_idx // row) * stride - 1))
    if y1 + window_shape[1] > img.shape[2]:
        y1 = img.shape[2] - window_shape[1]
    if y1 < 0:
        y1 = 0
    x2 = x1 + w
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values):
    cm = confusion_matrix(
        gts,
        predictions,
        range(len(label_values)))
    #     cm=cm[0:5,0:5]

    # print("Confusion matrix :")
    # print(cm)
    #
    # print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    # print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    #
    # print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    #     F1Score = np.zeros(len(label_values)-1)
    for i in range(len(label_values)):
        #     for i in range(len(label_values)-1):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    # print("F1Score :")
    # for l_id, score in enumerate(F1Score):
    #     print("{}: {}".format(label_values[l_id], score))
    #
    # print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe);
    # print("Kappa: " + str(kappa))
    return accuracy, cm




def convert_to_color(arr_2d, palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_to_uint16(pred,matches):
    for index, m in enumerate(matches):
        pred[pred == index] = m
    pred = pred.astype(np.uint16)
    return pred

def convert_from_uint16(label,matches):
    for index,m in enumerate(matches):
        label[label == m] = index
    return label

def convert_from_color(arr_3d, palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d