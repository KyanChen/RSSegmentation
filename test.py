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
import os
from PIL import Image
from resnet import resnet101
from torch.nn.functional import interpolate,normalize
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from utils import *
from model import DANet
from detaset import DF_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2,3,4,5"

DATA_FOLDER = '../train/image'
LABEL_FOLDER = '../train/label'
BATCH_SIZE=600
img_name_list = os.listdir(DATA_FOLDER)
img_name_list.sort()
gt_name_list = os.listdir(LABEL_FOLDER)
gt_name_list.sort()
N_CLASSES=8
LABELS =['water','Transportation','building','cultivated area','grassland','woodland','Soil','others']



net=torch.nn.DataParallel(DANet(nclass=8, backbone='resnet101'), device_ids=[0, 1, 2,3,4,5])
net.cuda()

net.load_state_dict(torch.load('./DA_DF_baseline_3*3_epoch30.pth'))

train_set = DF_dataset(data_files=DATA_FOLDER, label_files=LABEL_FOLDER,augmentation = False)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)

def test(net,dataloader):

    # Use the network on the test set
    cm = np.zeros((8, 8))
    # Switch the network to inference mode
    net.eval()
    total = len(img_name_list)
    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(dataloader):
            print('batch:',batch_idx)
            outs = net(img)
            outs = F.softmax(outs[0], dim=1)
            outs = outs.data.cpu().numpy() ## B*C*H*W
            pred = np.argmax(outs, axis=1) ## B*H*W

            accuracy, cm_temp = metrics(pred.ravel(), gt.cpu().numpy().ravel(),LABELS)
            cm = cm + cm_temp

    #         accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())
    print("All Confusion matrix :")
    print(cm)
    print("---")

    # Compute global accuracy

    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    print("---")

    # Compute F1 score
    #     F1Score = np.zeros(N_CLASSES)
    F1Score = np.zeros(N_CLASSES - 1)
    #     for i in range(N_CLASSES):
    for i in range(N_CLASSES - 1):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(LABELS[l_id], score))

    print("---")

    # Compute kappa coefficient

    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))

    pos = cm.sum(1)
    res = cm.sum(0)
    tp = np.diag(cm)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print("mean_IoU: " + str(mean_IoU))
    print(IoU_array)


PRED_FOLDER = './test_img_pred'
DATA_FOLDER = '../image_A'
img_name_list = os.listdir(DATA_FOLDER)

def save_img(net,DATA_FOLDER ,batch_size=BATCH_SIZE):

    # Use the network on the test set

    # Switch the network to inference mode
    net.eval()
    for img_name in img_name_list:
        test_image = 1 / 255 * np.asarray(io.imread(os.path.join(DATA_FOLDER,img_name)), dtype='float32')
        outs = net(test_image)
        outs = F.softmax(outs[0], dim=1)
        outs = outs.data.cpu().numpy() ## 1*C*H*W
        pred = np.argmax(outs, axis=1) ## 1*H*W


        saved_path = os.path.join(PRED_FOLDER,img_name.split('.')[0]+'.png')

        pred_img = convert_to_uint16(pred)

        saved_img = Image.fromarray(pred_img)
        saved_img.save(saved_path, format='PNG')

test(net,train_loader)