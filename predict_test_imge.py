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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATA_FOLDER = '../image_A'
BATCH_SIZE=600
img_name_list = os.listdir(DATA_FOLDER)
img_name_list.sort()
N_CLASSES=8
LABELS =['water','Transportation','building','cultivated area','grassland','woodland','Soil','others']
matches = [100, 200, 300, 400, 500, 600, 700, 800]


net=torch.nn.DataParallel(DANet(nclass=8, backbone='resnet101'), device_ids=[0])
net.cuda()

net.load_state_dict(torch.load('./DA_DF_baseline_3*3_epoch30.pth'))

PRED_FOLDER = "./results"

def save_img(net,DATA_FOLDER,PRED_FOLDER):

    # Use the network on the test set

    # Switch the network to inference mode
    net.eval()
    for img_name in img_name_list:
        test_image = 1 / 255 * np.asarray(io.imread(os.path.join(DATA_FOLDER,img_name)).transpose((2, 0, 1)), dtype='float32')
        test_image = torch.from_numpy(test_image).cuda().unsqueeze(0)
        outs = net(test_image)
        outs = F.softmax(outs[0], dim=1)
        outs = outs.data.cpu().numpy() ## 1*C*H*W
        pred = np.argmax(outs, axis=1) ## 1*H*W
        pred = np.squeeze(pred,axis=0)

        saved_path = os.path.join(PRED_FOLDER,img_name.split('.')[0]+'.png')

        pred_img = convert_to_uint16(pred,matches)

        saved_img = Image.fromarray(pred_img)
        saved_img.save(saved_path, format='PNG')


save_img(net,DATA_FOLDER,PRED_FOLDER)