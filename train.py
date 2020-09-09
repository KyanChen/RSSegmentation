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
from PIL import Image
from resnet import resnet101
from torch.nn.functional import interpolate,normalize
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from model import DANet
from detaset import DF_dataset
import os
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2,3"
random.seed(4)
torch.manual_seed(4)
torch.cuda.manual_seed(4)
np.random.seed(4)
cudnn.deterministic = True


# Parameters

WINDOW_SIZE = (256, 256) # Patch size
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "../train/"  # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 48 # Number of samples in a mini-batch


PRED_FOLDER = "./"

N_CLASSES = 8 # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (255, 0, 255),
           7 : (0, 0, 0)}       # Undefined (black)
matches = [100, 200, 300, 400, 500, 600, 700, 800]
invert_palette = {v: k for k, v in palette.items()}

CACHE = True # Store the dataset in-memory

DATA_FOLDER = '../train/image'
LABEL_FOLDER = '../train/label'


## network define


class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(self, se_loss=False, se_weight=0.2, nclass=6,
                 aux=False, aux_weight=0.2, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, size_average, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, size_average)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3


class SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""

    def __init__(self, nclass=-1, weight=None, size_average=True, ignore_index=-1):
        super(SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        pred1, pred2, pred3 = tuple(preds)

        loss1 = super(SegmentationMultiLosses, self).forward(pred1, target)
        #         loss2 = super(SegmentationMultiLosses, self).forward(pred2, target)
        #         loss3 = super(SegmentationMultiLosses, self).forward(pred3, target)
        #         loss = loss1 + loss2 + loss3
        loss = loss1
        return loss



# net = DANet(6, backbone='resnet101')
net=torch.nn.DataParallel(DANet(nclass=8, backbone='resnet101'), device_ids=[0, 1, 2,3])
net.cuda()
net.load_state_dict(torch.load('./DA_DF_baseline_3*3_epoch30.pth'))

# Random tile numbers for train/test split
# train_ids = random.sample(all_ids, 2 * len(all_ids) // 3 + 1)
# test_ids = list(set(all_ids) - set(train_ids))

# Exemple of a train/test split on Vaihingen :

# Exemple of a train/test split on Vaihingen :


train_set = DF_dataset(data_files=DATA_FOLDER, label_files=LABEL_FOLDER)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)

# criterion=Label_relax_loss(weight=WEIGHTS)
criterion=SegmentationMultiLosses(weight=WEIGHTS)
# criterion=Focalloss(weight=WEIGHTS,gamma=0)
criterion.cuda()

base_lr = 0.01
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if 'pretrained' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr/2}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr}]
# for key, value in criterion.named_parameters():
#     params += [{'params':[value],'lr': base_lr}]

optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40], gamma=0.1)

def adjust_learning_rate(optimizer,base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(net, optimizer, epochs, scheduler=None, weights=None, save_epoch=5):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(1000000)

    #     criterion = nn.NLLLoss2d(weight=weights)
    iter_ = 0

    for e in range(1, epochs + 1):
        #         for i in range(25):
        #             if scheduler is not None:
        #                 scheduler.step()
        adjust_learning_rate(optimizer,base_lr, 50, e, power=0.9)
        # if scheduler is not None:
        #     scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            # print(torch._unique(torch.argmax(output[0], dim=1)))
            #             loss = CrossEntropy2d(output, target, weight=weights)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='uint8')
                pred = np.argmax(output[0].data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), accuracy(pred, gt)))
                plt.plot(mean_losses[:iter_]) and plt.show()
                # fig = plt.figure()
                # fig.add_subplot(131)
                # plt.imshow(rgb)
                # plt.title('RGB')
                # fig.add_subplot(132)
                # plt.imshow(convert_to_color(gt,palette))
                # plt.title('Ground truth')
                # fig.add_subplot(133)
                # plt.title('Prediction')
                # plt.imshow(convert_to_color(pred,palette))
                # plt.show()
            iter_ += 1

            del (data, target, loss)


        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            # acc, cm = test(net, val_ids, all=False, stride=64)
            # for i in range(6):
            #     cm[i, :] = cm[i, :] / np.sum(cm[i, :])

            # acc, cm = test(net, test_ids, all=False, stride=64)
            # print(criterion.inv_T)
            torch.save(net.state_dict(), './DA_DF_baseline_3*3_epoch{}'.format(e))
    torch.save(net.state_dict(), './segnet_final')


train(net, optimizer, 50, weights=WEIGHTS)