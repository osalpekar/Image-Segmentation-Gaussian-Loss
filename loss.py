import os
import sys
import glob

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
from fastai.vision import *

class DMILoss(nn.Module):
    reduction = 'none'
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        output = output.float()
        target = target.float()
        #print(output.shape)
        #print(target.shape)
        #print(output)
        #print(target)
        #print(torch.sum(output[:,:1,:,:] - target))
        #print(torch.sum(output[:,1:,:,:] - target))
        #print(output[:,:1,:,:] + output[:,1:,:,:])
        inv_target = 1 - target
        target = torch.cat((target, inv_target), 1)
        #print(target.shape)
        outputs = F.softmax(output, dim=1)
        
        outputs = outputs.view(-1, 2 * 256 * 256)
        targets = target.view(-1, 2 * 256 * 256)
        #targets = target.reshape(target.size(0), 1).cpu()
        #y_onehot = torch.FloatTensor(target.size(0), num_classes).zero_()
        #y_onehot.scatter_(1, targets, 1)
        #y_onehot = y_onehot.transpose(0, 1).cuda()
        #targets_T = torch.transpose(targets, 0, 1)

        mat = torch.matmul(outputs, targets.T)
        mat = mat# / target.size(0)
        return 1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)

class BootstrapLoss(nn.Module):
    reduction = 'none'
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        feature_dim = 2 * 256 * 256
        output = output.float()
        target = target.float() 

