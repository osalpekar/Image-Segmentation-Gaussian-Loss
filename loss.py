import os
import sys
import glob

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
from fastai.vision import *
from sklearn.mixture import GaussianMixture

class DMILoss(nn.Module):
    reduction = 'none'
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        output = output.float()
        target = target.float()
        #print(output.shape)
        #print(target.shape)
        print(output)
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

class BootstrapStaticLoss(nn.Module):
    reduction = 'none'
    def __init__(self, bootstrap_type="soft"):
        super().__init__()
        print("LOSS INIT")

        self.bootstrap_type = bootstrap_type
        if bootstrap_type == "soft":
            self.w = 0.05
        elif bootstrap_type == "hard":
            self.w = 0.2

    def forward(self, output, target):
        feature_dim = 2 * 256 * 256
        output = output.float()
        target = target.float() 

        inv_target = 1 - target
        target = torch.cat((target, inv_target), 1)
        h_out = F.softmax(output, dim=1)
        preds = (h_out > 0.5).float()
        
        y = target.view(-1, feature_dim)
        z = preds.view(-1, feature_dim)
        h = h_out.view(-1, feature_dim)
        o = output.view(-1, feature_dim)

        weighted_true = ((1 - self.w) * y)
        weighted_pred = 0
        if self.bootstrap_type == "soft":
            weighted_pred = (self.w * h)
        elif self.bootstrap_type == "hard":
            weighted_pred = (self.w * z)

        weighted_out = weighted_true + weighted_pred
        log_h = torch.log(h)

        return torch.abs(torch.sum(weighted_out * log_h))

class BootstrapDynamicLoss(nn.Module):
    reduction = 'none'
    def __init__(self, bootstrap_type="soft", mixture_type="gaussian"):
        super().__init__()
        print("LOSS INIT")

        self.bootstrap_type = bootstrap_type

        self.mixture_type = mixture_type 
        self.mixture_momentum = 0.9
        if mixture_type == "gaussian":
            self.mixture = GaussianMixture(n_components=2)

        #self.running_loss = np.zeros((16,))

    def forward(self, output, target):
        feature_dim = 2 * 256 * 256
        output = output.float()
        target = target.float() 

        inv_target = 1 - target
        target = torch.cat((target, inv_target), 1)
        h_out = F.softmax(output, dim=1)
        preds = (h_out > 0.5).float()
        
        y = target.view(-1, feature_dim)
        z = preds.view(-1, feature_dim)
        h = h_out.view(-1, feature_dim)
        o = output.view(-1, feature_dim)

        # vector of losses per example
        raw_losses = F.binary_cross_entropy_with_logits(o, y, reduction="none")
        losses = raw_losses.sum(axis=1)
        losses = losses.to("cpu").detach().numpy()
        #if losses.shape[0] != 16:
        #    losses = np.pad(losses, 16 - losses.shape[0])
        # vector of distributions fit
        #self.running_loss = self.mixture_momentum * self.running_loss + (1 - self.mixture_momentum) * losses
        classes = self.mixture.fit_predict(losses.reshape(-1, 1))
        weights = self.mixture.means_[classes].flatten()
        weights = torch.from_numpy(weights).to("cuda:0")
        #print(weights.shape)
        #print(y.shape)

        weighted_true = ((torch.ones(losses.shape[0]).to("cuda:0") - weights).unsqueeze(dim=1) * y)
        weighted_pred = 0
        if self.bootstrap_type == "soft":
            weighted_pred = (weights.unsqueeze(dim=1) * h)
        elif self.bootstrap_type == "hard":
            weighted_pred = (weights.unsqueeze(dim=1) * z)

        weighted_out = weighted_true + weighted_pred
        log_h = torch.log(h)

        return torch.abs(torch.sum(weighted_out * log_h))**float(1/3)
