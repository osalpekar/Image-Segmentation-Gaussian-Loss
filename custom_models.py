import os
import sys
import glob

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
from fastai.vision import *
from fastai.vision.models.unet import *


class SiameseNet(nn.Module):
    def __init__(self):
        self.net = DynamicUnet(models.resnet34(), 2, (256, 256))

    def forward(self, in1, in2):
        out1 = self.net(in1)
        out2 = self.net(in2)

        return out1, out2
