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

from metrics import jaccard
from loss import BootstrapStaticLoss, BootstrapDynamicLoss, DMILoss
from data import get_data, create_siamese_databunch, create_standard_databunch

# Image File Paths
IMG_DIR = "../data/images-256/*.png"
MASK_DIR = "../data/masks-256/*.png"

# Important Constants
NUM_TRAIN_EXAMPLES = 32
BATCH_SIZE = 16
NUM_EPOCHS = 1
LR = 1e-5 

MODEL_NAME = "./test.pkl"
ROOT_PATH = "/home/ubuntu/cs231n_project/"

#df = get_data()
#print(df.head())

# How many ims for each area?
#df.groupby('area').count()['scene_id']


#df_small = df.head(NUM_TRAIN_EXAMPLES)

data = create_standard_databunch(BATCH_SIZE, NUM_TRAIN_EXAMPLES)
#data = create_siamese_databunch(BATCH_SIZE, NUM_TRAIN_EXAMPLES)

print("completed Data Parsing")
# Check out a batch
#data.show_batch(2, figsize=(10,7))
print("Done assembling databunch")

class SiameseNet(nn.Module):
    def __init__(self):
        self.net = DynamicUnet(models.resnet34(), 2, (256, 256))

    def forward(self, in1, in2):
        out1 = self.net(in1)
        out2 = self.net(in2)

        return out1, out2

class ContrastiveLoss(nn.Module):
    """Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin=5.):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, ops, target, size_average=True):
        op1, op2 = ops[0], ops[1]
        dist = F.pairwise_distance(op1, op2)
        pdist = dist*target
        ndist = dist*(1-target)
        loss = 0.5* ((pdist**2) + (F.relu(self.margin-ndist)**2))
        return loss.mean() if size_average else losses.sum()

"""
model = SiameseNet().cuda()
#apply_init(model.head, nn.init.kaiming_normal_)
loss_func=ContrastiveLoss().cuda()
learn = Learner(data, model, loss_func=loss_func)#, model_dir=Path(os.getcwd()))
learn.fit_one_cycle(NUM_EPOCHS, slice(LR))


"""
dmi_loss = DMILoss()

bootstrap_loss = BootstrapDynamicLoss()

learn = unet_learner(data, models.resnet34, metrics=jaccard)
learn.loss_func = bootstrap_loss

print("Constructed trainer")

# Train!
learn.fit_one_cycle(NUM_EPOCHS, slice(LR))
learn.export(MODEL_NAME)
