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
from loss import BootstrapStaticLoss, BootstrapDynamicLoss, ContrastiveLoss, DMILoss
from data import get_data, create_siamese_databunch, create_standard_databunch
from custom_models import SiameseNet

# Image File Paths
IMG_DIR = "../data/images-256/*.png"
MASK_DIR = "../data/masks-256/*.png"

# Important Constants
NUM_TRAIN_EXAMPLES = 32
BATCH_SIZE = 16
NUM_EPOCHS = 1
LR = 5e-4 

MODEL_NAME = "./checkpoints/base_unet.pkl"
ROOT_PATH = "/home/ubuntu/cs231n_project/"

#df = get_data()
#print(df.head())

# How many ims for each area?
#df.groupby('area').count()['scene_id']


#df_small = df.head(NUM_TRAIN_EXAMPLES)


# Check out a batch
#data.show_batch(2, figsize=(10,7))


dmi_loss = DMILoss()
bootstrap_static_soft_loss = BootstrapStaticLoss("soft")
bootstrap_static_hard_loss = BootstrapStaticLoss("hard")
bootstrap_dynamic_soft_loss = BootstrapDynamicLoss("soft")
bootstrap_dynamic_hard_loss = BootstrapDynamicLoss("hard")

losses = {
    "bootstrap_static_soft_loss"  : bootstrap_static_soft_loss,
    "bootstrap_static_hard_loss"  : bootstrap_static_hard_loss,
    "bootstrap_dynamic_soft_loss" : bootstrap_dynamic_soft_loss,
    "bootstrap_dynamic_hard_loss" : bootstrap_dynamic_hard_loss,
}


def create_model(model_type="unet", loss_type="base"):
    if model_type == "unet":
        data = create_standard_databunch(BATCH_SIZE, NUM_TRAIN_EXAMPLES)
        print("Done assembling databunch")
        learn = unet_learner(data, models.resnet34, metrics=jaccard)
        if loss_type != "base":
            learn.loss_func = losses[loss_type]

    elif model_type == "siamese":
        data = create_siamese_databunch(BATCH_SIZE, NUM_TRAIN_EXAMPLES)
        model = SiameseNet().cuda()
        loss_func = ContrastiveLoss().cuda()
        learn = Learner(data, model, loss_func=loss_func)#, model_dir=Path(os.getcwd()))

    return learn


def find_lr(learn):
    learn.lr_find()
    graph = learn.recorder.plot(return_fig=True)
    graph.savefig('loss.png')


def train():
    learn = create_model(
        model_type="unet",
        loss_type="base",
    )

    # find_lr(learn)

    learn.fit_one_cycle(
        NUM_EPOCHS,
        slice(LR),
    )
    learn.export(MODEL_NAME)


if __name__ == "__main__":
    train()
