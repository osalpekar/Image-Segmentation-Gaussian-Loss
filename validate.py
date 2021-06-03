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
from data import get_data, create_siamese_databunch, create_standard_databunch

# TODO: Change this to correct model name when doing eval
MODEL_NAME = "./checkpoints/base_unet_1epoch_500.pkl"
ROOT_PATH = "/home/ubuntu/cs231n_project/"

if __name__ == "__main__":
    # Load the trained and serialized model
    data = create_standard_databunch(16, None)
    learn = load_learner(Path(ROOT_PATH), MODEL_NAME)
    learn.data = data

    # Get predictions on the validation set
    preds, targets = learn.get_preds(DatasetType.Valid)

    # Compute the Jaccard Index given the predictions
    jaccard_score = learn.metrics[0](preds, targets)

    print("Jaccard Score for Model: {}".format(jaccard_score))
