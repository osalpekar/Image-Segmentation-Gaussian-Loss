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
from data import get_data, SiamDataset

# Image File Paths
IMG_DIR = "../data/images-256/*.png"
MASK_DIR = "../data/masks-256/*.png"

# Important Constants
NUM_TRAIN_EXAMPLES = 32
BATCH_SIZE = 16
NUM_EPOCHS = 2
LR = 1e-5 


df = get_data()
print("completed Data Parsing")
print(df.head())

# How many ims for each area?
#df.groupby('area').count()['scene_id']


df_small = df.head(NUM_TRAIN_EXAMPLES)
"""
# Override open method of SegmentationLabelList since our masks are 0 for class 0, 255 for class 1 (so need div=True)
def my_open(self, fn): return open_mask(fn, div=True)
SegmentationLabelList.open = my_open


np.random.seed(42)
src = (SegmentationItemList.from_df(path='', df=df_small, cols='img_path')
       .split_from_df(col='validation')
       .label_from_df(cols='mask_path', classes=["building", "not"]))

data = (src.transform(get_transforms(), size=256, tfm_y=True)
        .databunch(bs=BATCH_SIZE) # Change batch size if you're having memory issues
        .normalize(imagenet_stats))
"""

train_ds = SiamDataset(is_validation=False)
valid_ds = SiamDataset(is_validation=True)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
data = DataBunch(train_dl, valid_dl)

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
"""

src = (SegmentationItemList.from_df(path='', df=df_small[:4], cols='img_path')) # Did in two batches
#learn = load_learner('') # Loads the exported learner from earlier
learn.data.add_test(src, tfms=None, tfm_y=False)
preds, y = learn.get_preds(DatasetType.Test) # Careful with ram - do in batches if needed ()
#p,_,_ = learn.predict(df_small.head(1))
print(preds)
