import os
import sys
import glob

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
from fastai.vision import *

from loss import DMILoss

# Image File Paths
IMG_DIR = "../data/images-256/*.png"
MASK_DIR = "../data/masks-256/*.png"

# Important Constants
NUM_TRAIN_EXAMPLES = 300
BATCH_SIZE = 16
NUM_EPOCHS = 1
LR = 1e-5 

def get_data():
    # Get Image and Mask Filenames
    ims = glob.glob(IMG_DIR)
    masks = glob.glob(MASK_DIR)

    # Create pandas dataframe of filepaths. The validation column indicates
    # whether the column is part of the validation set or not.
    df = pd.DataFrame({
        "img_path"   : ims,
        "mask_path"  : masks,
        "validation" : False,
    })

    # Add Scene ID and area
    df['scene_id'] = df['img_path'].apply(lambda x: x.split("_")[1])
    df['area'] = df['img_path'].apply(lambda x: x.split("_")[0].split("/")[-1])

    # Let's use ptn as a valid set. Can also split randomly
    df.loc[df.area == 'dar', 'validation'] = True
    
    return df

df = get_data()
print("completed Data Parsing")
print(df.head())

# How many ims for each area?
#df.groupby('area').count()['scene_id']


# Override open method of SegmentationLabelList since our masks are 0 for class 0, 255 for class 1 (so need div=True)
def my_open(self, fn): return open_mask(fn, div=True)
SegmentationLabelList.open = my_open

df_small = df.head(NUM_TRAIN_EXAMPLES)

# Load the data from the dataframe
np.random.seed(42)
src = (SegmentationItemList.from_df(path='', df=df_small, cols='img_path')
       .split_from_df(col='validation')
       .label_from_df(cols='mask_path', classes=["building", "not"]))

data = (src.transform(get_transforms(), size=256, tfm_y=True)
        .databunch(bs=BATCH_SIZE) # Change batch size if you're having memory issues
        .normalize(imagenet_stats))

# Check out a batch
#data.show_batch(2, figsize=(10,7))
print("Done assembling databunch")


def jq(y_pred, y_true, thresh=0.5):
    scores = []
    y_pred = y_pred.to("cpu")
    y_true = y_true.to("cpu")
    for i in range(len(y_true)):
        binary_preds =  (y_pred[i][0].flatten()>thresh).int()
        score = sklearn.metrics.jaccard_score(y_true[i].flatten(), binary_preds, average='micro')
        scores.append(score)
    return torch.tensor(sum(scores)/len(scores))


dmi_loss = DMILoss()


learn = unet_learner(data, models.resnet34, metrics=jq)
learn.loss_func = dmi_loss

print("Constructed trainer")

# Train!
learn.fit_one_cycle(NUM_EPOCHS, slice(LR))


src = (SegmentationItemList.from_df(path='', df=df_small[:4], cols='img_path')) # Did in two batches
#learn = load_learner('') # Loads the exported learner from earlier
learn.data.add_test(src, tfms=None, tfm_y=False)
preds, y = learn.get_preds(DatasetType.Test) # Careful with ram - do in batches if needed ()
#p,_,_ = learn.predict(df_small.head(1))
print(preds)
