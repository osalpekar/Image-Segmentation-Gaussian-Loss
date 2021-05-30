import os
import sys
import glob

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
from fastai.vision import *

# Image File Paths
IMG_DIR = "../data/images-256/*.png"
MASK_DIR = "../data/masks-256/*.png"

# DataFrame serialized filename
DF_PATH = "./data.pkl"

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
    #df.loc[df.area == 'dar', 'validation'] = True
    np.random.seed(100)
    val_idx = np.random.choice(len(df), replace=False, size=1000)
    df.iloc[val_idx, df.columns.get_loc('validation')] = True
    
    return df

class SiamDataset(Dataset):
    def __init__(self, is_validation=False, num_train_examples=None):
        df = pd.read_pickle(DF_PATH)
        if num_train_examples is not None:
            df = df.head(num_train_examples)
        df = df[df['validation'] == is_validation]
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row1 = self.df.iloc[idx]
        row2 = self.df.iloc[(idx+1) % len(self.df)]

        img1 = PIL.Image.open(row1.img_path)
        img2 = PIL.Image.open(row2.img_path)
        mask1 = PIL.Image.open(mask1.mask_path)
        mask2 = PIL.Image.open(mask2.mask_path)

        img1 = torch.from_numpy(img1).float()/255
        img2 = torch.from_numpy(img2).float()/255
        mask1 = torch.from_numpy(mask1).float()
        mask2 = torch.from_numpy(mask2).float()
        label = torch.randint(0,1,(1,)) # This would be label on whether they are smae or differet

        return img1, img2, mask1, mask2, label

def create_standard_databunch(batch_size, num_train_examples):
    # Override open method of SegmentationLabelList since our masks are 0 for class 0, 255 for class 1 (so need div=True)
    def my_open(self, fn): return open_mask(fn, div=True)
    SegmentationLabelList.open = my_open

    df = pd.read_pickle(DF_PATH)
    print(df.head())
    df_small = df.head(num_train_examples)
    src = (SegmentationItemList.from_df(path='', df=df_small, cols='img_path')
           .split_from_df(col='validation')
           .label_from_df(cols='mask_path', classes=["building", "not"]))

    data = (src.transform(get_transforms(), size=256, tfm_y=True)
            .databunch(bs=batch_size) # Change batch size if you're having memory issues
            .normalize(imagenet_stats))

    return data

def create_siamese_databunch(batch_size, num_train_examples):
    train_ds = SiamDataset(False, num_train_examples)
    valid_ds = SiamDataset(True, num_train_examples)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    data = DataBunch(train_dl, valid_dl)

    return data

if __name__ == "__main__":
    df = get_data()
    df.to_pickle(DF_PATH)
