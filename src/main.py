from datetime import datetime
import os
import tempfile
from glob import glob
import torch
from torch.utils.data import random_split, DataLoader
import monai
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from init import Options
from utils import get_root
from data import MSD
from net import Net

sns.set()
plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()

def main():

    opt = Options().parse()

    get_root(opt.root_dir)

    data = MSD(
        task='Task04_Hippocampus',
        batch_size=16,
        train_val_ratio=0.8,
    )

    data.prepare_data()
    data.setup()
    print('Training:  ', len(data.train_set))
    print('Validation: ', len(data.val_set))
    print('Test:      ', len(data.test_set))

    unet = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
    )

    model = Net(
        net=unet,
        criterion=monai.losses.DiceCELoss(softmax=True),
        learning_rate=1e-3,
        optimizer_class=torch.optim.AdamW,
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
    )
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        callbacks=[early_stopping],
    )
    trainer.logger._default_hp_metric = False
