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
        batch_size=2,
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
        accelerator="gpu",
        precision=16,
        callbacks=[early_stopping],
    )
    trainer.logger._default_hp_metric = False

    start = datetime.now()
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', datetime.now() - start)

    model.to('cuda')
    all_dices = []
    get_dice = monai.metrics.DiceMetric(include_background=False, reduction='none')
    with torch.no_grad():
        for batch in data.val_dataloader():
            inputs, targets = model.prepare_batch(batch)
            logits = model.net(inputs.to(model.device))
            labels = logits.argmax(dim=1)
            labels_one_hot = torch.nn.functional.one_hot(labels).permute(0, 4, 1, 2, 3)
            get_dice(labels_one_hot.to(model.device), targets.to(model.device))
        metric = get_dice.aggregate()
        get_dice.reset()
        all_dices.append(metric)
    all_dices = torch.cat(all_dices)

    records = []

    for ant, post in all_dices:
        records.append({'Dice': ant, 'Label': 'Anterior'})
        records.append({'Dice': post, 'Label': 'Posterior'})
    df = pd.DataFrame.from_records(records)
    ax = sns.stripplot(x='Label', y='Dice', data=df, size=10, alpha=0.5)
    ax.set_title('Dice scores')

    with torch.no_grad():
        for batch in data.test_dataloader():
            inputs = batch['image'][tio.DATA].to(model.device)
            labels = model.net(inputs).argmax(dim=1, keepdim=True).cpu()
            break
    batch_subjects = tio.utils.get_subjects_from_batch(batch)
    tio.utils.add_images_from_batch(batch_subjects, labels, tio.LabelMap)

    for subject in batch_subjects:
        subject.plot()

if __name__ == '__main__':
    main()