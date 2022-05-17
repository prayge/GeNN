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

sns.set()
plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()
opt = Options().parse()

get_root(opt.root_dir)