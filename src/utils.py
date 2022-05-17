import os
import glob
import tempfile
import json
from init import Options
from datetime import datetime
opt = Options().parse()

def printf(*arg, **kwarg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(timestamp, *arg, **kwarg)

def set_task_dir(task):
    directory = tempfile.mkdtemp() if opt.root is None else opt.root
    root_dir = os.path.join(directory, task)
    return root_dir

def train_val_split(train, label):
    train_images = sorted(
        glob.glob(os.path.join(set_task_dir(opt.task), train, opt.type))) #Possible addition of extra opt for 
    train_labels = sorted(
        glob.glob(os.path.join(set_task_dir(opt.task), label, opt.type)))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    t_v_split = round(len(data_dicts)*opt.split)
    trainF, valF = data_dicts[t_v_split:], data_dicts[:t_v_split]
    return trainF, valF

def importCfg(config, task):
    with open(f"{os.path.join(config, task)}.json", "r") as jsonfile:
        cfg = json.load(jsonfile)
    return cfg

