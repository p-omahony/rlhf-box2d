import torch
import os
import yaml
from yaml import Loader

def save_weights(model, outputfilename):
    basepath = './weights'
    path = os.path.join(basepath, outputfilename)
    if not os.path.exists(basepath):
        os.mkdir(basepath)
    torch.save(model.state_dict(), path)

def load_config(cfg_path):
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader)
    return cfg

def repeat_items(l, c):
    return l * (c // len(l)) + l[:(c % len(l))]