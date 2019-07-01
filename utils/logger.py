"""

Training Logger

"""
import torch
from cfgs import config as cfg


class Logger:

    def __init__(self):
        pass


def save_checkpoints(model, optimizer, epoch, iteration, path):

    config = cfg.config['coco_baseline']

    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "iteration": iteration
    }

    if epoch % 100 == 0 or epoch == (config['epochs'] - 1):
        torch.save(state_dict, path)


def load_checkpoints(path):
    state_dict = torch.load(path)

    return state_dict['model'], state_dict['optimizer'], state_dict['epoch'], state_dict['iteration']

