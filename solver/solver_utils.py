import os
import shutil
import time
import pickle

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR

def build_scheduler(optimizer, optimizer_config, num_iters_per_epoch,last_epoch=-1):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=optimizer_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler

def build_optimizer(cfg, params):
    kwargs = dict(lr=cfg.SOLVER.LR,
                  weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM, **kwargs)
    elif cfg.SOLVER.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(params, **kwargs)
    elif cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(params, **kwargs)
    else:
        raise NotImplementedError
    return optimizer