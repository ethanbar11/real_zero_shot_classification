import torch
import numpy as np
from torch.utils.data import Dataset
from fvcore.common.config import CfgNode

########################################
########################################
# Get Loaders
########################################
########################################
def construct_loaders(trainset: Dataset, valset: Dataset, testset: Dataset, cfg: CfgNode):
    """
    Constructs the data loader for the given datasets.
    Args:
        cfg (CfgNode): configs. Details can be found in config/defaults.py
        trainset, valset, testset: the datasets of the data loader, inherited from torch.utils.data.Dataset
    """
    overfit = cfg.DATA_LOADER.OVERFIT
    if overfit:  # sample identical very few examples for both train ans val sets:
        num_samples_for_overfit = 10
        sampled = np.random.choice(np.arange(len(trainset)), num_samples_for_overfit)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(sampled),
                                                  shuffle=False, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(sampled),
                                                 shuffle=False, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
        testloader = None
        print("DATA: Sampling identical sets of {} ANNOTATED examples for train and val sets.. ".format(num_samples_for_overfit))

    else:
        # TODO - make it work with batch size > 1
        trainloader, valloader, testloader = [], [], []
        # --- Train: ---
        if trainset is not None:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                                      shuffle=True, pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                                      num_workers=cfg.DATA_LOADER.NUM_WORKERS)
        # --- Val: ---
        if valset is not None:
            valloader = torch.utils.data.DataLoader(valset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                                     shuffle=False, pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                                     num_workers=cfg.DATA_LOADER.NUM_WORKERS)
        # --- Test: ---
        if testset is not None:
            testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.TEST.BATCH_SIZE,
                                                    shuffle=False, pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                                    num_workers=cfg.DATA_LOADER.NUM_WORKERS)

    return trainloader, valloader, testloader



