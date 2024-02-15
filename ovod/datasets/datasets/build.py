import os
from typing import Dict

import clip
import torch
from fvcore.common.config import CfgNode
import pickle
from ovod.utils.class_name_fix import fix_classname_special_chars

from .BasicDataset import BasicDataset
from .CUBDataset import CUBDataset
from .CUBDataset import _transform as CUB_transform
from .CUBDatasetOneAttribute import CubDatasetOneAttribute
from .CUBDatasetPartsAndAttributes import CubDatasetsPartsAndAttributes
from .FLOWERSDataset import FLOWERSDataset
from .FLOWERSDataset import _transform as FLOWER_transform
from .OxfordIIITPetDataset import OxfordIIITPetDataset
from .OxfordIIITPetDataset import _transform as OxfordIIITPetDataset_transform

from .augmentations import build_transform


# ============================
# main load dataset module:
# ============================
def build_dataset(dataset_name: str, cfg: CfgNode, split: str, **args):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            config/
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """

    if 'CUB' == dataset_name:
        loader_func = [CUBDataset]
        input_transform = [CUB_transform(cfg.TRAIN.INPUT_SIZE)]

    elif 'FLOWERS' in dataset_name:
        loader_func = [FLOWERSDataset]
        input_transform = [FLOWER_transform(cfg.TRAIN.INPUT_SIZE)]

    # elif dataset_name == 'CUB_synthetic':
    #     loader_func = [CUBDataset]
    #     data_root = [cfg.OPENSET.PATH_TO_SYNTHETIC_FOLDER]
    #     input_transform = [CUB_transform(cfg.TRAIN.INPUT_SIZE)]
    #     is_train = [True if split == "train" else False]
    #     # read from text file, remove /n at end:
    #     with open(cfg.DATA.PATH_TO_CLASSNAMES, 'r') as f:
    #         list_of_categories = f.readlines()
    #     list_of_categories = [[x.strip() for x in list_of_categories]]

    elif 'OXFORD_PET' in dataset_name:
        loader_func = [OxfordIIITPetDataset]
        input_transform = [OxfordIIITPetDataset_transform(cfg.TRAIN.INPUT_SIZE)]

    elif 'CUBPartsAndAttributes' == dataset_name:
        loader_func = [CubDatasetsPartsAndAttributes]
        input_transform = [CUB_transform(cfg.TRAIN.INPUT_SIZE)]
    elif 'CubDatasetOneAttribute':
        loader_func = [CubDatasetOneAttribute]
        input_transform = [clip.clip._transform(cfg.TRAIN.INPUT_SIZE)]
    else:
        raise NotImplementedError("Can't use dataset {}.".format(dataset_name))

    # generic:
    data_root = [cfg.DATA.ROOT_DIR]
    if '_synthetic' in dataset_name:
        data_root = [cfg.OPENSET.PATH_TO_SYNTHETIC_FOLDER]
        loader_func = [CUBDataset]
    is_train = [True if split == "train" else False]
    # read from text file, remove /n at end:
    with open(cfg.DATA.PATH_TO_CLASSNAMES, 'r') as f:
        list_of_categories = f.readlines()
    list_of_categories = [[fix_classname_special_chars(x.strip()) for x in list_of_categories]]

    # input_transform = None
    # if cfg.AUG.ENABLE and split in ["train"]:
    #     input_transform = build_transform(augmentation_type=cfg.AUG.METHOD, augmentation_probability=cfg.AUG.PROB, input_size=cfg.TRAIN.INPUT_SIZE)

    datasets = []
    for ii, _loader_func in enumerate(loader_func):
        additional_dataset = _loader_func(root=data_root[ii], transform=input_transform[ii], train=is_train[ii],
                                          list_of_categories=list_of_categories[ii],
                                          ignore_indices_to_use=True,
                                          **cfg.DATA
                                          )  # , data_info=info, filenames_list=filenames_list, transform=input_transform)
        datasets.append(additional_dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)

    if dataset is not None:
        print("loading dataset : {}.. number of {} examples is {}".format(dataset_name, split, len(dataset)))
    else:
        print('loading empty dataset.')

    return dataset
