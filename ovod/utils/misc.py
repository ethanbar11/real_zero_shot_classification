import os
import numpy as np
import glob
from PIL import Image
import torch
import cv2
from typing import Tuple
import errno
import datetime
import importlib.util

def to_numpy(some_tensor: torch.Tensor) -> np.ndarray:
    """ Converts torch Tensor to numpy array """
    if torch.is_tensor(some_tensor):
        return some_tensor.detach().cpu().numpy()
    elif type(some_tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(some_tensor)))
    return some_tensor

def dict_to_numpy(some_dict: dict) -> dict:
    """ Converts all torch Tensors in a dict to numpy arrays """
    for key, value in some_dict.items():
        if torch.is_tensor(value):
            some_dict[key] = to_numpy(value)
            # check if value is scalar:
            if value.numel() == 1:
                some_dict[key] = some_dict[key].item()
    return some_dict

def checkpoint(net, epoch, name, opt):
    """ from https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/main/train.py """
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))

def basic_image_loader(dataset_dir):
    """ from https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/main/train.py """
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def get_run_id(logs_dir: str) -> str:
    ''' provides ID for experiment based on time-stamp or index '''
    ct = datetime.datetime.now()
    run_id = "{}{:02}{:02}{:02}{:02}".format(ct.month, ct.day, ct.hour, ct.minute, ct.second)
    #run_id = "{}{}{}{}{}{}".format(ct.year - 2000, ct.month, ct.day, ct.hour, ct.minute, ct.second)
    return run_id

def flatten_dict(nested_dict, parent_key='', sep='_'):
  """Flattens a nested dictionary into a single dictionary.

  Args:
    nested_dict: The nested dictionary to flatten.
    parent_key: The parent key to use for the flattened dictionary.
    sep: The separator to use between the parent key and the child key.

  Returns:
    A flattened dictionary.
  """

  flattened_dict = {}
  for key, value in nested_dict.items():
    new_key = parent_key + sep + key if parent_key else key
    if isinstance(value, dict):
      flattened_dict.update(flatten_dict(value, new_key, sep=sep))
    else:
      flattened_dict[new_key] = value
  return flattened_dict

# def load_module(module_file_path: str) -> object:
#     """ Loads a module from a .py file (also absolute path) """
#     # Load the module
#     spec = importlib.util.spec_from_file_location("module_name", module_file_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#
#     return module

import importlib.util
import os

def load_module(module_file_path: str) -> object:
    """ Loads a module from a .py file (handles both absolute and relative paths) """

    # Convert relative path to absolute path
    if not os.path.isabs(module_file_path):
        # Get the absolute path relative to the current working directory
        module_file_path = os.path.abspath(module_file_path)

    # Check if the file exists
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(f"No module found at {module_file_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("module_name", module_file_path)
    if spec is None or not hasattr(spec, 'loader'):
        raise ImportError(f"Could not load module from {module_file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module
