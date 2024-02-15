import os
import torch
from typing import List
from scipy.io import loadmat
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ovod.utils.class_name_fix import fix_classname_special_chars



class Dogs120Dataset(torch.utils.data.Dataset):
    """
    # Wrapper for the CUB-200-2011 dataset.
    # Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.
    # Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 train=True,
                 bboxes=False,
                 list_of_categories=[],
                 APPLY_NORMALIZE_TRANSFORM=True,
                 **args
                 ):
        self.apply_norm_img_transform = APPLY_NORMALIZE_TRANSFORM

        self._base_folder = root
        self.train = train

        self.classnames = list_of_categories
        self.class_name_to_idx = {name: idx for idx, name in enumerate(self.classnames)}
        self.idx_to_class_name = {idx: name for idx, name in enumerate(self.classnames)}

        self._split = "train" if train else "test"

        annotations = loadmat(os.path.join(self._base_folder, f"{self._split}_list.mat"), squeeze_me=True)
        fnames = annotations['annotation_list']
        self._image_files = [os.path.join(self._base_folder, 'Images', f'{x}.jpg') for x in fnames]
        self._labels = annotations['labels'] - 1  # Fix class range from 1-196 to 0-195
        self.transform_ = transform
        self.target_transform_ = target_transform

        if bboxes:
            # TODO: NOTE, BBOX NOT SUPPORTED ON CARS196 DATASET, NEED TO FIX THIS CODE, BASE CODE FROM CUB DATASET.
            raise NotImplementedError("BBOX NOT SUPPORTED ON CARS-196 DATASET, NEED TO FIX THIS CODE.")
        else:
            self.bboxes = None

        self._fit_dataset_to_category_list(list_of_categories=list_of_categories)

    def _fit_dataset_to_category_list(self, list_of_categories: List[str] = ["Laysan Albatross", "Sooty Albatross",
                                                                             "Rhinoceros Auklet", "Least Auklet"]):
        # Change list_of_categories to class idx.
        # list_of_cat_idx = [self.class_name_to_idx[fix_classname_special_chars(cat)] for cat in list_of_categories]
        list_of_cat_idx = [self.class_name_to_idx[cat] for cat in list_of_categories]
        new_imgs_file, new_labels = [], []
        for img_file, label in zip(self._image_files, self._labels):
            if label in list_of_cat_idx:
                new_labels.append(label)
                new_imgs_file.append(img_file)

        self._image_files = new_imgs_file
        self._labels = new_labels
        print(f"Dataset was fitted to category list [{len(list_of_categories)}]: {list_of_categories}")

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        # generate one sample
        image_file, target = self._image_files[index], self._labels[index]
        sample = Image.open(image_file).convert("RGB")

        if self.bboxes is not None:
            # TODO: NOTE, BBOX NOT SUPPORTED ON FLOWERS DATASET, NEED TO FIX THIS CODE. BASE CODE FROM CUB DATASET.
            raise NotImplementedError("BBOX NOT SUPPORTED ON FLOWERS DATASET, NEED TO FIX THIS CODE.")

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.apply_norm_img_transform:
            sample = _transform_normalize()(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        if isinstance(sample, Image.Image):  # Check if the sample is a Pillow image
            sample = np.array(sample)  # Convert Pillow image to NumPy array

        if sample is None:
            raise Error(image_file)

        return sample, target, image_file, self.classnames[target]


def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
    ])


def _transform_normalize():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
