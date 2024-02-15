import os
import torch
from torchvision.datasets import Food101
from typing import List
from scipy.io import loadmat
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class Food101Dataset(Food101):
    """
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 train=True,
                 classnames_file=None,
                 **data_config
                 ):
        if train:
            split = "train"
            # reflects the conbined split train
        else:
            split = "test"
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform)
        self._base_folder = root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        classnames_file = data_config.get("PATH_TO_CLASSNAMES")
        with open(classnames_file, "r") as f:
            names = f.read().splitlines()
        self._NAMES = names
        # that is the certain classnames format as in the outside, for consistancy

        self._fit_dataset_to_category_list(list_of_categories=names)

    def _fit_dataset_to_category_list(self, list_of_categories: List[str]):
        # Change list_of_categories to class idx.
        list_of_cat_idx = [self._NAMES.index(cat.lower()) for cat in list_of_categories]
        new_imgs_file, new_labels = [], []
        for img_file, label in zip(self._image_files, self._labels):
            if label in list_of_cat_idx:
                new_labels.append(label)
                new_imgs_file.append(img_file)

        self._image_files = new_imgs_file
        self._labels = new_labels
        print(f"Dataset was fitted to category list [{len(list_of_categories)}]: {list_of_categories}")

    # def __len__(self):
    #     return len(self._labels)

    def __getitem__(self, index):
        # generate one sample
        image_file, target_tensor = self._image_files[index], self._labels[index]
        image_tensor = Image.open(image_file).convert("RGB")

        if self.transform_ is not None:
            image_tensor = self.transform_(image_tensor)
        if self.target_transform_ is not None:
            target_tensor = self.target_transform_(target_tensor)

        return image_tensor, target_tensor, str(image_file), self._NAMES[target_tensor]

    @staticmethod
    def _transform(n_px):
        return transforms.Compose([
            transforms.Resize(n_px, interpolation=Image.BICUBIC),
            transforms.CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            transforms.ToTensor()
        ])


def _transform_normalize():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])