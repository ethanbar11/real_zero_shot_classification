import os
import torch
from torchvision import datasets
from typing import List
import numpy as np
# from fvcore.common.config import CfgNode

from PIL import Image
import torchvision.transforms as transforms

from ovod.utils.class_name_fix import fix_classname_special_chars


def removeduplicate(data):
    countdict = {}
    for element in data:
        if element in countdict.keys():

            # increasing the count if the key(or element)
            # is already in the dictionary
            countdict[element] += 1
        else:
            # inserting the element as key  with count = 1
            countdict[element] = 1
    data.clear()
    for key in countdict.keys():
        data.append(key)


class CUBDataset(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False,
                 list_of_categories=[],
                 ignore_indices_to_use=False,
                 APPLY_NORMALIZE_TRANSFORM=True,
                 **args):

        img_root = os.path.join(root, 'images')
        #        loader = datasets.folder.default_loader
        # bboxes = False
        # is_valid_file = None
        # train = True
        # transform = None
        # target_transform = None

        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.redefine_class_to_idx()

        # inverse to self.class_to_idx:
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        self.apply_norm_img_transform = APPLY_NORMALIZE_TRANSFORM

        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if use_train == 'is_train':
                    continue
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))
        self.indices_to_use = indices_to_use
        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = []
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use or ignore_indices_to_use:
                    filenames_to_use.append(fn)
        removeduplicate(filenames_to_use)
        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

        # TODO: add option to fit dataset to a list of categories
        self._fit_dataset_to_category_list(list_of_categories=list_of_categories)

    def _fit_dataset_to_category_list(self, list_of_categories: List[str] = ["Laysan Albatross", "Sooty Albatross",
                                                                             "Rhinoceros Auklet", "Least Auklet"]):
        # list_of_categories = ["Laysan Albatross", "Sooty Albatross", "Rhinoceros Auklet", "Least Auklet"]
        new_samples = []
        new_imgs = []
        for sample, img in zip(self.samples, self.imgs):
            if self.idx_to_class[sample[1]] in list_of_categories:
                new_samples.append(sample)
                new_imgs.append(img)
        self.samples = new_samples
        self.imgs = new_imgs
        self.class_to_idx = {k: v for k, v in self.class_to_idx.items() if k in list_of_categories}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(f"Dataset was fitted to category list [{len(list_of_categories)}]: {list_of_categories}")

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(CUBDataset, self).__getitem__(index)
        if self.bboxes is not None:
            # squeeze coordinates of the bounding box to range [0, 1]
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.apply_norm_img_transform:
            sample = _transform_normalize()(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        filename = self.imgs[index]

        if isinstance(sample, Image.Image):  # Check if the sample is a Pillow image
            sample = np.array(sample)  # Convert Pillow image to NumPy array

        return sample, target, filename[0], self.idx_to_class[filename[1]]

    def get_files_from_category(self, category):
        pass

    def redefine_class_to_idx(self):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = fix_classname_special_chars(k)
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict


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
