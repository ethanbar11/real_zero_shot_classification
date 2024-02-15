# Code based on: https://pytorch.org/vision/main/_modules/torchvision/datasets/oxford_iiit_pet.html

import os
import torch
from torchvision import datasets
from typing import List
#from fvcore.common.config import CfgNode
from scipy.io import loadmat

from PIL import Image
import torchvision.transforms as transforms

from ovod.utils.class_name_fix import fix_classname_special_chars


class OxfordIIITPetDataset(torch.utils.data.Dataset):
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
        self.apply_norm_img_transform = APPLY_NORMALIZE_TRANSFORM

        self._base_folder = root
        self._target_types = ["category"]
        self._images_folder = os.path.join(root, "images")
        self._anns_folder = os.path.join(root, "annotations")
        self._segs_folder = os.path.join(root, "trimaps")

        if train:
            self._split = "trainval"
        else:
            self._split = "test"

        image_ids = []
        self._labels = []
        with open(os.path.join(self._anns_folder, f"{self._split}.txt")) as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        # self.classes = [
        #     " ".join(part.title() for part in raw_cls.split("_"))
        #     for raw_cls, _ in sorted(
        #         {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
        #         key=lambda image_id_and_label: image_id_and_label[1],
        #     )
        # ]
        # self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        # self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self._image_files = [os.path.join(self._images_folder, f"{image_id}.jpg") for image_id in image_ids]
        self._segs = [os.path.join(self._segs_folder, f"{image_id}.png") for image_id in image_ids]

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train

        if bboxes:
            # TODO: NOTE, BBOX NOT SUPPORTED ON FLOWERS DATASET, NEED TO FIX THIS CODE, BASE CODE FROM CUB DATASET.
            raise NotImplementedError("BBOX NOT SUPPORTED ON FLOWERS DATASET, NEED TO FIX THIS CODE.")
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in image_ids:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

        self._fit_dataset_to_category_list(list_of_categories=list_of_categories)
        self.list_of_categories = list_of_categories


    def _fit_dataset_to_category_list(self, list_of_categories: List[str] = ["Laysan Albatross", "Sooty Albatross", "Rhinoceros Auklet", "Least Auklet"]):
        # Change list_of_categories to class idx.
        new_imgs_file, new_labels = [], []
        for img_file, label in zip(self._image_files, self._labels):
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

        if isinstance(sample, Image.Image):  # Check if the sample is a Pillow image
            sample = np.array(sample)  # Convert Pillow image to NumPy array

        return sample, target, image_file, self.list_of_categories[target]


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
