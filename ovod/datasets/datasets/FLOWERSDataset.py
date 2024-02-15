import os
import torch
from torchvision import datasets
from typing import List
#from fvcore.common.config import CfgNode
from scipy.io import loadmat

from PIL import Image
import torchvision.transforms as transforms

_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colts foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]

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


class FLOWERSDataset(torch.utils.data.Dataset):
    """
    # Wrapper for the CUB-200-2011 dataset.
    # Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.
    # Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
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
                 **args
                 ):

        self._base_folder = root
        self._images_folder = os.path.join(root, "jpg")
        self._splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

        #  TODO: WHAT ABOUT VAL SPLIT? I AM RECEIVING HERE A TRAIN BOOLEAN ARG
        if train:
            self._split = "train"
        else:
            self._split = "test"

        set_ids = loadmat(os.path.join(self._base_folder, "setid.mat"), squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(os.path.join(self._base_folder, "imagelabels.mat"), squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(os.path.join(self._images_folder, f"image_{image_id:05d}.jpg"))

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


    def _fit_dataset_to_category_list(self, list_of_categories: List[str] = ["Laysan Albatross", "Sooty Albatross", "Rhinoceros Auklet", "Least Auklet"]):
        # Change list_of_categories to class idx.
        list_of_cat_idx = [_NAMES.index(cat.lower()) for cat in list_of_categories]
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

        return sample, target, image_file, _NAMES[target]


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
