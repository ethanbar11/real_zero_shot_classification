import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, List
from fvcore.common.config import CfgNode
import albumentations as A
import cv2

class BasicDataset(Dataset):
    """
    Basic Dataset for Imagenet training
    """
    def __init__(self, cfg: CfgNode, data_root: str = None, data_info: Dict = None, filenames_list: str = None, transform: A.core.composition.Compose = None):
    #def __init__(self, data_root, patch_size=256):
        super(BasicDataset, self).__init__()
        self.cfg = cfg
        self.data_root = data_root
        self.data_info = data_info
        self.semantics_file = cfg.DATA.PATH_TO_SEMANTICS_FILE
        self.input_size = cfg.TRAIN.INPUT_SIZE
        self.seenclasses = self.data_info['seenclasses']
        self.unseenclasses = self.data_info['unseenclasses']

        self.data_items_list = filenames_list
        print('fetch {} samples for training'.format(len(self.data_items_list)))

        self.semantics = None
        with open(self.semantics_file, 'rb') as f:
            self.semantics = pickle.load(f)

        # np.ndarray to torch.tensor
        self.basic_transformer = transforms.Compose([transforms.ToTensor()])
        self.fig = None

    def get_img_and_meta(self, index: int) -> Dict:
        """
        Load and parse a single image.
        Args:
            index (int): Index
        Returns:
            img (ndarray): RGB image in required input_size
            img_path (string): full path to image file in image format (PNG or equivalent)
            ref_path (string): full path to reference image file, if available (PNG or equivalent)
        """
        # get paths:
        filename = self.data_items_list[index]
        img_path = os.path.join(self.data_root, self.input_image_folder, filename)
        # ref_path = os.path.join(self.data_root, self.ref_image_folder, filename)
        #
        # img, ratio_margin_input = ultrasound_img_load(img_path)
        # # optional:
        # ref, ratio_margin_ref = ultrasound_img_load(ref_path)
        #
        # #assert ratio_margin_ref != ratio_margin_input, "ratio_ref {} is not equal to ratio_input {}".format(ratio_margin_ref, ratio_margin_input)
        # assert img.shape == ref.shape, "input img {}x{} is not equal to ref image {}x{} for file {}".format(
        #     img.shape[0], img.shape[1], ref.shape[0], ref.shape[1], filename)
        #
        # img, ratio_size_input = self.resize_img(img, required_size=self.input_size)
        # # random crop: (in original n2n code)
        # # img = np.array(img, dtype=np.float32)
        # #img = self.get_random_crop(im=img, patch_size=self.input_size)
        # ref, ratio_size_ref = self.resize_img(ref, required_size=self.input_size)
        #
        # img = np.array(img, dtype=np.float32)
        # ref = np.array(ref, dtype=np.float32)
        #
        # data = {"img": img,
        #         "img_path": img_path,
        #         "ratio": ratio_margin_input,
        #         "ref_path": ref_path,
        #         "ref": ref,
        #         "filename": filename,
        #         }
        # return data

    def get_random_crop(self, im: np.ndarray, patch_size: int) ->np.ndarray:
        """ performs random crop """
        H = im.shape[0]
        W = im.shape[1]
        if H - patch_size > 0:
            xx = np.random.randint(0, H - patch_size)
            im = im[xx:xx + patch_size, :, :]
        if W - patch_size > 0:
            yy = np.random.randint(0, W - patch_size)
            im = im[:, yy:yy + patch_size, :]
        return im

    def resize_img(self, img, required_size):
        height, width, dim = img.shape

        ratio = [required_size / float(width), required_size / float(height)]
        if height != required_size or width != required_size:
            img = cv2.resize(img, dsize=(required_size, required_size),
                             interpolation=cv2.INTER_LINEAR_EXACT)

        return img, ratio

    def denormalized_img(self, img: np.ndarray) -> np.ndarray:
        """ Convert to channel last numpy format, at rage [0, 255]"""
        return 255 * np.transpose(img, (1, 2, 0))    # H*W*C

    def __getitem__(self, index):
        data = self.get_img_and_meta(index)
        img = self.basic_transformer(data["img"] / 255.0)
        ref = self.basic_transformer(data["ref"] / 255.0)
        return img, ref, data["filename"]

    def __len__(self):
        return len(self.data_items_list)

    def plot_item(self, index: int,  do_augmentation: bool = True, plot_directory: str = './visu/') -> None:
        """ Plot frame and gt annotations for a single data point """
        data = self.get_img_and_meta(index)
        basename = os.path.splitext(data["img_path"].replace(self.data_root, ""))[0].replace("/", "_")
        input_img = data["img"].astype(int)
        ref_img = data["ref"].astype(int)
        #ref, _ = ultrasound_img_load(data["ref_path"])
        # metrics:
        if self.cfg.SHOWDATA.SHOW_METRIC:
            out_ref_metrics = {"psnr": metrics.calculate_psnr(input_img, ref_img), "ssim": metrics.calculate_ssim(input_img, ref_img)}
        # plot:
        if self.fig == None:
            self.fig = plt.figure(figsize=(16, 10))
        self.fig.clf()
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax1.imshow(input_img)
        title = "{}, Input (size: {})".format(data["filename"], input_img.shape[:2])
        ax1.set_title(title)
        ax1.axis('off')
        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.imshow(ref_img)
        title = "{}, Output (size: {})".format(data["filename"], ref_img.shape[:2])
        if self.cfg.SHOWDATA.SHOW_METRIC:
            additional_text = "PSNR: {:.2f}, SSIM: {:.2f}".format(out_ref_metrics["psnr"], out_ref_metrics["ssim"])
            title = "{} {}".format(title, additional_text)
        ax2.set_title(title)
        ax2.axis('off')
        # save
        fname = os.path.join(plot_directory, "{}_INDEX_{}.jpg".format(basename, index))
        self.fig.savefig(fname=fname)
        print("Plot is saved to {}".format(fname))






"""
import os
import pickle
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image


class BaseDataset(Dataset):
    def __init__(self, dataset_path, image_files, labels, transform=None):
        super(BaseDataset, self).__init__()
        self.dataset_path = dataset_path
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_file = self.image_files[idx]
        image_file = os.path.join(self.dataset_path, image_file)
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class UNIDataloader():
    def __init__(self, config):
        self.config = config
        with open(config.pkl_path, 'rb') as f:
            self.info = pickle.load(f)

        self.seenclasses = self.info['seenclasses'].to(config.device)
        self.unseenclasses = self.info['unseenclasses'].to(config.device)

        (self.train_set,
         self.test_seen_set,
         self.test_unseen_set) = self.torch_dataset()

        self.train_loader = DataLoader(self.train_set,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=config.num_workers)
        self.test_seen_loader = DataLoader(self.test_seen_set,
                                           batch_size=config.batch_size,
                                           shuffle=False,
                                           num_workers=config.num_workers)
        self.test_unseen_loader = DataLoader(self.test_unseen_set,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers)

    def torch_dataset(self):
        data_transforms = transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.CenterCrop(self.config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        baseset = BaseDataset(self.config.dataset_path,
                              self.info['image_files'],
                              self.info['labels'],
                              data_transforms)

        train_set = Subset(baseset, self.info['trainval_loc'])
        test_seen_set = Subset(baseset, self.info['test_seen_loc'])
        test_unseen_set = Subset(baseset, self.info['test_unseen_loc'])

        return train_set, test_seen_set, test_unseen_set

"""