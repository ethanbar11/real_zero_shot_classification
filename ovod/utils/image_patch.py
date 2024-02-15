from __future__ import annotations

from torch.nn import functional as F
import numpy as np
import torch
from PIL import Image
from clip import clip
from torchvision import transforms
from typing import Union

from ovod.utils import vlpart_utils
from third_party.VLPart.demo.demo import setup_cfg as vlpart_setup_cfg
from third_party.VLPart.demo.predictor import VisualizationDemo

# from torchmetrics.multimodal.clip_score import CLIPScore


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    exists_full(parts_list: List[str])->Tuple[Dict[str, bool], Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]
        Returns 3 dictionaries that the keys are the parts to be detected
        exists_results -> Boolean value indicating if part was detected or not
        bbox_results -> A list of b-bboxes [np.array shape=(4)] (several instances of a part are potential to be found), list will be empty if part not detected
        mask_results -> A list of seg. masks [np.array shape=img_shape)] (several instances of a part are potential to be found), list will be empty if part not detected
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(self, device: torch.device, clip_model_card_path, vlpart_model=None):
        clip_model, preprocess = clip.load(clip_model_card_path, device=device, jit=False)  # ViT-B/32
        self.device = device
        clip_model.eval()
        clip_model.requires_grad_(False)  # Notice: clip_model is not trainable
        self.clip_model, self.preprocess = clip_model, preprocess

        if vlpart_model:
            self.vlpart_model = vlpart_model
        else:
            vlpart_args = vlpart_utils.DEFAULT_VLPART_ARGS
            vlpart_config = vlpart_setup_cfg(vlpart_args)
            self.vlpart_model = VisualizationDemo(vlpart_config, vlpart_args)
        self.exists_full_results = None
        self.exists_full_instances = None

    def initialize_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray], left: int = None, lower: int = None,
                         right: int = None, upper: int = None, parent_left=0, parent_lower=0, queues=None,
                         parent_img_patch=None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """


        self.original_image = np.asarray(image) if isinstance(image, Image.Image) else image
        self.exists_full_results = None
        self.exists_full_instances = None

        # Normalize image to continue normal flow as now
        normalization_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        tensor_to_image_transform = transforms.Compose([
            lambda x: Image.fromarray(x.cpu().detach().numpy())
        ])

        if torch.is_tensor(image): # transforms are separated in case 'image' input is a numpy object
            image = tensor_to_image_transform(image)
        image = normalization_transform(image)

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, image.shape[1] - upper:image.shape[1] - lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower
        self.cropped_image = self.cropped_image.to(self.device)
        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")



    def exists(self, object_name) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        if self.vlpart_model:
            self.vlpart_model.set_vocabulary(object_name)
            # Convert image to numpy for inference the vlpart model
            img = self.original_image
            if torch.is_tensor(img):
                img = img.cpu().detach().numpy()
                img = img[:, :, ::-1]
            predictions, vis_output = self.vlpart_model.run_on_image(img, visualize=False)
            return len(predictions["instances"]) > 0
        else:
            return True
        # if object_name.isdigit() or object_name.lower().startswith("number"):
        #     object_name = object_name.lower().replace("number", "").strip()
        #
        #     object_name = w2n.word_to_num(object_name)
        #     answer = self.simple_query("What number is written in the image (in digits)?")
        #     return w2n.word_to_num(answer) == object_name
        #
        # patches = self.find(object_name)
        #
        # filtered_patches = []
        # for patch in patches:
        #     if "yes" in patch.simple_query(f"Is this a {object_name}?"):
        #         filtered_patches.append(patch)
        # return len(filtered_patches) > 0

    def exists_full(self, parts_list):
        """
        Input: a list of parts to detect.
        Output: 3 dictionaries that the keys are the parts to be detected
        exists_results -> Boolean value indicating if part was detected or not
        bbox_results -> A list of b-bboxes [np.array shape=(4)] (several instances of a part are potential to be found), list will be empty if part not detected
        mask_results -> A list of seg. masks [np.array shape=img_shape)] (several instances of a part are potential to be found), list will be empty if part not detected
        """
        exists_results = dict.fromkeys(parts_list, False)
        bbox_results = dict.fromkeys(parts_list, [])
        mask_results = dict.fromkeys(parts_list, [])

        if self.vlpart_model:
            self.vlpart_model.set_vocabulary(parts_list)
            # Convert image to numpy for inference the vlpart model
            img = self.original_image
            if torch.is_tensor(img):
                img = img.cpu().detach().numpy()
                img = img[:, :, ::-1]
            predictions, vis_output = self.vlpart_model.run_on_image(img, visualize=False)
            if len(predictions) > 0:  # We have found something
                boxes = predictions['instances'].pred_boxes.tensor.cpu().detach().detach().numpy()
                masks = predictions['instances'].pred_masks.cpu().detach().detach().numpy()
                classes_idxs = predictions['instances'].pred_classes.cpu().detach().detach().numpy()
                classes_names = np.asarray(parts_list)[classes_idxs]
                for cls, box, mask in zip(classes_names, boxes, masks):
                    exists_results[cls] = True
                    bbox_results[cls].append(box)
                    mask_results[cls].append(mask)

            self.exists_full_results = {
                'exists_results': exists_results,
                'bbox_results': bbox_results,
                'mask_results': mask_results
            }
            self.exists_full_instances = predictions if len(predictions['instances']) > 0 else None

        return exists_results, bbox_results, mask_results


    def clip_similarity(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        name = f"{attribute} {object_name}"
        self.device = self.cropped_image.device
        return self.__inner_clip_similarity(self.cropped_image.unsqueeze(0), name)

    def clip(self, object_name: str, attribute: str) -> bool:
        return self.clip_similarity(object_name, attribute)

    def __inner_clip_similarity(self, images, text):
        image_encodings = self.clip_model.encode_image(images).float()
        image_encodings = F.normalize(image_encodings)
        text_tokens = clip.tokenize(text)

        text_features = F.normalize(
            self.clip_model.encode_text(text_tokens.to(self.device))).float()  # perpare features in float-32bit
        cos_sim = image_encodings @ text_features.T
        return cos_sim[0]

    def add_original_image(self, original_image):
        self.original_image = original_image