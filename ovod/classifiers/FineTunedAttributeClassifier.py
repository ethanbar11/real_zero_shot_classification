import time
from typing import List

from ovod.classifiers import BaseAttributeClassifier
import torch
from fvcore.common.config import CfgNode
import open_clip


class FineTunedAttributeClassifier(BaseAttributeClassifier):
    def __init__(self, openset_params: CfgNode, device: torch.device, openset_categories: List,
                 model_size: str, model_card_path: str = "/shared-data5/guy/modelzoo/clip/ViT-B-16.pt", ):
        self.model_size = model_size
        super().__init__(openset_params, device, openset_categories, model_card_path)

    def init_base_functions(self, device, model_card_path, *args, **kwargs):
        # check if model_card_path is str:
        # if isinstance(model_card_path, str):
        is_valid = False
        tries = 0
        clip_model, _, preprocess = None, None, None
        while not is_valid:
            try:
                clip_model, _, preprocess = open_clip.create_model_and_transforms(self.model_size,
                                                                                  pretrained=model_card_path,
                                                                                  device=device, *args, **kwargs)
            except:
                is_valid = False
                tries += 1
                if tries > 10:
                    raise Exception(f"Failed to initialize model after 10 tries, from checkpoint {model_card_path}")
                time.sleep(3)
            else:
                is_valid = True
        self.set_clip_params(clip_model, preprocess)

    def set_base_functions(self, clip_model, device):
        self.clip_model = clip_model
        self.device = device
        self.set_clip_params(clip_model, self.preprocess)
