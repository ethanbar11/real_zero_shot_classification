import os
import torch
import numpy as np
import clip
from typing import List
import torch.nn as nn
from fvcore.common.config import CfgNode
import json
from torch.nn import functional as F
from collections import OrderedDict

from .BaseClassifier import BaseClassifier
from ovod.utils.vis import plot_attributes_predictions


class BaseAttributeClassifier(BaseClassifier):
    """
    The AttributeClassifier model for classification by description.
    """

    def __init__(self, openset_params: CfgNode, device: torch.device, openset_categories: List,
                 model_card_path: str = "/shared-data5/guy/modelzoo/clip/ViT-B-16.pt"):
        super(BaseAttributeClassifier, self).__init__(openset_params, device, openset_categories, model_card_path)
        #super(BaseAttributeClassifier, self).__init__()
        self.openset_categories = openset_categories
        self.openset_params = openset_params
        self.device = device
        # load clip model
        self.model_card_path = model_card_path
        self.init_base_functions(device, model_card_path)
        self.num_classes = len(openset_categories)
        # load attributes

    def __str__(self):
        name = self.__class__.__name__
        return f"{name}{{}}"

    def init_base_functions(self, device, model_card_path):
        clip_model, preprocess = clip.load(model_card_path, device=device, jit=False)  # ViT-B/32
        self.set_clip_params(clip_model, preprocess)

    def set_clip_params(self, clip_model, preprocess):
        clip_model.eval()
        #clip_model.requires_grad_(False)  # Notice: clip_model is not trainable
        self.clip_model, self.preprocess = clip_model, preprocess
        self.attributes = self.compute_attributes_encodings(attributes_json_path=self.openset_params.ATTRIBUTES_FILE,
                                                            clip_model=self.clip_model,
                                                            openset_categories=self.openset_categories)
        for i, (category, category_attributes) in enumerate(self.attributes.items()):
            category_attributes["text_features"] = category_attributes["text_features"].to(self.device)
        self.index2class = {i: k for i, k in enumerate(self.attributes.keys())}
        self.class2index = {k: i for i, k in enumerate(self.attributes.keys())}
        self.num_classes = len(self.attributes)

    def compute_attributes_encodings(self, attributes_json_path: str, clip_model, openset_categories: List):

        with open(attributes_json_path) as f:
            attributes_json = json.load(f)
        # all_classnames = attributes_json.keys()
        classnames = openset_categories # list(set(all_classnames).intersection(set(openset_categories)))
        attributes = OrderedDict()
        # new_attributes_without_lines = {}
        # for name, atts in attributes_json.items():
        #     new_attributes_without_lines[name.replace('-',' ')] = atts
        # attributes_json = new_attributes_without_lines
        print(f"preparing attributes to {len(classnames)} open-set classes..")
        for category in classnames:
            queries = attributes_json[category]  # [att for key, att in attributes_json[category].items()]
            text_tokens = clip.tokenize(queries)
            text_features = F.normalize(
                clip_model.encode_text(text_tokens.to(self.device))).float()  # perpare features in float-32bit
            text_tokens = text_tokens.detach().cpu()
            text_features = text_features.detach().cpu()  # detach from gpu
            class_dict = {"queries": queries, "text_tokens": text_tokens, "text_features": text_features}
            attributes[category] = class_dict

        return attributes

    # def compute_label_encodings(self, model):
    #     label_encodings = F.normalize(model.encode_text(clip.tokenize(
    #         [hparams['label_before_text'] + wordify(l) + hparams['label_after_text'] for l in label_to_classname]).to(self.device)))
    #     return label_encodings

    # def attributes_image_matching(self, image_features: np.ndarray, text_features: np.ndarray):
    #     """ XXX """
    #     with torch.no_grad():
    #         image_features /= image_features.norm(dim=-1, keepdim=True)
    #         text_features /= text_features.norm(dim=-1, keepdim=True)
    #         similarity = (image_features @ text_features.T)  # .softmax(dim=-1)
    #
    #     return similarity

    def aggregate_similarity(slef, similarity_matrix_chunk, aggregation_method='mean'):
        if aggregation_method == 'max':
            return similarity_matrix_chunk.max(dim=1)[0]
        elif aggregation_method == 'sum':
            return similarity_matrix_chunk.sum(dim=1)
        elif aggregation_method == 'mean':
            return similarity_matrix_chunk.mean(dim=1)
        else:
            raise ValueError("Unknown aggregate_similarity")

    def forward(self, images: torch.Tensor,paths=None):  # , labels: torch.Tensor):

        image_encodings = self.clip_model.encode_image(images).float()
        image_encodings = F.normalize(image_encodings)

        # for category, attributes_features in self.attributes.items():
        #     # compute cosine similarity:
        #     cosine_sim_array = self.attributes_image_matching(image_encodings, attributes_features["text_features"])    #.cpu().numpy()

        image_description_similarity = [None] * self.num_classes
        image_description_similarity_cumulative = [None] * self.num_classes

        for i, (category, category_attributes) in enumerate(self.attributes.items()):
            attributes_features = category_attributes["text_features"]
            dot_product_matrix = image_encodings @ attributes_features.T

            image_description_similarity[i] = dot_product_matrix
            image_description_similarity_cumulative[i] = self.aggregate_similarity(image_description_similarity[i])

        # create tensor of similarity means
        cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)
        # stack the list 2D tensors in image_description_similarity and pad zero if lists are not equal in length
        # find max len for all tensor rows:
        max_len = max([x.shape[1] for x in image_description_similarity])
        # pad each torch tensor with zeros to max_len, make sure same device is kept, notice 2D tensor:
        image_description_similarity_padded = [
            torch.cat((x, torch.zeros((x.shape[0], max_len - x.shape[1])).to(self.device)), dim=1) for x in
            image_description_similarity]
        explanations = torch.stack(image_description_similarity_padded, dim=1)

        # descr_predictions = cumulative_tensor.argmax(dim=1)

        # itm_score = image_encodings @ text_features.T
        # itm_score = self.attributes_image_matching(image_features, text_features)

        return cumulative_tensor, explanations  # itm_clip_score, itm_attributes_score

    def plot_prediction(self, img: str, output: dict, plot_folder: str):
        return plot_attributes_predictions(img, output, plot_folder)

        # def plot_program(program: str, program_output_dict: dict, program_name: str, program_score: float, image: Image,
        #          output_file: str, with_bounding_box=False):
