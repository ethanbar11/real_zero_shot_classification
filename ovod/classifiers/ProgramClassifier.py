import os
import torch
import clip
from typing import List
import torch.nn as nn
from fvcore.common.config import CfgNode
import json
from torch.nn import functional as F
from collections import OrderedDict
import importlib
import inspect
from PIL import Image

from .AttributeClassifier import BaseAttributeClassifier
from ovod.utils.vis_programs import plot_program
from ovod.utils.misc import flatten_dict, dict_to_numpy, load_module


class BaseProgramClassifier(BaseAttributeClassifier):
    """
    The AttributeClassifier model for classification by description.
    """

    def __init__(self, openset_params: CfgNode, device: torch.device, openset_categories: List,
                 model_card_path: str = "/shared-data5/guy/modelzoo/clip/ViT-B-16.pt"):
        super(BaseProgramClassifier, self).__init__(openset_params, device, openset_categories, model_card_path)
        self.set_program_folder(openset_params.PATH_TO_PROGRAMS_FOLDER)

    def __str__(self):
        name = self.__class__.__name__
        return f"{name}{{programs folder: {self.programs_folder}}}"

    def clip_similarity(self, images, text):
        image_encodings = self.clip_model.encode_image(images).float()
        image_encodings = F.normalize(image_encodings)
        text_tokens = clip.tokenize(text)
        text_features = F.normalize(
            self.clip_model.encode_text(text_tokens.to(self.device))).float()  # perpare features in float-32bit
        cos_sim = image_encodings @ text_features.T
        return cos_sim

    def set_program_folder(self, programs_folder):
        self.programs_folder = programs_folder

    def execute_program(self, execute_command_function, image, category=None):
        return execute_command_function(image, self.clip_similarity)

    def forward(self, images: torch.Tensor):  # , labels: torch.Tensor):

        image_description_similarity_cumulative = [None] * self.num_classes
        explanation_cumulative = [None] * self.num_classes
        for i, (category, category_attributes) in enumerate(self.attributes.items()):
            program_path = os.path.join(self.programs_folder, category.replace(" ", "_").lower())
            execute_command = load_module(f"{program_path}.py").execute_command
            #execute_command = importlib.import_module(program_path.replace("/", ".")).execute_command
            scores, explanations = [], []
            for image in images:
                # score, explanation = execute_command(image, self.clip_similarity)
                score, explanation = self.execute_program(execute_command, image, category)
                scores.append(score)
                # explanations.append(dict_to_numpy(flatten_dict(explanation)))
                explanations.append(flatten_dict(nested_dict=explanation, sep="-> "))
            image_description_similarity_cumulative[i] = torch.stack(scores)
            explanation_cumulative[i] = explanations

        # create tensor of similarity means
        cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)
        # explanations = torch.stack(explanation_cumulative, dim=1)

        # transpose list of lists:
        explanation_cumulative = list(map(list, zip(*explanation_cumulative)))

        return cumulative_tensor, explanation_cumulative  # itm_clip_score, itm_attributes_score

    def plot_prediction(self, img: str, output: dict, plot_folder: str):

        category = output['label_pred']
        program_path = os.path.join(self.programs_folder, category.replace(" ", "_").lower())
        execute_command = importlib.import_module(program_path.replace("/", ".")).execute_command
        program = inspect.getsource(execute_command)
        program_output_dict = output['explanation_pred']
        program_name = f"Program for {category}"
        program_score = output['desc_predictions'][category]
        image = Image.open(img)
        output_file = os.path.join(plot_folder, f"PRED_for_{os.path.basename(img)}")
        plot_program(program, program_output_dict, program_name, program_score, image, output_file,
                     with_bounding_box=False)

        # plot gt category:
        category = output['label_gt']
        program_path = os.path.join(self.programs_folder, category.replace(" ", "_").lower())
        execute_command = importlib.import_module(program_path.replace("/", ".")).execute_command
        program = inspect.getsource(execute_command)
        program_output_dict = output['explanation_gt']
        program_name = f"Program for {category}"
        program_score = output['desc_predictions'][category]
        image = Image.open(img)
        output_file = os.path.join(plot_folder, f"GT_for_{os.path.basename(img)}")
        plot_program(program, program_output_dict, program_name, program_score, image, output_file,
                     with_bounding_box=False)

# class BaseProgramClassifier(nn.Module):
#     """
#     The AttributeClassifier model for classification by description.
#     """
#     def __init__(self, openset_params: CfgNode, device: torch.device, openset_categories: List, model_card_path: str = "/shared-data5/guy/modelzoo/clip/ViT-B-16.pt"):
#         super(BaseProgramClassifier, self).__init__()
#         self.openset_categories = openset_categories
#         self.device = device
#         # # load clip model
#         self.model_card_path = model_card_path
#         clip_model, preprocess = clip.load(model_card_path, device=device, jit=False)   # ViT-B/32
#         clip_model.eval()
#         clip_model.requires_grad_(False)    # Notice: clip_model is not trainable
#         self.clip_model, self.preprocess = clip_model, preprocess
#         # # load attributes
#         self.attributes = self.compute_attributes_encodings(attributes_json_path=openset_params.ATTRIBUTES_FILE,
#                                                   clip_model=self.clip_model, openset_categories=openset_categories)
#
#         for i, (category, category_attributes) in enumerate(self.attributes.items()):
#             category_attributes["text_features"] = category_attributes["text_features"].to(self.device)
#         self.index2class = {i: k for i, k in enumerate(self.attributes.keys())}
#         self.class2index = {k: i for i, k in enumerate(self.attributes.keys())}
#         self.num_classes = len(self.attributes)
#         #
#         self.programs_folder = openset_params.PATH_TO_PROGRAMS_FOLDER
#
#
#     def compute_attributes_encodings(self, attributes_json_path: str, clip_model, openset_categories: List):
#
#         with open(attributes_json_path) as f:
#             attributes_json = json.load(f)
#         all_classnames = attributes_json.keys()
#         classnames = list(set(all_classnames).intersection(set(openset_categories)))
#         attributes = OrderedDict()
#         print(f"preparing attributes to {len(classnames)} open-set classes..")
#         for category in classnames:
#             queries = attributes_json[category] #[att for key, att in attributes_json[category].items()]
#             text_tokens = clip.tokenize(queries)
#             text_features = F.normalize(clip_model.encode_text(text_tokens.to(self.device))).float()     # perpare features in float-32bit
#             text_tokens = text_tokens.detach().cpu()
#             text_features = text_features.detach().cpu()    # detach from gpu
#             class_dict = {"queries": queries, "text_tokens": text_tokens, "text_features": text_features}
#             attributes[category] = class_dict
#
#         return attributes
#
#     def aggregate_similarity(slef, similarity_matrix_chunk, aggregation_method='mean'):
#         if aggregation_method == 'max':
#             return similarity_matrix_chunk.max(dim=1)[0]
#         elif aggregation_method == 'sum':
#             return similarity_matrix_chunk.sum(dim=1)
#         elif aggregation_method == 'mean':
#             return similarity_matrix_chunk.mean(dim=1)
#         else:
#             raise ValueError("Unknown aggregate_similarity")
#
#     def clip_similarity(self, images, text):
#         image_encodings = self.clip_model.encode_image(images).float()
#         image_encodings = F.normalize(image_encodings)
#         text_tokens = clip.tokenize(text)
#         text_features = F.normalize(self.clip_model.encode_text(text_tokens.to(self.device))).float()     # perpare features in float-32bit
#         cos_sim = image_encodings @ text_features.T
#         return cos_sim
#
#     def forward(self, images: torch.Tensor): #, labels: torch.Tensor):
#
#         image_encodings = self.clip_model.encode_image(images).float()
#         image_encodings = F.normalize(image_encodings)
#         outcum = []
#         outsim = []
#         for idx in range(images.shape[0]):
#             image = images[idx]
#             image_description_similarity = [torch.zeros(1)] * self.num_classes
#             image_description_similarity_cumulative = [None] * self.num_classes
#             for i, category in enumerate(['Laysan Albatross', 'Heermann Gull']):
#                 # import execute_command for category:
#                 program_path = os.path.join(self.programs_folder, category.replace(" ", "_").lower())
#                 execute_command = importlib.import_module(program_path.replace("/", ".")).execute_command
#                 image_description_similarity_cumulative[i] = execute_command(image, self.clip_similarity) # self.aggregate_similarity(image_description_similarity[i])
#             print(image_description_similarity_cumulative)
#             exit()
#             outcum.append(torch.Tensor(image_description_similarity_cumulative))
#             outsim.append(torch.Tensor(image_description_similarity))
#
#         # create tensor of similarity means
#         cumulative_tensor = torch.stack(outcum, dim=1)
#         explanations = torch.stack(outsim, dim=1)
#
#         # descr_predictions = cumulative_tensor.argmax(dim=1)
#
#         # itm_score = image_encodings @ text_features.T
#         # itm_score = self.attributes_image_matching(image_features, text_features)
#
#         # TODO: label
#
#         return cumulative_tensor, explanations #itm_clip_score, itm_attributes_score
#
#
