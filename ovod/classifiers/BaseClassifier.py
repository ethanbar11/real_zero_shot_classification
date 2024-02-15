import torch
from typing import List
import torch.nn as nn
from fvcore.common.config import CfgNode

class BaseClassifier(nn.Module):
    """
    The AttributeClassifier model for classification by description.
    """

    def __init__(self, openset_params: CfgNode, device: torch.device, openset_categories: List,
                 model_card_path: str = "/shared-data5/guy/modelzoo/clip/ViT-B-16.pt"):
        super(BaseClassifier, self).__init__()
        self.index2class = {i: k for i, k in enumerate(openset_categories)}
        self.class2index = {k: i for i, k in enumerate(openset_categories)}

    def init_base_functions(self, device, model_card_path):
        # to implement in child class
        raise NotImplementedError

    def compute_attributes_encodings(self, attributes_json_path: str, clip_model, openset_categories: List):
        # to implement in child class
        raise NotImplementedError

    def aggregate_similarity(slef, similarity_matrix_chunk, aggregation_method='mean'):
        # to implement in child class
        raise NotImplementedError

    def forward(self, images: torch.Tensor):  # , labels: torch.Tensor):
        # to implement in child class
        raise NotImplementedError

    def plot_prediction(self, img: str, output: dict, plot_folder: str):
        # to implement in child class
        raise NotImplementedError

