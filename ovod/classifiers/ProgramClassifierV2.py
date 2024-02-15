import os
import torch
import clip
from typing import List
from fvcore.common.config import CfgNode
from torch.nn import functional as F
# from ovod.utils.image_patch import ImagePatch
from .ProgramClassifier import BaseProgramClassifier
import ovod.utils.logger as logging
logger = logging.get_logger(__name__)


class BaseProgramClassifierV2(BaseProgramClassifier):
    """
    The AttributeClassifier model for classification by description.
    """

    def __init__(self, openset_params: CfgNode, device: torch.device, openset_categories: List,
                 model_card_path: str = "/shared-data5/guy/modelzoo/clip/ViT-B-16.pt"):
        super(BaseProgramClassifier, self).__init__(openset_params, device, openset_categories, model_card_path)
        self.set_program_folder(openset_params.PATH_TO_PROGRAMS_FOLDER)

    def init_base_functions(self, device, model_card_path):
        from ovod.utils import vlpart_utils
        from third_party.VLPart.demo.demo import setup_cfg as vlpart_setup_cfg
        from third_party.VLPart.demo.predictor import VisualizationDemo

        vlpart_args = vlpart_utils.DEFAULT_VLPART_ARGS
        vlpart_config = vlpart_setup_cfg(vlpart_args)
        self.vlpart_model = VisualizationDemo(vlpart_config, vlpart_args)

        self.image_patch = ImagePatch(self.device, self.model_card_path, self.vlpart_model)
        self.attributes = {k: None for k in self.openset_categories}
        self.index2class = {i: k for i, k in enumerate(self.attributes.keys())}
        self.class2index = {k: i for i, k in enumerate(self.attributes.keys())}


    def set_program_folder(self, programs_folder):
        self.programs_folder = programs_folder

    def execute_program(self, execute_command_function, image, category=None):
        attributes, weights = execute_command_function(image, self.image_patch)
        total_score = 0
        total_weight = 0
        for name, att_value in attributes.items():
            weight = 1
            if name in weights:
                weight = weights[name]
            total_score += att_value * weight
            total_weight += weight
        if category and not self.attributes[category]:
            self.attributes[category] = {'queries': attributes.keys()}
        total_score /= total_weight
        try:
            total_score = total_score[0]  # If it is not a float, take it.
        except Exception:
            pass
        return total_score, attributes

    def validate_program_is_valid(self, program_as_string, img, verbose=False):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp:
            temp_name = temp.name
            temp.write(program_as_string)

        import sys
        import importlib.util

        # Add the directory containing the temp file to sys.path
        sys.path.append(tempfile.gettempdir())

        # Import the temporary module
        module_name = temp_name.split("/")[-1].replace(".py", "")  # get the filename without the extension
        spec = importlib.util.spec_from_file_location(module_name, temp_name)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.debug(f"Program can not be loaded. Error: {e}")
            return False

        try:
            score, attributes = self.execute_program(module.execute_command, img)
            # assert type(score) == torch.Tensor
            assert type(attributes) == dict
            return True
        except Exception as e:
            logger.debug(f"Program is not valid. Error: {e}")
            if verbose:
                print(program_as_string)
            return False
