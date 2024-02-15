import torch
from fvcore.common.config import CfgNode
from .BasePrompter import BasePrompter
from .ObjectDescriptionImproved import ObjectDescriptionImproved
from .ObjectProgramSelfImprovingV1 import ObjectProgramSelfImprovingV1
from .ObjectProgramSelfImprovingWithActions import ObjectProgramSelfImprovingWithActions
from .OxfordObjectDescriptionPrompter import OxfordObjectDescriptionPrompter, OxfordObjectDescriptionPrompterIN21k
from .ColumbiaObjectDescriptionPrompter import ColumbiaObjectDescriptionPrompter
from .ObjectProgramPrompter import ObjectProgramPrompter


# def build_prompter(cfg: CfgNode, device: torch.device) -> BasePrompter:
def build_prompter(prompter_name: str, base_folder: str = "") -> BasePrompter:
    print(f"Loading prompter '{prompter_name}' ...")
    if prompter_name == 'ox':
        prompter = OxfordObjectDescriptionPrompter(base_folder=[], include_object_name=True)
    elif prompter_name == 'ox_noname':
        prompter = OxfordObjectDescriptionPrompter(base_folder=[], include_object_name=False)
    elif prompter_name == 'ox_noname_imagenet21k':
        prompter = OxfordObjectDescriptionPrompterIN21k(base_folder=[], include_object_name=False)
    elif prompter_name == 'col':
        prompter = ColumbiaObjectDescriptionPrompter(base_folder=[], include_object_name=True)
    elif prompter_name == 'program_base':
        prompter = ObjectProgramPrompter(base_folder=base_folder, include_object_name=False)
    elif prompter_name == 'program_self_improving':
        prompter = ObjectProgramSelfImprovingV1(base_folder=base_folder, include_object_name=False)
    elif prompter_name == 'description_improved':
        prompter = ObjectDescriptionImproved(base_folder=base_folder, include_object_name=False)
    elif prompter_name == 'program_self_improving_with_actions':
        prompter = ObjectProgramSelfImprovingWithActions(base_folder=base_folder, include_object_name=False)
    else:
        raise NotImplementedError("Model name is not supported..")

    return prompter
