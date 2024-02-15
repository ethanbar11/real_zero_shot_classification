import os
import sys
import argparse
import json
import openai
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ovod.utils.parser import parse_args, load_config
import ovod.utils.logger as logging
from ovod.config.defaults import assert_and_infer_cfg
from ovod.classifiers.build import build_classifier
from ovod.llm_wrappers.build import build_llm_wrapper
from ovod.datasets.build import build_dataset
from ovod.utils.parser import load_config
from ovod.prompters.build import build_prompter
from ovod.engine.programs import generate_programs_from_descriptions

logger = logging.get_logger(__name__)

os.environ['REQUESTS_CA_BUNDLE'] = r"/etc/ssl/certs/ca-certificates.crt"
with open('api.key') as f:
    openai.api_key = f.read().strip()

def generate_programs(cfg):
    """ Generate programs for all objects in the dataset, given their descriptions """

    path_to_classnames = cfg.DATA.PATH_TO_CLASSNAMES
    description_path = cfg.OPENSET.ATTRIBUTES_FILE

    with open(description_path) as f:
        descriptions = json.load(f)

    output_dir = cfg.OPENSET.PATH_TO_PROGRAMS_FOLDER
    os.makedirs(output_dir, exist_ok=True)
    logging.setup_logging(output_dir)
    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### GET DESCRIPTIONS
    # if path_to_classnames is a file:
    if path_to_classnames is not None and os.path.isfile(path_to_classnames):
        # read lines, remove \n:
        with open(path_to_classnames) as f:
            classes = f.readlines()
        classes = [c.strip() for c in classes]
        # filter descriptions:
        descriptions = {k: v for k, v in descriptions.items() if k in classes}
        descriptions = list(descriptions.items())
    elif path_to_classnames is not None and path_to_classnames.isdigit():
        descriptions = list(descriptions.items())[:int(path_to_classnames)]
    else:
        descriptions = list(descriptions.items())

    ### BUILD PROMPTER:
    prompter = build_prompter(prompter_name='program_base', base_folder=cfg.OPENSET.PROGRAM_PROMPT_FOLDER)
    classifier = build_classifier(cfg, device=device)
    llm_wrapper = build_llm_wrapper(cfg)

    # Get one image example
    dataset_name, subset, arch = cfg.TEST.DATASET, cfg.TEST.SUBSET, cfg.MODEL.ARCH
    test_dataset = build_dataset(dataset_name, cfg, subset)
    #image_example = test_dataset[0][0].to(device)
    image_example = test_dataset[0][0]#.to(device)
    image_example = torch.from_numpy(image_example).to(device)

    ### GENERATE PROGRAMS
    generate_programs_from_descriptions(prompter, descriptions, output_dir, classifier, llm_wrapper, image_example)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    generate_programs(cfg)
