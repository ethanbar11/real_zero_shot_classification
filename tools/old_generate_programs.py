import os
import sys
import argparse
import json
import openai
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ovod.classifiers.build import build_classifier
from ovod.llm_wrappers.build import build_llm_wrapper
from ovod.datasets.build import build_dataset
from ovod.utils.parser import load_config
from ovod.prompters.build import build_prompter
from ovod.engine.programs import generate_programs_from_descriptions

os.environ['REQUESTS_CA_BUNDLE'] = r"/etc/ssl/certs/ca-certificates.crt"
with open('api.key') as f:
    openai.api_key = f.read().strip()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--description_path",
        type=str,
        default="files/descriptors/CUB/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4.json"
    )
    # parser.add_argument(
    #     "-o",
    #     "--output_path",
    #     type=str,
    #     default="{}_prompt_{}",  # "files/programs/set7/"
    # )
    parser.add_argument(
        "-l",
        "--limit",
        type=str,
        default=None
    )
    parser.add_argument(
        "-cfg_file",
        type=str,
        default=None
    )

    parser.add_argument(
        "-opts",
        type=str,
        default=None
    )

    args = parser.parse_args()
    return args


def generate_programs(args):
    """ Generate programs for all objects in the dataset, given their descriptions """

    with open(args.description_path) as f:
        descriptions = json.load(f)

    limit = args.limit

    cfg = load_config(args)
    output_dir = cfg.OPENSET.PATH_TO_PROGRAMS_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### GET DESCRIPTIONS
    # if limit is a file:
    if limit is not None and os.path.isfile(limit):
        # read lines, remove \n:
        with open(limit) as f:
            classes = f.readlines()
        classes = [c.strip() for c in classes]
        # filter descriptions:
        descriptions = {k: v for k, v in descriptions.items() if k in classes}
        descriptions = list(descriptions.items())
    elif limit is not None and limit.isdigit():
        descriptions = list(descriptions.items())[:int(limit)]
    else:
        descriptions = list(descriptions.items())

    ### BUILD PROMPTER:
    prompter = build_prompter(prompter_name='program_base', base_folder=cfg.OPENSET.PROGRAM_PROMPT_FOLDER)
    classifier = build_classifier(cfg, device=device)
    llm_wrapper = build_llm_wrapper(cfg)

    # Get one image example
    dataset_name, subset, arch = cfg.TEST.DATASET, cfg.TEST.SUBSET, cfg.MODEL.ARCH
    test_dataset = build_dataset(dataset_name, cfg, subset)
    image_example = test_dataset[0][0].to(device)

    ### GENERATE PROGRAMS
    generate_programs_from_descriptions(prompter, descriptions, output_dir, classifier, llm_wrapper,
                                        image_example)


if __name__ == "__main__":
    args = get_args()
    generate_programs(args)