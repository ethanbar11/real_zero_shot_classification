import os
import sys
import argparse
import time
import json
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from projects.clip_training.filter_names_from_descriptions_file import NameFilter

from ovod.llm_wrappers.open_ai_wrapper import OpenAIWrapper
from ovod.utils.openai_utils import parse_ann_file
from ovod.prompters.build import build_prompter
from ovod.utils.class_name_fix import fix_classname_special_chars

os.environ['REQUESTS_CA_BUNDLE'] = r"/etc/ssl/certs/ca-certificates.crt"
with open('api.key') as f:
    my_api_key = f.read().strip()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptions-file",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
    )

    parser.add_argument(
        "-p",
        "--prompt_folder",
        default="files/prompts/negative_mining_prompts",
        type=str,
        help="The folder where the prompts are being read from. Dont touch this unless you know you need to change "
             "default negative prompt mining dir."
    )

    args = parser.parse_args()
    return args


def get_negatives(sentences, prompter, llm_wrapper):
    prompt_batch = []
    for sentence in sentences:
        prompt_batch.append(prompter.get_prompt(sentence))
    answers = llm_wrapper.forward_batch(prompt_batch)
    answers_dict = {}
    for idx in range(len(answers)):
        answers_per_sentence = answers[idx]#.split('\n')
        answers_dict[sentences[idx]] = answers_per_sentence
    return answers_dict


def main(args):
    with open(args.descriptions_file, 'r') as f:
        descriptions = json.load(f)

    all_responses = {}
    prompter = build_prompter("program_base", args.prompt_folder)
    prompt_params = prompter.get_llm_params()
    llm_wrapper = OpenAIWrapper(**prompt_params)
    for category, atts in tqdm(descriptions.items()):
        all_responses[category] = get_negatives(atts, prompter, llm_wrapper)

    with open(args.output_path, 'w') as f:
        json.dump(all_responses, f, indent=4)
    print(f"descriptions were generated to {args.output_path} ")


if __name__ == "__main__":
    args = get_args()
    main(args)

    # /shared-data5/guy/data/lvis/lvis_v1_val.json
