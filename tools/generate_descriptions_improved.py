import os
import sys
import argparse
import json
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ovod.llm_wrappers.open_ai_wrapper import OpenAIWrapper

from ovod.utils.openai_utils import openai_request, parse_ann_file
from ovod.prompters.build import build_prompter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--ann-path",
        type=str,
        default="files/classnames/shoes_boots.txt"  # "datasets/lvis/lvis_v1_val.json"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="tmp"
    )
    parser.add_argument(
        "-m",
        "--openai-model",
        type=str,
        default="text-davinci-003",
        help="openai model to use for generation, options are 'text-davinci-003', 'text-davinci-002'"
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=0
    )
    parser.add_argument(
        "-p",
        "--prompt-type",
        type=str,
        default="ox",
        help="prompt type, e.g., 'own_noname', 'own' "
    )
    parser.add_argument(
        "-base_folder",
        type=str,
        default=None,
        help="If prompt type is DescriptionPrompterAdvanced, contains the prompting files."
    )

    parser.add_argument(
        "-to_baseline_folder",
        type=str,
        default=None,
        help="If prompt type is DescriptionPrompterAdvanced, contains the prompting files to convert from" +
             "raw description to baseline description."
    )

    args = parser.parse_args()
    return args


def save_file(program_descriptions, baseline_descriptions, output_path):
    os.makedirs(output_path, exist_ok=True)
    program_descriptions_path = os.path.join(output_path, "program_descriptions.json")
    baseline_descriptions_path = os.path.join(output_path, "baseline_descriptions.json")
    with open(program_descriptions_path, 'w') as f:
        json.dump(program_descriptions, f, indent=4)
    with open(baseline_descriptions_path, 'w') as f:
        json.dump(baseline_descriptions, f, indent=4)
    print(f"Program descriptions were generated to {program_descriptions_path} ")
    print(f"Baseline descriptions were generated to {baseline_descriptions_path} ")


def main(args):
    category_list = parse_ann_file(filepath=args.ann_path)
    output_path = args.output_path
    assert category_list, output_path
    print(f'Starting to run. Output path: f{output_path}')
    print(f'base prompt folder : {args.base_folder}, prompt of description to sentences : {args.to_baseline_folder}')
    program_descriptions = {}
    baseline_descriptions = {}
    first_prompter = build_prompter(args.prompt_type, base_folder=args.base_folder)
    to_baseline_prompter = build_prompter(args.prompt_type, base_folder=args.to_baseline_folder)

    prompt_params = first_prompter.get_llm_params()
    prompt_params2 = to_baseline_prompter.get_llm_params()
    prompt_params['temperature'] = 1.0
    prompt_params['frequency_penalty'] = 1.0
    prompt_params['presence_penalty'] = 1.0

    llm_wrapper_raw_description = OpenAIWrapper(**prompt_params)
    llm_wrapper_description_to_sentences = OpenAIWrapper(**prompt_params2)

    # loop over category_list with progress bar with tqdm.
    for ii, category in tqdm(enumerate(category_list), total=len(category_list)):
        messages = first_prompter.get_prompt(category)
        print('Getting description for category: ', category)
        program_description_with_name = llm_wrapper_raw_description.forward(messages)
        program_description = program_description_with_name
        messages2 = to_baseline_prompter.get_prompt(program_description)
        baseline_description = llm_wrapper_description_to_sentences.forward(messages2)
        program_descriptions[category] = program_description
        baseline_descriptions[category] = baseline_description
        print(program_description)
        print('Description split into sentences:')
        print(baseline_description)
        print('\n\n')
        if ii % 2 == 0:
            save_file(program_descriptions, baseline_descriptions, output_path)
    save_file(program_descriptions, baseline_descriptions, output_path)




if __name__ == "__main__":
    args = get_args()
    main(args)
