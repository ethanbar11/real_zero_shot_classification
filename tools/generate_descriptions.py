import os
import sys
import argparse
import time
import json
from tqdm import tqdm


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
        "-a",
        "--ann-path",
        type=str,
        default="files/classnames/misc/small_shoes_boots.txt"  # "datasets/lvis/lvis_v1_val.json"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="{}_gpt3_{}_descriptions_{}.json"
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
        help="prompt type, e.g., 'own_noname', 'own' 'program_base', 'col'"
    )

    parser.add_argument(
        "--prompt-directory",
        type=str,
        default=None,
        help="If you choose ProgramPrompter, choose the directory to read the prompts from"
    )

    parser.add_argument(
        "--super-category",
        type=str,
        default=None,
        help="What is the super category of the dataset you are creating, e.g Dogs120 is dog, Cars196 is Cars, ImageNet21k is object."
    )




    args = parser.parse_args()
    return args


def get_descriptions(category, prompter, llm_wrapper, prompt_params, name_filter=None):
    prompt = prompter.get_prompt(category)
    prompt_batch = [prompt for i in range(10)]
    answers = llm_wrapper.forward_batch(prompt_batch)
    if name_filter:
        answers_without_name = name_filter.filter_names_from_atts(category,answers)
        print(category+'\n\n')
        for answer in answers_without_name:
            print(answer)
        print(answers_without_name)
        return answers_without_name
    return answers

def get_description_batched(categories, prompter, llm_wrapper,name_filter =None):
    prompt_batch = []
    for category in categories:
        prompt_batch.append(prompter.get_prompt(category))
    answers = llm_wrapper.forward_batch(prompt_batch)
    parsed_answers = [prompter.convert_answer_to_description(answer) for answer in answers]

    answer_dict = {}
    for i, category in enumerate(categories):
        answer_dict[category] = parsed_answers[i]
    if name_filter:
        for category, atts in answer_dict.items():
            answers_without_name = name_filter.filter_names_from_atts(category,atts)
            print(category+'\n\n')
            for answer in answers_without_name:
                print(answer)
            answer_dict[category] = answers_without_name
    return answer_dict

def save_results(results, output_path):
    # Fix classes names
    # results = {fix_classname_special_chars(k): v for k, v in results.items()}
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def main(args):
    category_list = parse_ann_file(filepath=args.ann_path)

    # setup
    prompter = build_prompter(args.prompt_type, base_folder=args.prompt_directory)
    prompt_params = prompter.get_llm_params()
    llm_wrapper = OpenAIWrapper(**prompt_params)
    super_category = args.super_category if args.super_category else 'object'
    name_filter = NameFilter(super_category)
    BATCH_SIZE = 10

    # output file
    annfile_basename = os.path.splitext(os.path.basename(args.ann_path))[0]
    output_path = args.output_path.format(annfile_basename, args.openai_model, args.prompt_type)

    # loop:
    all_responses = {}
    if args.prompt_type == 'program_base':
        for idx in tqdm(range(0, len(category_list), BATCH_SIZE)):
            categories_to_generate_description_for = category_list[idx:idx + BATCH_SIZE]
            all_responses.update(get_description_batched(categories_to_generate_description_for,
                                                         prompter, llm_wrapper, name_filter))
    else:
        # loop over category_list with progress bar with tqdm.
        if "ox" in args.prompt_type:
            for i, category in tqdm(enumerate(category_list), total=len(category_list)):
                all_responses[category] = get_descriptions(category, prompter, llm_wrapper, prompt_params, name_filter)
                if i % 100 == 0:
                    save_results(all_responses, output_path)
        elif "col" in args.prompt_type:
            prompt_batch, category_batch = [], []
            for ii, category in tqdm(enumerate(category_list), total=len(category_list)):
                prompt = prompter.get_prompt(category)
                prompt_batch.append(prompt)
                category_batch.append(category)
                if (ii + 1) % 10 == 0 or ii == len(category_list) - 1:
                    answers = llm_wrapper.forward_batch(prompt_batch)
                    descriptions = [prompter.convert_answer_to_description(answer) for answer in answers]
                    if name_filter:
                        descriptions = [name_filter.filter_names_from_atts(category_batch[k], descriptions[k])
                                                      for k in range(len(descriptions))]
                    for jj, desc in enumerate(descriptions):
                        all_responses[category_batch[jj]] = desc
                    prompt_batch, category_batch = [], []

                if ii % 100 == 0:
                    save_results(all_responses, output_path)


    # save results
    save_results(all_responses, output_path)
    print(f"{args.openai_model} descriptions were generated to {output_path} ")

if __name__ == "__main__":
    args = get_args()
    main(args)

    # /shared-data5/guy/data/lvis/lvis_v1_val.json
