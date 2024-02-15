import tqdm
import json
import argparse
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ovod.prompters.build import build_prompter
from ovod.llm_wrappers.open_ai_wrapper import OpenAIWrapper
from projects.clip_training.description_readers import build_description_reader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Path to the json file containing the descriptions."
    )

    parser.add_argument(
        "--description_reader",
        type=str,
        choices=['yaml', 'list'],
        help="Path to the json file containing the descriptions."
    )

    parser.add_argument(
        "--super-category",
        type=str,
        default='object',
        help="Path to the json file containing the descriptions."
    )
    args = parser.parse_args()
    return args


class NameFilter:
    def __init__(self,super_category='object'):
        params = {
            'model': 'gpt-3.5-turbo',
            'temperature': 1.0,
            'max_tokens': 100,
            'top_p': 1.0,
            'stop': '.',
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        self.super_category = super_category
        self.llm_wrapper = OpenAIWrapper(**params)
        prompt_dir = r'files/prompts/filter_names'
        self.prompter = build_prompter('program_base', prompt_dir)

    def filter_names_from_atts(self, category, atts):
        batch = []
        for att in atts:
            query = f'{category}\n{att}'
            prompt = self.prompter.get_prompt(query)
            batch.append(prompt)
        answers_from_llm = self.llm_wrapper.forward_batch(batch)
        return list(map(lambda x: filter_name(category, x, self.super_category), answers_from_llm))


def filter_name(category, description, default):
    base_names = [category.replace(' ', ''),category.replace('_', ' '),'{super category}']
    names = []
    for base_name in base_names:
        names += [base_name, base_name.lower(), base_name.upper(), base_name.capitalize()]
    new_description = description
    for name in names:
        new_description = new_description.replace(name, default)
    return new_description


if __name__ == '__main__':
    args = get_args()
    description_reader = build_description_reader(args.description_reader)
    name_filter = NameFilter(args.super_category)
    with open(args.file_path, 'r') as f:
        data = json.load(f)
    clean_data = {}
    for category, atts in tqdm.tqdm(data.items()):
        clean_atts = []
        responses = name_filter.filter_names_from_atts(category, atts)
        print(f'\n\n Category: {category}\n')
        print(responses)
        clean_atts.append(responses)
        clean_data[category] = clean_atts
    output_path = args.file_path.replace('.json', '_clean.json')
    with open(output_path, 'w') as f:
        json.dump(clean_data, f, indent=4)
