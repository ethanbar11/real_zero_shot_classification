import json
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def add_names_to_description(json_file: str):
    """
    Adds the class name to the description.
    """
    # read json:
    with open(json_file, 'r') as f:
        data = json.load(f)

    for class_name, descriptions in data.items():
        descriptions = [f"{class_name}. {description}" for description in descriptions]
        data[class_name] = descriptions

    # save json:
    output_file1 = json_file.replace('.json', '_with_classname.json')
    with open(output_file1, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved json to file {output_file1}.")

    # for class_name, descriptions in data.items():
    #     data[class_name] = [f"A photo of {class_name}"] # * 10
    #     #data[class_name] = [f"{class_name}"] * 10
    # output_file2 = json_file.replace('.json', '_with_classname_only.json')
    # with open(output_file2, 'w') as f:
    #     json.dump(data, f, indent=4)
    # print(f"Saved json to file {output_file2}.")


    # main
if __name__ == '__main__':
    # json_file1 = "files/descriptors/Cars196/cars196_gpt3_text-davinci-003_descriptions_ox_noname_filtered.json"
    # json_file1 = "files/descriptors/CUB/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4.json"
    # json_file1 = "files/descriptors/Dogs120/descriptions_for_real_zs.json"
    # json_file1 = "files/descriptors/FLOWERS102/flowers102_gpt3_text-davinci-003_descriptions_ox_noname_clean.json"
    # json_file1 = "files/descriptors/Food101/food_gpt3_text-davinci-003_descriptions_ox_noname_filtered.json"
    # json_file1 = "files/descriptors/OXFORD_PET/oxford_pet_gpt3_text-davinci-003_descriptions_ox_noname.json"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', "--description-file",
        type=str,
        default=None,
        help="The json file to add the class name to."
    )
    args = parser.parse_args()
    add_names_to_description(args.description_file)
