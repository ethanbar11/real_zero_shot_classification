import os
import json
import argparse
import pandas as pd
import random

import sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from projects.parts_attributes_dataset.extract_captions import build_imagenet21k_map, find_imagenet21k_image

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
        default='list',
        choices=['yaml', 'list'],
        help="Path to the json file containing the descriptions."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default='tmp',
        help="Where to output the two csvs, train and test, and the descriptions files for Zero shot."
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    random.seed(args.seed)  # Replace 'your_seed_value' with your desired seed
    description_reader = build_description_reader(args.description_reader)

    classnames_to_indices = build_imagenet21k_map()
    output_path = f'{args.out_dir}/train.csv'
    imagenet21_k_parsed_name_to_original = {}
    for key in classnames_to_indices.keys():
        new_name = key.lower().replace('-','').replace('_','').replace(',','').replace(' ','')
        imagenet21_k_parsed_name_to_original[new_name] = key
    with open(args.file_path, 'r') as f:
        data = json.load(f)

    classes_atts = {}
    for name, attributes_as_txt in data.items():
        atts_to_add = description_reader(attributes_as_txt)
        classes_atts[name] = atts_to_add
    imagenet1k_parsed_name_to_original = {}
    for key in classes_atts.keys():
        new_name = key.lower().replace('-','').replace('_','').replace(',','').replace(' ','')
        imagenet1k_parsed_name_to_original[key] =new_name

    imagenet_21k_class_name_to1k = {}
    tot_class_names = list(classes_atts.keys())
    for imgnet_1k_cls_name in tot_class_names:
        parsed_name = imagenet1k_parsed_name_to_original[imgnet_1k_cls_name]
        imagenet_21k_name = imagenet21_k_parsed_name_to_original[parsed_name]
        imagenet_21k_class_name_to1k[imagenet_21k_name] = imgnet_1k_cls_name

    imagenet_21k_class_name_to_csv = list(imagenet_21k_class_name_to1k.keys())
    # get random image for each classname in
    data_as_lst = []
    for ii, classname in tqdm(enumerate(imagenet_21k_class_name_to_csv)):
        class_info = classnames_to_indices[classname]
        class_id = class_info["class_id"]
        image_paths = find_imagenet21k_image(class_info)
        print(f"Processing class name {classname}. class_id: {class_id}. Class contains {len(image_paths)} images.")
        descriptions = classes_atts[imagenet_21k_class_name_to1k[classname]]
        for description in descriptions:
            for path in image_paths:
                data_as_lst.append((path, classname, description))

        if ii % 10 == 0:
            df = pd.DataFrame(data_as_lst, columns=['image_path', 'class_name', 'description'])
            df.to_csv(output_path, index=False)
            print(f"df was saved to file {output_path}, contains {len(df)} rows.")

    df = pd.DataFrame(data_as_lst, columns=['image_path', 'class_name', 'description'])
    df.to_csv(output_path, index=False)
    print(f"df was saved to file {output_path}, contains {len(df)} rows.")