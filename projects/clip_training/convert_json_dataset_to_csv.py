import os
import json
import argparse
import pandas
import random

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ovod.datasets.build import build_dataset
from ovod.utils.parser import load_config
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
        "--cfg_file",
        type=str,
        default=None,
        help="Path to the dataset config."
    )

    parser.add_argument(
        "--opts",
        type=str,
        default=None,
        help="DONT TOUCH THIS."
    )

    parser.add_argument(
        "--test_percentage",
        type=float,
        default=0.2,
        help="Between [0,1], what percentage of the classes should go to test split. Notice that I'm splitting over classes and not indices."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='tmp',
        help="Where to output the two csvs, train and test, and the descriptions files for Zero shot."
    )

    parser.add_argument(
        "--seed",
        type=int,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    random.seed(args.seed)  # Replace 'your_seed_value' with your desired seed
    cfg = load_config(args)
    description_reader = build_description_reader(args.description_reader)
    dataset_name, subset, arch = cfg.TEST.DATASET, cfg.TEST.SUBSET, cfg.MODEL.ARCH
    train_dataset = build_dataset(dataset_name, cfg, 'train')
    test_dataset = build_dataset(dataset_name, cfg, 'test')

    with open(args.file_path, 'r') as f:
        data = json.load(f)

    classes_atts = {}
    for name, attributes_as_txt in data.items():
        atts_to_add = description_reader(attributes_as_txt)
        classes_atts[name] = atts_to_add

    tot_class_names = list(classes_atts.keys())
    # Shuffle the list
    random.shuffle(tot_class_names)

    # Split the list based on the test percentage
    split_index = int(len(tot_class_names) * (1 - args.test_percentage))
    train_class_names = tot_class_names[:split_index]
    test_class_names = tot_class_names[split_index:]

    descriptions_path = os.path.join(args.out_dir, 'descriptions_for_real_zs.json')
    paths = [os.path.join(args.out_dir, 'train_classes.txt'), os.path.join(args.out_dir, 'test_classes.txt')]
    for path in paths:
        names = train_class_names if 'train_classes.txt' in path else test_class_names
        with open(path, 'w') as f:
            for name in names:
                f.write(name + '\n')
    with open(descriptions_path, "w") as f:
        json.dump(classes_atts, f, indent=4)
    print(f'Saved descriptions and classs names to {args.out_dir}')
    train_data = []
    test_data = []
    datasets = [train_dataset, test_dataset]
    for dataset in datasets:
        for img, target, img_path, class_name in dataset:
            sentences = classes_atts[class_name]
            data_table = train_data if class_name in train_class_names else test_data
            for sentence in sentences:
                data_table.append((img_path, class_name, sentence))

    train_df = pandas.DataFrame(train_data, columns=['image_path', 'class_name', 'description'])
    test_df = pandas.DataFrame(test_data, columns=['image_path', 'class_name', 'description'])

    train_df.to_csv(os.path.join(args.out_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(args.out_dir, 'test.csv'), index=False)
    print('Saved csvs to: ', args.out_dir)
    print('Train file : ', os.path.join(args.out_dir, 'train.csv'))
    print('Test file : ', os.path.join(args.out_dir, 'test.csv'))
