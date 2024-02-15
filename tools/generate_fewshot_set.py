import shutil
import os
import sys
import argparse
import openai

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ovod.config.defaults import assert_and_infer_cfg
from ovod.utils.parser import parse_args, load_config
from ovod.datasets.build import build_dataset
from ovod.utils.openai_utils import parse_ann_file

# os.environ['REQUESTS_CA_BUNDLE'] = r"/etc/ssl/certs/ca-certificates.crt"
# with open('api.key') as f:
#     openai.api_key = f.read().strip()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='data/CUB_200_2011/annotations/train.txt')

    args = parser.parse_args()
    return args


def main(cfg):
    print('Starting to move files.')
    category_list = parse_ann_file(filepath=cfg.DATA.PATH_TO_CLASSNAMES)
    print('Categories: ', category_list)
    dataset = build_dataset(cfg.TEST.DATASET, cfg=cfg, split=cfg.TEST.SUBSET)
    output_dir = cfg.OPENSET.PATH_TO_SYNTHETIC_FOLDER
    class_to_paths = {}
    for row in dataset:
        if row[-1] not in class_to_paths:
            class_to_paths[row[-1]] = []
        class_to_paths[row[-1]].append(row[-2])
    files = {}
    category_list = [category.replace(' ', '_') for category in category_list]
    for category in category_list:
        paths = class_to_paths[category.replace('_',' ')][:cfg.OPENSET.SYNTHETIC_IMAGE_AMOUNT]
        file_names = list(map(os.path.basename, paths))
        files[category] = file_names

        for idx, path in enumerate(paths):
            out_path = os.path.join(output_dir, 'images', category, f'image_{idx}.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy(path, out_path)
    print('Creating files.txt and images.txt')
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        for category, file_names in files.items():
            for idx, file_name in enumerate(file_names):
                f.write(f'{idx} {category.replace(" ", "_")}/image_{idx}.png\n')

    # Copying last file
    orig_path = r'/shared-data5/guy/exps/grounding_synthetic/openjourney_cub_gpt3_text-davinci-003_descriptions_ox_noname/train_test_split.txt'
    dest_path = os.path.join(output_dir, 'train_test_split.txt')
    shutil.copy(orig_path, dest_path)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    main(cfg)
