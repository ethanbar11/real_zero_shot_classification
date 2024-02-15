import json
import os
import sys
import torch
from clip import clip
from torch.nn import functional as F
import numpy as np
import open_clip

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ovod.utils.parser import parse_args, load_config
import ovod.utils.logger as logging
from ovod.config.defaults import assert_and_infer_cfg
from ovod.datasets.build import build_dataset

logger = logging.get_logger(__name__)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def print_statistics(results):
    for dec_type, parts in results.items():
        for part, results in parts.items():
            print(f"{dec_type} {part}: {np.mean(results)}")


def test_openset(cfg):
    """
    Perform open-set testing on AttributeClassifier and ProgramClassifier model. Performs evaluation using `ClassEvaluator`
    Plot the results if needed
    Args:
        cfg (CfgNode): configs. Details can be found in
            config/defaults.py
    """
    # Set up environment.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set random seed from configs.
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    import random, os
    seed = cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Setup logging format.
    dataset_name, subset, arch = cfg.TEST.DATASET, cfg.TEST.SUBSET, cfg.MODEL.ARCH
    # Create testing dataset and loader.
    test_dataset = build_dataset(dataset_name, cfg, subset)
    negative_descriptions_path = r'files/descriptors/CUB/cub_gpt3_text-davinci-003_descriptions_col_negatives.json'
    with open(negative_descriptions_path, 'r') as f:
        negative_descriptions = json.load(f)
    # model, _, preprocess = open_clip.create_model_and_transforms(cfg["MODEL"]["MODEL_SIZE"],
    #                                                                   pretrained=cfg["MODEL"]["BACKBONE"],
    #                                                                   device='cuda')

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/16',
                                                                 pretrained='openai',
                                                                 device=device, *(), **{})
    model.eval()

    results = {}
    # data loaders:
    for idx, data in enumerate(tqdm(test_dataset)):
        image, target_index, img_path, target = data
        descriptions = negative_descriptions[target]
        for original_description, description_dict in descriptions.items():
            d = yaml.load(description_dict, Loader=yaml.FullLoader)
            parts = d['part']
            desc_type = d['attribute_type']
            # Check if original description contain number
            if any(char.isdigit() for char in original_description) or desc_type!='size':
                continue
            image = image.to(device)
            image_encodings = model.encode_image(image.unsqueeze(0)).float()
            image_encodings = F.normalize(image_encodings)
            tokenized_descriptions = clip.tokenize([original_description] + d['negatives']).to(device)
            text_features = F.normalize(model.encode_text(tokenized_descriptions.to(device))).float()
            text_probs = (100.0 * image_encodings @ text_features.T).softmax(dim=-1)
            success = torch.argmax(text_probs[0]).item() == 0
            if desc_type not in results:
                results[desc_type] = {}
            for part in parts:
                if part not in results[desc_type]:
                    results[desc_type][part] = []
            for i, part in enumerate(parts):
                results[desc_type][part].append(success)
            if idx % 1000 == 0:
                print('-------idx : ', idx, '-------')
                print_statistics(results)
                output_path = r'tmp/clip_results.json'
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=4)

    print('------------------------')
    print('FINAL:')
    print_statistics(results)
    output_path = r'tmp/clip_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    test_openset(cfg)
