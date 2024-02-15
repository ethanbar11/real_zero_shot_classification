import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from projects.parts_attributes_dataset.OpenClipModel import OpenClipModel


def prase_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--model_name', type=str, default='ViT-B-32', help='model name')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='checkpoint path')
    parser.add_argument('-f', '--filename', type=str, default='samples/dog.png', help='image filename')
    parser.add_argument('-l', '--labels', type=str, nargs='+', default=["a diagram", "a dog", "a cat"], help='labels')
    parser.add_argument('-o', '--output_folder', type=str, default="outputs", help='output folder')
    parser.add_argument('-p', '--prompt', type=str, default="", help="prompt, e.g., 'a photo of ' ")
    args = parser.parse_args()
    return args

def build_model(model_name: str, model_path: str = None):
    model = OpenClipModel(model_name, model_path)
    return model

def match_prediction_to_labels(prediction, label):
    # if prediction is a string:
    if isinstance(prediction, str):
        # if label is a string:
        if isinstance(label, str):
            return prediction == label
        # if label is a list of strings:
        elif isinstance(label, str):
            return prediction in label

    # if prediction is a list of strings:
    elif isinstance(prediction, list):
        # if label is a string:
        if isinstance(label, str):
            return label in prediction
        # if label is a list of strings:
        elif isinstance(label, list):
            return any([x in prediction for x in label])

    else:
        raise ValueError("prediction/label must be a string or a list of strings")

def compute_top_k_accuracy(results: dict, k: int = 1):
    topk = 0
    for r in results:
        #if r["gt"] in [r["labels"][i] for i in np.argsort(r["text_probs"])[-3:]]:
        topk_labels = [r["labels"][i] for i in np.argsort(r["text_probs"])[-k:]]
        if match_prediction_to_labels(topk_labels, r["gt"]):
            topk += 1
    topk /= len(results)
    return topk


def run_itm(args):
    # check if args.filename is a json file:
    import json
    if os.path.splitext(args.filename)[1] == '.json':
        with open(args.filename) as f:
            data = json.load(f)
        paths = [x["path"] for x in data]
        labels_list = [x["labels"] for x in data]
        if "gt" in data[0]:
            gt_list = [x["gt"] for x in data]
        else:
            gt_list = [None] * len(data)
        output_folder = os.path.join(args.output_folder, os.path.splitext(os.path.basename(args.filename))[0])
    else:
        paths = [args.filename]     # ["samples/dog.png"]
        labels_list = [args.labels]     # [["a diagram", "a dog", "a cat"]]
        output_folder = args.output_folder

    model = build_model(model_name=args.model_name, model_path=args.checkpoint)   # 'ViT-B-32'
    if args.prompt != "":
        model.set_prompt(args.prompt)

    os.makedirs(output_folder, exist_ok=True)

    results = []
    num_plots = 10
    num_plots = min(num_plots, len(paths))
    model_basename = args.model_name
    # if basename contains ":" or too long, make is easier to read:
    model_basename = model_basename.replace(":", "_").replace("/", "_")
    # make basename shorter:
    model_basename = model_basename[:30]

    print(f"Starting image-text matching on {len(paths)} images. Will plot {num_plots} images to {output_folder}.")
    for image_path, gt, labels in tqdm(zip(paths, gt_list, labels_list), total=len(paths)):
        image = model.process_image(image_path)
        text = model.process_text(labels)
        text_probs = model.run_itm(image, text)
        # # apply softmax (text_prob is torch tensor):
        # text_probs = torch.nn.functional.softmax(text_probs, dim=1)

        #print(np.sum(text_probs))
        # sum up the text_probs:
        #print(np.sum(text_probs[0], axis=0, keepdims=True))
        results.append({"image_path": image_path, "labels": labels, "text_probs": text_probs[0].tolist(), "gt": gt})

        if num_plots > 0:
            # print all labels with their corresponding probabilities
            # for i, label in enumerate(labels):
            #     print(f"{label}: {text_probs[0][i]:.5f}")

            # plot image and labels with their corresponding probabilities
            output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + f"_{model_basename}_itm.png")
            model.plot_itm_results(image_path, labels, text_probs, gt, output_path)
            print(f"image-text matching on {image_path} saved to {output_path}")
            num_plots -= 1

    return results, output_folder


if "__main__" == __name__:
    args = prase_args()

    results, output_folder = run_itm(args)

    # save statistics to text file:
    model_basename = args.model_name
    # if basename contains ":" or too long, make is easier to read:
    model_basename = model_basename.replace(":", "_").replace("/", "_")
    # make basename shorter:
    model_basename = model_basename[:30]
    with open(os.path.join(output_folder, f"{model_basename}_itm_results.txt"), "w") as f:
        # compute top-k accuracy:
        for k in [1, 2, 3]:
            topk = compute_top_k_accuracy(results, k)
            print(f"top-{k} accuracy: {topk:.3f}")
            f.write(f"top-{k} accuracy: {topk:.3f}\n")
        print(f"results saved to {os.path.join(output_folder, f'{model_basename}_itm_results.txt')}")

    # save results to csv file:
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, f"{model_basename}_itm_results.csv"), index=False)
    print(f"results saved to {os.path.join(output_folder, f'{model_basename}_itm_results.csv')}")

















    # import os
    # import glob
    # from collections import OrderedDict
    #
    # import torch
    # from PIL import Image
    # from transformers import CLIPVisionModel, CLIPTextModel, CLIPModel, CLIPProcessor, CLIPTokenizer
    # import open_clip
    #

    # model, tokenizer, preprocess_train, preprocess_val = load_model()
    # #model, tokenizer, preprocess_val = load_openclip_model()
    # print("loaded model")
    # run_classification(model, preprocess_val, tokenizer)

