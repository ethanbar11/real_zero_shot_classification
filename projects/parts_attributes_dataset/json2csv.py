import random
import json
import pandas as pd
import argparse
from tqdm import tqdm

from extract_captions import build_imagenet21k_map, find_imagenet21k_image, parse_description


def parse_args():
    parser = argparse.ArgumentParser(description='Create CSV file from JSON file')
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--attributes-file",
                        type=str,
                        default="files/descriptors/ImageNet21k/imagenet21k_filtered_classes_v2_300c.json",
                        help="Path to the attributes file"
                        )
    parser.add_argument('-c', '--classes-file',
                        type=str,
                        default=None,
                        help="Path to the classes file"
                        )
    parser.add_argument("-o", "--output-file",
                        type=str,
                        default="files/clip_training_data/ImageNet21k/ox_no_name_gpt3/imagenet21k_filtered_classes_v2_300c.csv",
                        help="Path to the output file"
                        )
    parser.add_argument("-m", "--max-num-images",
                        type=int,
                        default=50,
                        help="Max number of images per class"
                        )

    return parser.parse_args()


def convert_classname(classname):
    """ Convert the class name to the format used in the dataset """
    ans = classname.replace(",", " ").replace("-", " ").replace("_", " ").replace("\'","").replace(' ', '').lower()
    return ans  # .replace(" ", "").lower()


def create_clip_training_data(attributes, classnames_to_indices_new, classes=None):
    """ Create the data for training CLIP """
    rows = []
    max_num_images = args.max_num_images
    json_name_to_imagenet21_name = {}
    for key in classnames_to_indices_new.keys():
        json_name_to_imagenet21_name[convert_classname(key)] = key
    for classname_in_json, sentences in tqdm(attributes.items()):
        try:
            class_info = classnames_to_indices_new[json_name_to_imagenet21_name[convert_classname(classname_in_json)]]
            class_id = class_info["class_id"]
            label = class_info["label"]
            image_paths = find_imagenet21k_image(class_info)

            if classes is None or (classes is not None and label in classes)\
                    or (classes is not None and classname_in_json in classes):
                for ii, sentence in enumerate(sentences):
                    for image_path in image_paths[:max_num_images]:
                        rows.append([image_path, label, sentence])
        except:
            print(f"Error with class {classname_in_json}")

    # shuffle the rows
    random.shuffle(rows)

    # add rows to df without for loop:
    df = pd.DataFrame(rows, columns=["image_path", "class_name", "description"])

    return df


# main
if __name__ == '__main__':
    args = parse_args()
    attributes_file = args.attributes_file
    output_file = args.output_file
    print(f"attributes_file: {attributes_file}")

    data_root = "/shared-data5/guy/data/imagenet21k/imagenet21k_resized"
    splits = ["imagenet21k_val", "imagenet21k_train", "imagenet21k_small_classes"]
    classnames_to_indices = build_imagenet21k_map(data_root, splits)

    classnames_to_indices_new = {}
    for k, v in classnames_to_indices.items():
        v["label"] = k
        k_new = convert_classname(k)
        classnames_to_indices_new[k_new] = v

    # load json file:
    with open(attributes_file, 'r') as f:
        attributes = json.load(f)

    classes = None
    if args.classes_file is not None:
        # load text file:
        with open(args.classes_file, 'r') as f:
            classes = f.read().splitlines()

    # create df:
    print(f"Creating CLIP training file for {len(attributes)} classes, {args.max_num_images} each class.")
    df = create_clip_training_data(attributes, classnames_to_indices_new, classes)
    # number of unique labels:
    num_unique_labels = len(df['class_name'].unique())
    print(f"df contains {len(df)} rows, {num_unique_labels} classes.")
    A = df['class_name']
    nan_idx = A[A.isnull()].index.tolist()
    print(f"Found {len(nan_idx)} rows with nan class_name.")
    # save df to csv
    output_file = output_file.replace('.csv', f"_{str(num_unique_labels)}c.csv")
    df.to_csv(output_file, index=False)
    print(f"df was saved to file {output_file}.")

    # load df:
    df = pd.read_csv(output_file)
    A = df['class_name']
    nan_idx = A[A.isnull()].index.tolist()
    print(f"Found {len(nan_idx)} rows with nan class_name.")
    if len(nan_idx) > 0:
        df = df.drop(nan_idx)
        df.to_csv(output_file, index=False)
        print(f"df was saved to file {output_file}.")
    # save df to csv
    output_file = output_file.replace('.csv', f"_{str(num_unique_labels)}c.csv")
