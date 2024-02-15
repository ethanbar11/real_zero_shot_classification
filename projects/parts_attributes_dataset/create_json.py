import yaml
import json
import os
import random
import argparse
import pandas as pd

def get_random_image_paths(directory, k=1):
    """ Function to randomly select an image from a given directory """
    try:
        images = [file for file in os.listdir(directory) if file.endswith(('.png', '.jpg', '.jpeg', '.JPEG'))]
        if images and len(images) >= k:
            return [os.path.join(directory, image) for image in random.sample(images, k)]
        else:
            return None  # Not enough images to choose from
    except FileNotFoundError:
        return None

def create_labels(input_yaml_file: str, attr_categories: str, k: int = 5):
    """ XXX """
    # List of all possible category options (read from yaml file):
    with open(attr_categories, 'r') as file:
        all_attribute_categories = yaml.safe_load(file)

    # Read the YAML data
    with open(input_yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    new_data = []
    for item in data:
        B = list(item)  # item[list(item)[0]]['attributes']
        B.remove('name')
        B.remove('path')
        if type(item[B[0]]) is dict:
            new_data.append({'name': item['name'], 'attributes': item[B[0]]['attributes'], 'path': item['path']})
    # write to file
    with open(input_yaml_file, 'w') as file:
        yaml.dump(new_data, file)
    print(f"saved new_data file")
    data = new_data

    if data is dict:
        # convert to list, where key is the 'name' attribute
        data = [{'name': key, **value} for key, value in data.items()]

    # Process the data and construct the JSON structure
    labels_and_gt = []

    #for bird, attributes_info in data.items():
    for attributes_info in data:
        object_name = attributes_info['name']
        attributes = attributes_info['attributes']
        object_dir = attributes_info['path']
        # Get a random image path from the object's folder
        image_paths = get_random_image_paths(object_dir, k=k)
        # loop over attr categories
        for attr in attributes:
            attr_category = attr['category']
            attribute_type = attr['attribute_type']
            valid_options = attr['valid']
            attr_prompt = attr['value']
            if attribute_type in all_attribute_categories:
                attr_labels = all_attribute_categories[attribute_type]
                # Create labels list with all possible colors
                labels = [attr_prompt.format(attr_label) for attr_label in attr_labels]
                # build the gt list, assume valid is a list with few options:
                gt = [attr_prompt.format(valid_label) for valid_label in valid_options] if valid_options else []
                # add item for each image
                for image_path in image_paths:
                    labels_and_gt.append({"path": image_path, "labels": labels, "gt": gt, "object": object_dir})
                print(f"Added {len(image_paths)} images for {object_name} ({attr_category})")
            else:
                print(f"Skipping {object_name} ({attr_category}, {attribute_type})")

    # Shuffle the data
    random.shuffle(labels_and_gt)

    return labels_and_gt


def convert_to_csv_openclip(labels_and_gt: dict):
    """ Convert the labels to a CSV file """
    df = pd.DataFrame(columns=["image_path", "class_name", "description"])
    for item in labels_and_gt:
        #csv_data.append([item['path'], item['labels'][0], item['gt'][0]])
        for gt in item['gt']:
            df.loc[len(df)] = [item['path'], item['object'], gt]
    return df

def parse_args():
    parser = argparse.ArgumentParser(description='Create JSON file from YAML file')
    parser.add_argument('-i', '--input_yaml_file', type=str, default='projects/parts_attributes_dataset/CUB_attributes_gt_Guy1.yaml',
                        help='Input YAML file')
    parser.add_argument('--attr_categories', type=str, default='projects/parts_attributes_dataset/categories.yaml',
                        help='Attribute categories YAML file')
    parser.add_argument('-k', '--k', type=int, default=5,
                        help='Number of images per category')
    parser.add_argument('-o', '--output_json_file', type=str, default=None,
                        help='Output JSON file')
    return parser.parse_args()


# main
if __name__ == '__main__':

    args = parse_args()
    input_yaml_file = args.input_yaml_file
    attr_categories = args.attr_categories
    k = args.k
    if args.output_json_file:
        output_json_file = args.output_json_file
        output_csv_file = output_json_file.replace('.json', '.csv')
    else:
        output_json_file = input_yaml_file.replace('.yaml', '.json')
        output_csv_file = input_yaml_file.replace('.yaml', '.csv')

    # Create the labels
    labels = create_labels(input_yaml_file, attr_categories, k=k)
    # Write the JSON data to a file
    with open(output_json_file, 'w') as file:
        json.dump(labels, file, indent=4)

    print(f"JSON file created: {output_json_file}, containing {len(labels)} samples.")

    # Convert to CSV format
    df = convert_to_csv_openclip(labels)
    df.to_csv(output_csv_file, index=False)
    print(f"df was saved to file {output_csv_file}, contains {len(df)} rows.")
