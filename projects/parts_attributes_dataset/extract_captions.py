import os
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import textwrap
import pandas as pd
import yaml
import ast
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ovod.utils.parser import parse_args, load_config
from ovod.config.defaults import assert_and_infer_cfg
from ovod.llm_wrappers.build import build_llm_wrapper


def build_imagenet21k_map(data_root: str = "/shared-data5/guy/data/imagenet21k/imagenet21k_resized",
                          splits: str = ["imagenet21k_val", "imagenet21k_train", "imagenet21k_small_classes"]):
    # read text files, where each line is a class name
    classname_path = "files/classnames/imagenet21K_classes.txt"
    index_path = "files/classnames/imagenet21K_ids.txt"

    with open(classname_path) as f:
        classnames = f.readlines()
    classnames = [x.strip() for x in classnames]

    with open(index_path) as f:
        class_indices = f.readlines()
    class_indices = [x.strip() for x in class_indices]
    # map classnames to indices, classnames_to_indices[classname] = {"class_id": class_id, "path": None}
    classnames_to_indices = {}
    for idx, classname in enumerate(classnames):
        classnames_to_indices[classname] = {"class_id": class_indices[idx], "path": None}

    # iterate over all class_ids, make sure they exist in the data_root
    for classname, class_info in classnames_to_indices.items():
        folder_path = None
        for split in splits:
            p = os.path.join(data_root, split, class_info["class_id"])
            if os.path.exists(p):
                folder_path = p
                break
        class_info["path"] = folder_path
    # remove class_ids that don't exist in the data_root
    classnames_to_indices = {classname: class_info for classname, class_info in classnames_to_indices.items() if
                             class_info["path"] is not None}

    print(f"len(class_indices): {len(class_indices)}, len(classnames): {len(classnames)}, "
          f"len(classnames_to_indices): {len(classnames_to_indices)}")

    return classnames_to_indices


def find_imagenet21k_image(class_info):
    folder_path = class_info["path"]
    # get all images in folder
    images = os.listdir(folder_path)
    # get full path for random image:
    image_paths = [os.path.join(folder_path, image) for image in images]  # os.path.join(folder_path, images[0])
    return image_paths


def get_llm_description_ox(llm_wrapper, class_name):
    description_prompt = (f"There is an image of a {class_name}. "
                          f"Describe the essential parts and attributes of parts "
                          f"that the object in the image must have.")
    answer = llm_wrapper.forward([{'role': 'user', 'content': description_prompt}])
    return answer


def get_llm_description_col(llm_wrapper, class_name, attr_options=None, exmaple_directory=None):
    files = os.listdir(exmaple_directory)
    files = [file for file in files if file.endswith(".yaml")]
    messages = [{'role': 'system',
                 'content': f"Your task is to describe the essential parts and attributes "
                            f"of parts that an object in a given image must have. "
                            f"So focus on visual features that can be seen in the image. "
                            f"Please provide the answer in a yaml format. "
                 }]
    for file in files:
        example_class_name = file.split(".")[0]
        with open(os.path.join(exmaple_directory, file)) as f:
            description = f.read()
        messages.append({'role': 'user',
                         'content': f"assume the given image is of a: {example_class_name} \n"
                                    f"Can you describe the essential parts and attributes of parts that the object "
                                    f"in the image must have"})
        messages.append({'role': 'assistant',
                         'content': description})
    messages.append({'role': 'user',
                     'content': f"Now assume the image is of a {class_name}."
                                f"Can you describe the essential parts and attributes of parts that the object "
                                f"in the image must have?"
                                f"Start immediately in the yaml format. "
                     })

    answer = llm_wrapper.forward(messages)
    print(answer)
    return answer


def parse_description(descriptions, classname):
    names = classname.replace("_", " ").split(",")
    for name in names:
        descriptions = descriptions.replace(name, "object")

    sentences = descriptions.split("\n")

    # take sentences that are long enough:
    sentences = [sentence for sentence in sentences if len(sentence) > 10]

    return sentences, names


def display_image_with_caption(image_path, caption, wrap_width=70, output_file=None):
    # Load the image
    img = mpimg.imread(image_path)

    # Create a figure and axis to display the image
    fig, ax = plt.subplots(figsize=(10, 15))  # Adjust figure size as needed

    # Display the image
    ax.imshow(img)
    ax.axis('off')  # Hide axis

    # Wrap the caption text
    wrapped_caption = textwrap.fill(caption, wrap_width)

    # Display the caption below the image
    plt.figtext(0.5, 0.01, wrapped_caption, wrap=True, ha='center', fontsize=12)  # Adjust fontsize as needed

    # Show the plot with enough space for caption
    plt.subplots_adjust(bottom=0.3)  # Adjust the bottom parameter as needed to fit the caption

    if output_file is not None:
        plt.savefig(output_file)
    else:
        # Show the plot
        plt.show()


# main
if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    description_type = "col"  # "ox" or "col"
    data_root = "/shared-data5/guy/data/imagenet21k/imagenet21k_resized"
    example_directory = cfg.EXAMPLE_DIRECTORY
    # output_file = "projects/parts_attributes_dataset/imagenet21k_parts_descriptions_train"
    # output_file = "projects/parts_attributes_dataset/imagenet21k_parts_descriptions_train"
    # attr_options = None
    # splits = ["imagenet21k_val", "imagenet21k_train"]
    # num_random_classnames = 200

    output_file = "files/clip_training_data/ImageNet21k/imagenet21k_parts_descriptions_val"
    splits = ["imagenet21k_small_classes"]
    attr_options = None  # "projects/parts_attributes_dataset/categories.yaml"
    num_random_classnames = 50

    # output_file = "projects/parts_attributes_dataset/imagenet21k_parts_descriptions_debug"
    # splits = ["imagenet21k_small_classes"]
    # attr_options = None
    # num_random_classnames = 3

    # build ImageNet21k mapping:
    classnames_to_indices = build_imagenet21k_map(data_root, splits)

    # build LLM wrapper
    llm_wrapper = build_llm_wrapper(cfg=cfg)

    get_llm_description = get_llm_description_col if description_type == "col" else get_llm_description_ox

    list_of_classnames = []
    if description_type == "col":
        if os.path.exists(f"{output_file}.yaml"):
            # read yaml to dict:
            with open(f"{output_file}.yaml") as f:
                yam = yaml.safe_load(f)
            print(f"Loaded {len(yam)} items from file {output_file}.yaml.")
            list_of_classnames = [item["name"] for item in yam]
        else:
            # init yaml data:
            yam = []
    elif description_type == "ox":
        if os.path.exists(f"{output_file}.csv"):
            # read csv to df:
            df = pd.read_csv(f"{output_file}.csv")
            print(f"Loaded {len(df)} items from file {output_file}.csv.")
            list_of_classnames = df["class_name"].unique()
        else:
            # init df with columns: image_path, class_name, description
            df = pd.DataFrame(columns=["image_path", "class_name", "description"])

    # get classnames from classnames_to_indices
    possible_classes = list(classnames_to_indices.keys())
    # exclude classes that already exist in the output file:
    possible_classes = [classname for classname in possible_classes if classname not in list_of_classnames]
    # get random classnames:
    random_classnames = random.sample(possible_classes, num_random_classnames)
    # get random image for each classname in
    for ii, classname in tqdm(enumerate(random_classnames)):
        class_info = classnames_to_indices[classname]
        class_id = class_info["class_id"]
        image_paths = find_imagenet21k_image(class_info)
        print(f"Processing class name {classname}. class_id: {class_id}. Class contains {len(image_paths)} images.")
        descriptions = get_llm_description(llm_wrapper, classname, attr_options, example_directory)

        if description_type == "col":
            try:
                # descriptions is in yaml format. Convert to dict:
                object_dict = yaml.safe_load(descriptions)
                object_dict["name"] = classname
                object_dict["path"] = class_info["path"]
                # add to yam:
                yam.append(object_dict)
            except:
                print(f"Failed to parse description for class {classname}.")
                continue

        elif description_type == "ox":
            # parse description
            sentences, names = parse_description(descriptions, classname)
            print(f"Found {len(sentences)} sentences in description.")

            # shuffle the sentences
            random.shuffle(sentences)
            # shuffle the images
            random.shuffle(image_paths)
            min_size_sentences_images = min(len(sentences), len(image_paths))
            for ii, sentence in enumerate(sentences[:min_size_sentences_images]):
                # add to df, do not use append:
                df.loc[len(df)] = [image_paths[ii], classname, sentence]

        # if ii % 10 == 0 or :
        # save each 10 interactions, and at the end
        if ii % 10 == 0 or ii == (len(random_classnames) - 1):
            if description_type == "col":
                # save as yaml:
                with open(f"{output_file}.yaml", 'w') as f:
                    yaml.dump(yam, f)
                print(f"df was saved to file {output_file}.yaml, contains {len(yam)} items.")

            elif description_type == "ox":
                # save df to csv
                df.to_csv(f"{output_file}.csv", index=False)
                print(f"df was saved to file {output_file}.csv, contains {len(df)} rows.")

# A = [
#     "Trunk: The trunk of the silver ash tree is usually straight and tall. It is covered in a smooth, grey bark that can sometimes have a silverish hue, hence the name. \n",
#     "2. Branches: The branches of the silver ash tree spread out widely from the trunk. They are also covered in the same smooth, grey bark as the trunk. \n",
#     "3. Leaves: The leaves of the silver ash tree are its most distinctive feature. They are usually a bright, glossy green on top and a silvery-white underneath. They are typically oval or lance-shaped and arranged in opposite pairs along the branches. \n",
#     "4. Flowers: If the image is taken during the right season, the silver ash tree may be in bloom. Its flowers are small, white, and fragrant, and they grow in clusters at the ends of the branches. \n",
#     "5. Fruits: After flowering, the silver ash tree produces small, winged fruits that are usually a light brown color. These fruits contain the tree's seeds and are dispersed by the wind. \n",
#     "6. Roots: While not usually visible in an image, the silver ash tree has a deep and extensive root system that helps it to withstand drought and other adverse conditions. \n",
#     "7. Overall Shape: The overall shape of the silver ash tree is usually rounded or oval, with a broad canopy that provides plenty of shade. \n",
#     "8. Size: The silver ash tree is a large tree, typically reaching heights of up to 20-30 meters. The size of the tree in the image can give an indication of its age, as these trees can live for many decades. \n"
# ]
# description = f"{classname} \n {''.join(A)} \n"

# read df:
# df = pd.read_csv("imagenet21k_parts_descriptions_train_v1_200c.csv")
