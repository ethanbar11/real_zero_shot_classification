import nltk
from nltk.corpus import wordnet as wn
import random
import argparse
from extract_captions import build_imagenet21k_map

nltk.download('wordnet')


def get_hypernyms(synset, visited=None):
    if visited is None:
        visited = set()
    hypernyms = set()
    if synset not in visited:
        visited.add(synset)
        for hypernym in synset.hypernyms():
            hypernyms.add(hypernym)
            hypernyms.update(get_hypernyms(hypernym, visited))
    return hypernyms


def is_animal(category):
    synsets = wn.synsets(category, pos=wn.NOUN)
    for synset in synsets:
        hypernyms = get_hypernyms(synset)
        animal_hypernym = wn.synset('animal.n.01')
        if animal_hypernym in hypernyms:
            return True
    return False


def is_man_made_object(category):
    synsets = wn.synsets(category, pos=wn.NOUN)
    for synset in synsets:
        hypernyms = get_hypernyms(synset)
        artifact_hypernym = wn.synset('artifact.n.01')
        if artifact_hypernym in hypernyms:
            return True
    return False


def is_plant_or_flower(category):
    synsets = wn.synsets(category, pos=wn.NOUN)
    for synset in synsets:
        hypernyms = get_hypernyms(synset)
        plant_hypernym = wn.synset('plant.n.02')
        if plant_hypernym in hypernyms:
            return True
    return False


# Define a specific set of keywords for large man-made objects with meaningful internal parts
large_man_made_objects = {'bike', 'bicycle', 'airplane', 'aircraft', 'ship', 'boat', 'car',
                          'automobile', 'chair', 'dining_table', 'motorcycle', 'bus', 'truck', 'train',
                          'couch', 'sofa', 'bed', 'bench', 'van',
                          }


def is_large_man_made_object(category):
    category_keywords = set(category.split('_'))
    return any(obj in category_keywords for obj in large_man_made_objects)


# Define keywords for each specific group
dog_keywords = {'dog', 'canine', 'puppy'}
cat_keywords = {'cat', 'feline', 'kitten'}
flower_keywords = {'flower', 'blossom', 'floral'}
bird_keywords = {'bird', 'avian'}
car_keywords = {'car', 'automobile', 'vehicle'}


def belongs_to_specific_group(category, group_keywords):
    category_keywords = set(category.lower().split('_'))
    return any(keyword in category_keywords for keyword in group_keywords)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-path",
                        type=str,
                        default=r"tmp/imagenet21k_filtered_classes_v1_200c.txt",
                        help="Path to the output file"
                        )
    parser.add_argument("-n", "--num-classes",
                        type=int,
                        default=20000,
                        help="Number of classes to select")
    return parser.parse_args()


# Main function
if __name__ == '__main__':

    args = parse_args()

    # list of categories (assuming the file exists and contains the categories)
    # imagenet21k_classes = "files/classnames/imagenet21K_classes.txt"
    # with open(imagenet21k_classes, 'r') as f:
    #     categories = f.readlines()
    # # Remove the trailing newline character
    # categories = [category.strip() for category in categories]
    data_root = "/shared-data5/guy/data/imagenet21k/imagenet21k_resized"
    splits = ["imagenet21k_val", "imagenet21k_train", "imagenet21k_small_classes"]
    classnames_to_indices = build_imagenet21k_map(data_root, splits)
    categories = list(classnames_to_indices.keys())

    # Apply the filter to the list
    # filtered_animals = [category for category in categories if is_animal(category.split(',')[0])]
    # correct such that includes all category.split(','):
    filtered_animals = [category for category in categories if any([is_animal(k) for k in category.split(',')])]

    # filtered_plants_or_flowers = [category for category in categories if is_plant_or_flower(category.split(',')[0])]
    filtered_plants_or_flowers = [category for category in categories if
                                  any([is_plant_or_flower(k) for k in category.split(',')])]
    #
    # filtered_man_made_objects = [category for category in categories if is_man_made_object(category.split(',')[0])]
    # filtered_man_made_objects = [category for category in categories if any([is_man_made_object(k) for k in category.split(',')])]
    # filtered_man_made_objects = [category for category in categories if is_large_man_made_object(category.split(',')[0])]
    filtered_man_made_objects = [category for category in categories if
                                 any([is_large_man_made_object(k) for k in category.split(',')])]

    # load categories from file
    ethan_classes_path = "files/classnames/ImageNet1k/only_first_400_without_dogs120.txt"
    with open(ethan_classes_path, 'r') as f:
        ethan_categories_280 = f.readlines()
    # Remove the trailing newline character
    ethan_categories_280 = [category.strip() for category in ethan_categories_280]

    dogs_120_path = "files/classnames/Dogs120/dogs120.txt"
    with open(dogs_120_path, 'r') as f:
        dogs120_categories = f.readlines()
    # Remove the trailing newline character
    dogs120_categories = [category.strip() for category in dogs120_categories]

    flowers_path = "files/classnames/FLOWERS102/flowers102.txt"
    with open(flowers_path, 'r') as f:
        flowers_102 = f.readlines()
    # Remove the trailing newline character
    flowers_102 = [category.strip() for category in flowers_102]

    # category_file3 = "files/classnames/ImageNet21k/imagenet21k_filtered_classes_v3_300c.txt"
    # with open(category_file3, 'r') as f:
    #     file3_categories = f.readlines()
    # # Remove the trailing newline character
    # file3_categories = [category.strip() for category in file3_categories]

    # Apply the filter to the list for each group
    filtered_dogs = [category for category in categories if belongs_to_specific_group(category, dog_keywords)]
    filtered_cats = [category for category in categories if belongs_to_specific_group(category, cat_keywords)]
    filtered_flowers = [category for category in categories if belongs_to_specific_group(category, flower_keywords)]
    filtered_birds = [category for category in categories if belongs_to_specific_group(category, bird_keywords)]
    filtered_cars = [category for category in categories if belongs_to_specific_group(category, car_keywords)]

    # Select random 10 categories from each filtered list and print them
    verbose = False
    if verbose:
        print(f"Animals: \n{random.sample(filtered_animals, 10)}\n")
        print(f"Plants: \n{random.sample(filtered_plants_or_flowers, 10)}\n")
        print(f"Man-made objects: \n{random.sample(filtered_man_made_objects, 10)}\n")
        print(f"Dogs: \n{random.sample(filtered_dogs, min(10, len(filtered_dogs)))}\n")
        print(f"Cats: \n{random.sample(filtered_cats, min(10, len(filtered_cats)))}\n")
        print(f"Flowers: \n{random.sample(filtered_flowers, min(10, len(filtered_flowers)))}\n")
        print(f"Birds: \n{random.sample(filtered_birds, min(10, len(filtered_birds)))}\n")
        print(f"Cars: \n{random.sample(filtered_cars, min(10, len(filtered_cars)))}\n")

    # take the union of all the animal and plant categories, and exclude dogs
    # filtered_classes = (set(filtered_animals +
    #                         filtered_plants_or_flowers +
    #                         filtered_cats +
    #                         filtered_flowers +
    #                         filtered_birds +
    #                         filtered_dogs +
    #                         ethan_categories_280)
    #                     # -set(file2_categories)
    #                     - set(flowers_102)
    #                     #       #file3_categories)
    #                     )
    # filtered_classes = (set(filtered_animals +
    #                         filtered_plants_or_flowers +
    #                         filtered_cats +
    #                         filtered_flowers +
    #                         filtered_birds)
    #                     #-set(file2_categories)
    #                      - set(file1_categories +
    #                            file2_categories)
    #                     #       #file3_categories)
    #                     )
    filtered_classes = (set(filtered_man_made_objects))

    num_classes_to_select = min(args.num_classes, len(filtered_classes))
    print(f"Selected {num_classes_to_select} classes from the filtered list of {len(filtered_classes)} classes\n")
    # select num_classes_to_select random classes from the filtered list
    filtered_classes = random.sample(list(filtered_classes), num_classes_to_select)

    data_root = "/shared-data5/guy/data/imagenet21k/imagenet21k_resized"
    splits = ["imagenet21k_val", "imagenet21k_train", "imagenet21k_small_classes"]
    classnames_to_indices = build_imagenet21k_map(data_root, splits)
    # write the filtered categories to a file
    # out_file = "files/classnames/ImageNet21k/imagenet21k_filtered_classes_v1_200c.txt"
    out_file = args.out_path
    with open(out_file, 'w') as f:
        for category in filtered_classes:
            # if several names exits for the same category, take the first one
            # category = category.split(',')[0]
            f.write(category + '\n')
    print(f"Filtered classes written to {out_file}")
    out_file_single = out_file.replace(".txt", "single_name.txt")
    with open(out_file_single, 'w') as f:
        for category in filtered_classes:
            # if several names exits for the same category, take the first one
            category = category.split(',')[0]
            f.write(category + '\n')
    print(f"Filtered classes written to {out_file_single}")
    out_file_indx = out_file.replace(".txt", "indx.txt")
    with open(out_file_indx, 'w') as f:
        for category in filtered_classes:
            f.write(classnames_to_indices[category]['path'] + '\n')
    print(f"Filtered classes written to {out_file_indx}")
