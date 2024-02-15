import clip
import numpy as np
import os
import yaml
from torchvision import datasets
import torch
from ovod.datasets.CUBDataset import CUBDataset

bird_parts = ['bill', 'belly', 'breast', 'forehead', 'nape', 'leg', 'wing', 'tail',
              'throat', 'head']

conversion_dict = {'bill': 'beak',
                   'throat': 'neck',
                   'breast': 'torso',
                   'belly': 'torso',
                   'forehead': 'head',
                   'nape': 'throat'
                   }


def get_part_name_of_attribute_type(att_type):
    for part in bird_parts:
        if part in att_type:
            if part in conversion_dict:
                return conversion_dict[part]
            return part
    return None


def get_general_attributes_dictionary(path):
    assert os.path.exists(path)
    attributes = {}
    index_to_attribute_name = {}
    with open(path, 'r') as file:
        for line in file:
            att_index, att_data = line.split(' ')
            att_type, att_name = att_data.split('::')
            att_name = att_name.replace('\n', '').replace('_', ' ')
            if att_type not in attributes:
                attributes[att_type] = []
            attributes[att_type].append(att_name)
            index_to_attribute_name[att_index] = (att_type, att_name)
    return attributes, index_to_attribute_name


def get_list_of_images_and_their_attributes(path, attributes, index_to_attribute_name, attribute_cls_name):
    assert os.path.exists(path)
    images_to_attribute = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                img_index, att_index, exists, certainty, time_it_took = line.split(' ')
                if index_to_attribute_name[att_index][0] == attribute_cls_name and exists == '1':
                    key = f'{img_index}_{attribute_cls_name}'
                    att_instance = index_to_attribute_name[att_index][1]
                    if key not in images_to_attribute:
                        images_to_attribute[key] = {'img_index': img_index,
                                                    'att_class_name': attribute_cls_name,
                                                    'pos_atts': []}
                    images_to_attribute[key]['pos_atts'].append(att_instance)
            except Exception as e:
                print(f'Coudn\'t proccess line {line}')
                print(e)
    return images_to_attribute


class CubDatasetOneAttribute(CUBDataset):
    def __init__(self, root, transform=None, target_transform=None, loader=datasets.folder.default_loader,
                 is_valid_file=None, train=True, bboxes=False, list_of_categories=[], ignore_indices_to_use=False
                 , APPLY_NORMALIZE_TRANSFORM=True, ATTRIBUTES_DIRECTORY=None, ATTRIBUTE_NAME=None, **args):
        super().__init__(root, transform, target_transform, loader, is_valid_file, train, bboxes, list_of_categories,
                         ignore_indices_to_use, APPLY_NORMALIZE_TRANSFORM)
        assert ATTRIBUTES_DIRECTORY and os.path.exists(
            ATTRIBUTES_DIRECTORY), "Please use the create_cub_attribute_dataset_for_one.py " \
                                   "before Running experiments with the CubDatasetOneAttribute. " \
                                   "This dataset demands creating attribute cache before, for a specific attribute."
        assert ATTRIBUTE_NAME, 'Please specify an attribute name to check on from CUB.'
        self.attribute_cls = ATTRIBUTE_NAME
        path = os.path.join(ATTRIBUTES_DIRECTORY, f'{ATTRIBUTE_NAME}.yaml')
        self.attributes_dataset = None
        print(f'Starting to load attributes file from {path}')
        with open(path) as f:
            self.attributes_dataset = yaml.full_load(f)
        print(f'Finished reading {len(self.attributes_dataset)} elements.')
        self.attributes_dataset = list(self.attributes_dataset.values())
        self.attributes_dataset = list(filter(lambda x: int(x['img_index']) in self.indices_to_use, self.attributes_dataset))

        attributes_type_list_path = os.path.join(root, '..', 'attributes.txt')
        self.attributes, self.index_to_attribute_name = get_general_attributes_dictionary(attributes_type_list_path)
        self.label_size = len(self.attributes[self.attribute_cls])

    def __len__(self):
        return len(self.attributes_dataset)

    def __getitem__(self, item):
        d = self.attributes_dataset[item]
        total_attributes = self.attributes[d['att_class_name']]
        # Convert it to hot encoding
        label = torch.Tensor([1 if att in d['pos_atts'] else 0 for att in total_attributes])

        img_index = int(d['img_index'])
        pil_image, target = super(CUBDataset, self).__getitem__(img_index)
        image = self.transform_(pil_image)

        return image, label

    @classmethod
    def create_attribute_dataset(cls, root, att_name):
        attributes_type_list_path = os.path.join(root, '..', 'attributes.txt')
        image_to_attributes_path = os.path.join(root, 'attributes', 'image_attribute_labels.txt')
        assert os.path.exists(attributes_type_list_path)
        assert os.path.exists(image_to_attributes_path)
        attributes, index_to_attribute_name = get_general_attributes_dictionary(attributes_type_list_path)
        image_to_attributes = get_list_of_images_and_their_attributes(image_to_attributes_path, attributes,
                                                                      index_to_attribute_name, att_name)
        return image_to_attributes


if __name__ == '__main__':
    root = r'/shared-data5/guy/data/CUB/CUB_200_2011'
    att_name = 'has_back_color'
    CubDatasetOneAttribute.create_attribute_dataset(root, att_name)
