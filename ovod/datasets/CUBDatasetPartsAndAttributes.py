import os

import torch
import yaml
from torchvision import datasets

from ovod.datasets.CUBDataset import CUBDataset
from ovod.datasets.CUBDatasetLinearProbing import CubDatasetLinearProbing, get_general_attributes_dictionary

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


def get_list_of_images_and_their_attributes(path, index_to_attribute_name, attribute_to_find_in_dataset):
    assert os.path.exists(path)
    images_to_attribute = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                img_index, att_index, exists, certainty, time_it_took = line.split(' ')
                att_cls_name = index_to_attribute_name[att_index][0]
                is_relevant_attribute = att_cls_name == attribute_to_find_in_dataset or \
                                        not attribute_to_find_in_dataset
                part = get_part_name_of_attribute_type(att_cls_name)
                if is_relevant_attribute and exists == '1' and part is not None:
                    key = f'{img_index}_{att_cls_name}'
                    att_instance = index_to_attribute_name[att_index][1]
                    if key not in images_to_attribute:
                        images_to_attribute[key] = {'img_index': img_index,
                                                    'att_class_name': att_cls_name,
                                                    'pos_atts': [],
                                                    'part': part}
                    images_to_attribute[key]['pos_atts'].append(att_instance)
            except Exception as e:
                if 'too many values' in str(e):
                    pass
                else:
                    raise e
    return images_to_attribute


MAX_ATTRIBUTE_OPTIONS = 30


class CubDatasetPartsAndAttributes(CubDatasetLinearProbing):
    def __init__(self, root, ATTRIBUTES_DATASET_PATH, **args):
        self.attributes_dataset_path = ATTRIBUTES_DATASET_PATH
        super().__init__(root, **args)

    def read_attributes_dataset(self):
        assert os.path.exists(self.attributes_dataset_path), f'Path {self.attributes_dataset_path} does not exist'
        with open(self.attributes_dataset_path, 'r') as f:
            self.attributes_dataset = yaml.load(f, Loader=yaml.FullLoader)

    def __getitem__(self, item):
        d = self.attributes_dataset[item]
        img_index = int(d['img_index'])
        total_attributes = self.attributes[d['att_class_name']]
        # Convert it to hot encoding

        label = torch.Tensor([1 if att in d['pos_atts'] else 0 for att in total_attributes])
        # Pad label to MAX_ATTRIBUTE_OPTIONS
        label = torch.nn.functional.pad(label, (0, MAX_ATTRIBUTE_OPTIONS - len(label)), mode='constant', value=0)

        file_path = self.imgs[img_index][0]

        part = d['part']
        pil_image, _ = super(CUBDataset, self).__getitem__(img_index)
        att_title = d['att_class_name']
        return pil_image, att_title, file_path, part, total_attributes, label

    @classmethod
    def create_attribute_dataset(cls, root, relevant_att_name=None):
        # If relevant_att_name is None, then we create a dataset for all attributes
        attributes_type_list_path = os.path.join(root, '..', 'attributes.txt')
        image_to_attributes_path = os.path.join(root, 'attributes', 'image_attribute_labels.txt')
        assert os.path.exists(attributes_type_list_path)
        assert os.path.exists(image_to_attributes_path)
        attributes, index_to_attribute_name = get_general_attributes_dictionary(attributes_type_list_path)
        image_to_attributes = get_list_of_images_and_their_attributes(image_to_attributes_path, index_to_attribute_name,
                                                                      relevant_att_name)
        return image_to_attributes
