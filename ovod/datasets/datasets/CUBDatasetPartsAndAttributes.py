import os
import yaml
from torchvision import datasets

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
    attribute_index_to_data = {}
    index_to_attribute_name = {}
    with open(path, 'r') as file:
        for line in file:
            att_index, att_data = line.split(' ')
            att_type, att_name = att_data.split('::')
            att_name = att_name.replace('\n', '').replace('_', ' ')
            if att_type not in attributes:
                part = get_part_name_of_attribute_type(att_type)
                if not part:
                    # It is not a specific part of the bird so we don't use it.
                    continue
                attributes[att_type] = {}
                attributes[att_type]['part'] = part
                attributes[att_type]['attributes'] = []
            attributes[att_type]['attributes'].append(att_name)
            index_to_attribute_name[att_index] = (att_type, att_name)
    for index, (att_type, att_name) in index_to_attribute_name.items():
        neg_attributes = list(filter(lambda x: x != att_name, attributes[att_type]['attributes']))
        data = {
            'pos_attribute': att_name,
            'neg_attributes': neg_attributes,
            'part': attributes[att_type]['part']
        }
        attribute_index_to_data[index] = data
    return attribute_index_to_data


def get_list_of_images_and_their_attributes(path, attribute_index_to_data):
    assert os.path.exists(path)
    images_and_attributes = []
    with open(path, 'r') as f:
        for line in f:
            try:
                img_index, att_index, exists, certainty, time_it_took = line.split(' ')
                if exists == '1' and att_index in attribute_index_to_data:
                    data = attribute_index_to_data[att_index].copy()
                    data['img_index'] = img_index
                    data['certainty'] = certainty
                    images_and_attributes.append(data)
            except Exception as e:
                print(f'Coudn\'t proccess line {line}')
                print(e)
    return images_and_attributes


class CubDatasetsPartsAndAttributes(CUBDataset):
    def __init__(self, root, transform=None, target_transform=None, loader=datasets.folder.default_loader,
                 is_valid_file=None, train=True, bboxes=False, list_of_categories=[], ignore_indices_to_use=False
                 , APPLY_NORMALIZE_TRANSFORM=True, ATTRIBUTES_DATASET_PATH=None, **args):
        super().__init__(root, transform, target_transform, loader, is_valid_file, train, bboxes, list_of_categories,
                         ignore_indices_to_use, APPLY_NORMALIZE_TRANSFORM)
        assert ATTRIBUTES_DATASET_PATH and os.path.exists(
            ATTRIBUTES_DATASET_PATH), "Please use the create_cub_attributes_dataset " \
                                      "before Running experiments with the CUbPartsAndAttributes. " \
                                      "This dataset demands creating attribute cache before."
        self.attributes_dataset = None
        print(f'Starting to load attributes file from {ATTRIBUTES_DATASET_PATH}')
        with open(ATTRIBUTES_DATASET_PATH) as f:
            self.attributes_dataset = yaml.full_load(f)
        print(f'Finished reading {len(self.attributes_dataset)} elements.')
        pass

    def __len__(self):
        return len(self.attributes_dataset)

    def __getitem__(self, item):
        d = self.attributes_dataset[item]
        img_index = int(d['img_index'])
        pos_attribute = d['pos_attribute']
        neg_attributes = d['neg_attributes']
        part = d['part']
        file_path = self.imgs[img_index][0]
        pil_image, target = super(CUBDataset, self).__getitem__(img_index)
        image_for_clip, _, _, _ = super(CubDatasetsPartsAndAttributes, self).__getitem__(img_index)
        return pil_image, image_for_clip, file_path, part, pos_attribute, neg_attributes

    @classmethod
    def create_attribute_dataset(cls, root):
        attributes_type_list_path = os.path.join(root, '..', 'attributes.txt')
        image_to_attributes_path = os.path.join(root, 'attributes', 'image_attribute_labels.txt')
        assert os.path.exists(attributes_type_list_path)
        assert os.path.exists(image_to_attributes_path)
        attribute_index_to_data = get_general_attributes_dictionary(attributes_type_list_path)
        image_to_attributes = get_list_of_images_and_their_attributes(image_to_attributes_path,
                                                                      attribute_index_to_data)
        return image_to_attributes


if __name__ == '__main__':
    root = r'/shared-data5/guy/data/CUB/CUB_200_2011'
    CubDatasetsPartsAndAttributes(root)
