# from typing import Any
#
# from torchvision.datasets import ImageNet
#
# class ImageNetWithPartOfClasses(ImageNet):
#     def __init__(self, root: str, **kwargs: Any):
#         print('Starting to initialize father.')
#         super().__init__(root, **kwargs)
#         pass
from projects.parts_attributes_dataset.extract_captions import build_imagenet21k_map

if __name__ == '__main__':
    # path = r'/shared-data5/ethan_baron/data/imagenet_1k'
    # ImageNetWithPartOfClasses(root = path)
    bla = build_imagenet21k_map()
    pass
