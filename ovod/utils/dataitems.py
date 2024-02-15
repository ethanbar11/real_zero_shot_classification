from typing import List, Dict
import json
import numpy as np
from ovod.utils.class_name_fix import fix_classname_special_chars


def create_data_list_from_file(textfile: str) -> List:
        """
        Called during construction. Creates a list containing paths to images in the dataset, given a folder path
        """
        with open(textfile) as f:
            l = f.readlines()
        data_list = []
        for filename in l:
            data_list.append(filename.strip())

        # 12-Nov adding code to fix the category name
        # data_list = [fix_classname_special_chars(c) for c in data_list]
        return data_list


