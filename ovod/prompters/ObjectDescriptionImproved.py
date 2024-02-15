import os
import glob

from .ObjectDescriptionPrompter import filter_name_from_description
from .ObjectProgramPrompter import ObjectProgramPrompter


class ObjectDescriptionImproved(ObjectProgramPrompter):
    def __init__(self, base_folder, include_object_name: bool = True):
        super().__init__(base_folder, include_object_name, description_sep='')

    def get_llm_params(self):
        params = {
            'model': 'gpt-4',
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'temperature': 0.15,
            'top_p': 1.0,
            'max_tokens': 2 ** 10,
            'n': 1,
            'stop': None
        }
        return params