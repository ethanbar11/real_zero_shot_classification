from .ObjectDescriptionPrompter import ObjectDescriptionPrompter
from .ObjectDescriptionPrompter import filter_name_from_description


class OxfordObjectDescriptionPrompter(ObjectDescriptionPrompter):
    def __init__(self, base_folder, include_object_name: bool = True):
        super().__init__(base_folder, include_object_name)  # Calling the parent Prompter class __init__ method
        self.vowel_list = ['a', 'e', 'i', 'o', 'u']

    def get_prompt(self, text_query):

        category = text_query

        if category[0] in self.vowel_list:
            article = 'an'
        else:
            article = 'a'
        prompts = []
        prompt = f"Describe what {article} {category} looks like."
        if not self.include_object_name:
            prompt += (f" Do not use the latin name for '{category}'."
                       f" Do not use other known nicknames for '{category}'."
                       f" If you use the `{category}` in your description, make sure to spell it exactly as shown."
                       )
        return [prompt]

    def get_llm_params(self):
        params = {
            'llm_model': 'text-davinci-003',
            'temperature': 0.99,
            'max_tokens': 50,
            'top_p': 1.0,
            'n': 10,
            'stop': '.',
            "frequency_penalty": 1.0,
            'presence_penalty': 1.0,
            "model": "gpt-3.5-turbo-instruct"
        }
        return params

    @staticmethod
    def filter_name_from_description(description: str, category: str):
        return filter_name_from_description(description, category)


class OxfordObjectDescriptionPrompterIN21k(OxfordObjectDescriptionPrompter):
    def get_prompt(self, text_query):
        category = text_query
        different_names = category.split(', ')
        if category[0] in self.vowel_list:
            article = 'an'
        else:
            article = 'a'
        prompts = []
        prompt = f"The following are different name for the same object: {category}. Describe what it looks like."
        if not self.include_object_name:
            prompt += (f" Do not use the latin name for '{category}'."
                       f" Do not use other known nicknames for '{category}'."
                       # f" If you use the `{category}` in your description, make sure to spell it exactly as shown."
                       )
        return [{'role': 'user', 'content': super().get_prompt(text_query)[0]}]

    def get_llm_params(self):
        params = {
            'model': 'gpt-3.5-turbo-1106',
            'temperature': 1.5,
            'max_tokens': 50,
            'top_p': 1.0,
            'stop': '.',
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        return params
