import os
import glob
from .BasePrompter import BasePrompter



def is_part_of_list(sentence):
    # Check if the sentence starts with a hyphen
    if sentence.startswith('-'):
        return True

    # Check if the sentence starts with a number followed by a period
    if any(sentence.startswith(f"{i}") for i in range(1, 9)):
        return True
    return False

class ObjectProgramPrompter(BasePrompter):
    def __init__(self, base_folder, include_object_name: bool = True, description_sep=''):
        super().__init__(base_folder)  # Calling the BasePrompter's __init__ method
        self.include_object_name = include_object_name
        self.instruction = self.extract_prompt_from_files(self.base_folder)
        self.description_sep = description_sep

    def include_object_name_in_prompt(self) -> bool:
        return self.include_object_name

    def get_llm_params(self):
        params = {
            'model': 'gpt-3.5-turbo',
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'temperature': 0.5,
            'top_p': 1.0,
            'max_tokens': 2 ** 10,
            'n': 1,
            'stop': None
        }
        return params

    def extract_prompt_from_files(self, prompt_folder: str):
        system_prompt_path = os.path.join(prompt_folder, "system.prompt")
        with open(system_prompt_path) as f:
            system_prompt = f.read()

        initial_prompt_path = os.path.join(prompt_folder, "init.prompt")
        with open(initial_prompt_path) as f:
            initial_prompt = f.read()

        final_prompt_path = os.path.join(prompt_folder, "final.prompt")
        with open(final_prompt_path) as f:
            final_prompt = f.read()

        prompt = {
            'system': system_prompt,
            'initial': initial_prompt,
            'path': prompt_folder,
            'final': final_prompt
        }
        return prompt

    def get_prompt(self, text_query,*args, **kwargs):  # object_name, description, prompt):

        description = text_query

        messages = [{"role": "system", "content": self.instruction['system']}]
        if len(self.instruction['initial']) > 0:
            messages.append({"role": "user", "content": self.instruction['initial']})
        # get lists of files in the directory
        prompts_files = glob.glob(os.path.join(self.instruction['path'], "*.txt"))
        for i in range(1, len(prompts_files)+ 1):
            query_path = os.path.join(self.instruction['path'], f"query_{i}.txt")
            func_path = os.path.join(self.instruction['path'], f"func_{i}.py")
            with open(query_path) as f:
                query = f.read()
            with open(func_path) as f:
                func = f.read()
            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": func})
        # final_prompt = '\n'.join(description) + "\n" + self.instruction["final"].format(object_name)
        final_prompt = "\n" + self.instruction["final"] + self.description_sep.join(description)
        messages.append({"role": "user", "content": final_prompt})

        return messages

    def add_object_name(self, messages, object_name):
        for message in messages:
            if message["role"] == "user":
                message["content"] = message["content"].format(object_name)
        return messages

    def convert_answer_to_description(self,answer):
        sentences = answer.split('\n')
        filtered_sentences = []
        for sentence in sentences:
            if is_part_of_list(sentence):
                filtered_sentences.append(sentence)
        return filtered_sentences
