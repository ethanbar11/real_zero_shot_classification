import os
import glob
from .ObjectProgramPrompter import ObjectProgramPrompter

import ovod.utils.logger as logging

logger = logging.get_logger(__name__)


class ObjectProgramSelfImprovingV1(ObjectProgramPrompter):

    def extract_prompt_from_files(self, prompt_folder: str):
        system_prompt_path = os.path.join(prompt_folder, "system.prompt")
        with open(system_prompt_path) as f:
            system_prompt = f.read()

        initial_prompt_path = os.path.join(prompt_folder, "init.prompt")
        with open(initial_prompt_path) as f:
            initial_prompt = f.read()

        #final_prompt_path = os.path.join(prompt_folder, "final.prompt")
        final_prompt_path = os.path.join(prompt_folder, "final_refinement.prompt")
        with open(final_prompt_path) as f:
            final_prompt = f.read()

        #standard_final = os.path.join(prompt_folder, "standard_final.prompt")
        standard_final = os.path.join(prompt_folder, "final.prompt")
        with open(standard_final) as f:
            standard_final_prompt = f.read()

        prompt = {
            'system': system_prompt,
            'initial': initial_prompt,
            'path': prompt_folder,
            'final': final_prompt,
            'standard_final': standard_final_prompt
        }
        return prompt

    def convert_context_step_into_message(self, context_step, category, is_new_message=False):
        results = context_step['confusion_matrix'][context_step['confusion_matrix'].index == category]
        confusion_matrix = context_step['confusion_matrix']
        FN = confusion_matrix.loc[category].sum() - confusion_matrix.at[category, category]
        FP = confusion_matrix[category].sum() - confusion_matrix.at[category, category]
        if FN > FP:
            # That means that mostly the category is being assigned to other category
            # Drop the category from the results

            other_biggest_category = confusion_matrix.drop(columns=[category]).loc[category].idxmax()
            description = list(filter(lambda x: x[0] == other_biggest_category, context_step['descriptions']))[0][1][0]
            description_text = """I\'m using other programs for other types of birds as well, based on their descriptions.
                        a program based on the following description, received higher score than a program for
                        your description :
                        {}
                        """.format(description)
        else:
            # That means that mostly other categories are being assigned to this category
            # Drop the category from the results
            other_biggest_category = confusion_matrix.drop(index=[category]).loc[:, category].idxmax()
            description = list(filter(lambda x: x[0] == other_biggest_category, context_step['descriptions']))[0][1][0]
            description_text = """I\'m using other programs for other types of birds as well, based on their descriptions.
                                    a program based on the following description, received higher score for the program
                                    you wrote than for her own. description :
                                    {}
                                    """.format(description)
        return self.instruction['final'].format(FP, FN, description_text)

    def get_prompt(self, description, context=None, category=None):  # object_name, description, prompt):
        """
        description: The raw description of the object
        data: Should be structured in the following manner:
        data[
        """
        ## Add starting messages
        messages = [{"role": "system", "content": self.instruction['system']}]
        if len(self.instruction['initial']) > 0:
            messages.append({"role": "user", "content": self.instruction['initial']})
        # get lists of files in the directory
        prompts_files = glob.glob(os.path.join(self.instruction['path'], "*.txt"))
        amount_of_standard_queries = max(len(prompts_files) // 2 - len(context), 0)
        for i in range(1, amount_of_standard_queries + 1):
            query_path = os.path.join(self.instruction['path'], f"query_{i}.txt")
            func_path = os.path.join(self.instruction['path'], f"func_{i}.txt")
            with open(query_path) as f:
                query = f.read()
            with open(func_path) as f:
                func = f.read()
            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": func})
        final_prompt = "\n" + self.instruction["standard_final"] + self.description_sep.join(description)
        messages.append({"role": "user", "content": final_prompt})

        for idx, context_step in enumerate(context[-2:]):
            func = context_step['programs'][category.lower().replace(' ', '_')]
            messages.append({"role": "assistant", "content": func})
            is_new_message = True if idx == len(context) - 1 else False
            messages.append({"role": "user",
                             "content": self.convert_context_step_into_message(context_step, category, is_new_message)})
        logger.debug(f'Using the following action : {messages[-1]["content"]}')
        return messages

    def add_object_name(self, messages, object_name):
        for message in messages:
            if message["role"] == "user":
                message["content"] = message["content"].format(object_name)
        return messages
