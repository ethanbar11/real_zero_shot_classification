from ovod.prompters.ObjectProgramSelfImprovingV1 import ObjectProgramSelfImprovingV1
import random


class ObjectProgramSelfImprovingWithActions(ObjectProgramSelfImprovingV1):
    def __init__(self, base_folder, include_object_name: bool = True, description_sep='\n'):
        super().__init__(base_folder, include_object_name, description_sep)
        self.actions = ['Create a more interesting program based on the description.',
                        'Use for loops and try to average over descriptions']

    def convert_context_step_into_message(self, context_step, category, is_new_message=False):
        action_index = random.randint(0, len(self.actions))
        if action_index == 0 or not is_new_message:
            return super().convert_context_step_into_message(context_step, category)
        else:
            return self.actions[action_index - 1]
