import time
from typing import List
from fvcore.common.config import CfgNode
import yaml
from ovod.llm_wrappers.build import build_llm_wrapper
from ovod.utils.vis import plot_vqa_prediction
import ovod.utils.logger as logging
from ovod.vlms.llava_client import LLaVaWrapped
from .BaseClassifier import BaseClassifier

from PIL import Image
import torch
import io
import yaml

logger = logging.get_logger(__name__)


def tensor_to_bytearray(tensor_img):
    pil_image = Image.fromarray(tensor_img.cpu().clone().detach().numpy().transpose(1, 2, 0).astype("uint8"))
    return pil_image
    # Save PIL image to a bytes buffer
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="JPEG")

    # Convert bytes buffer to base64 encoded string
    return img_buffer.getvalue()


class LLaVaGptClassifier(BaseClassifier):
    def __init__(self, openset_params: CfgNode, device: torch.device, openset_categories: List, config):
        super().__init__(openset_params, device, openset_categories)
        self.categories = openset_categories
        model_id = config.MODEL.MODEL_ID
        system_prompt_location = config.MODEL.SYSTEM_PROMPT_LOCATION
        temp = config.MODEL.TEMPERATURE
        top_p = config.MODEL.TOP_P
        max_new_tokens = config.MODEL.MAX_NEW_TOKENS
        questions_path = config.MODEL.QUESTIONS_PATH
        self.llava_wrapper = LLaVaWrapped(model_id, system_prompt_location, temp, top_p, max_new_tokens)
        with open(questions_path, 'r') as f:
            self.questions = yaml.safe_load(f)
        self.llm_wrapper = build_llm_wrapper(config)

    def convert_answers_to_llm_question(self, questions_and_answers):
        # PROMPT = 'I would like to use you as a bird identifier, from a photo.' \
        #          ' The following are questions and answers about the bird I\'m seeing, and I\'d like you to ' \
        #          'say what bird do you think it is.\n'
        PROMPT = 'I would like to use you as a bird identifier, from a photo.' \
                 ' The following are answers about visual questions for the bird I\'m seeing, and I\'d like you to ' \
                 'say what bird do you think it is.\n'


        full_string = PROMPT
        for question, answer in questions_and_answers:
            #full_string += f'question: {question}\nanswer: {answer}\n'
            full_string += f'answer: {answer}\n'

        categories_as_str = ' , '.join(self.categories) # "All 200 Bird categories in the CUB dataset" #
        # full_string += 'Based on these answers and questions, what type of bird do you think it is out of the following' \
        #                f' options: {categories_as_str}. Think carefully and describe your steps, and at the end just output the name of the bird' \
        #                f' in the following manner **BIRD NAME**'
        full_string += 'Based on these answers to visual questions, what type of bird do you think it is out of the following' \
                       f' options: {categories_as_str}. Think carefully and describe your steps, and at the end just output the name of the bird' \
                       f' in the following manner **BIRD NAME**'

        return full_string

    def forward(self, images: torch.Tensor, images_paths: List):  # , labels: torch.Tensor):
        # explanations = [{'question': 'question', 'answer': 'answer'} for _ in range(len(images))]
        # return torch.Tensor([[0 for i in range(len(self.categories))] for j in range(len(images))]),explanations
        outputs = []
        explanations = []
        for path in images_paths:
            img_as_pil = Image.open(path)

            images_duplicated = [img_as_pil for _ in range(len(self.questions))]
            questions = []
            for question in self.questions:
                question_txt = question['name']
                options = question['options']
                questions.append(question_txt)
            start = time.time()
            answers = self.llava_wrapper.forward(images_duplicated, questions)
            end = time.time()
            logger.debug(f"Time taken for LLaVa: {end - start} seconds")
            questions_and_answers = [(questions[idx], answers[idx]) for idx in range(len(questions))]
            question_for_llm = self.convert_answers_to_llm_question(questions_and_answers)

            answer = self.llm_wrapper.forward([{'role': 'user', 'content': question_for_llm}])
            bird_found = None
            bird_found_idx = None
            # try to find the bird in the answer for max 5 times
            for ii in range(5):
                answer = self.llm_wrapper.forward([{'role': 'user', 'content': question_for_llm}])
                print(f"ANSWER ATTEMPT #{ii}: /n {answer}")
                for idx, category in enumerate(self.categories):
                    if f'**{category}**'.lower() in answer.lower():
                        bird_found = category
                        bird_found_idx = idx
                        break
                if bird_found:
                    break


            # assert bird_found, 'Bird was\'nt found'
            output = [0 for i in range(len(self.categories))]
            if bird_found:
                output[bird_found_idx] = 1
            outputs.append(output)
            explanation = {'question': question_for_llm, 'answer': answer}
            explanations.append(explanation)
        return torch.Tensor(outputs), explanations

    def plot_prediction(self, img, output: dict, plot_folder: str):
        return plot_vqa_prediction(img, output, plot_folder)


if __name__ == '__main__':
    attributes_path_file = r'/shared-data5/guy/data/CUB/attributes.txt'
    classes = {}
    with open(attributes_path_file, 'r') as f:
        for line in f:
            _, x = line.split(' ')
            cls_name, cls_instance = x.split('::')
            if not cls_name in classes:
                classes[cls_name] = []
            classes[cls_name].append(cls_instance)
    questions = []
    for cls_name, instances in classes.items():
        name = cls_name.replace('has_', '').replace('_', ' ')
        question = {}
        question['name'] = f'How would you characterize the bird\'s {name}?'
        question['options'] = [instance.replace('\n', '').replace('_', ' ') for instance in instances]
        questions.append(question)
    output_path = r'/home/ethan/ge_code_that_is_cloned_here/grounding_files/questions_for_vqa/CUB_llava_questions.yaml'

    with open(output_path, 'w') as f:
        yaml.dump(questions, f)
