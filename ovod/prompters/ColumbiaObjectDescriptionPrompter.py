from .ObjectDescriptionPrompter import ObjectDescriptionPrompter

class ColumbiaObjectDescriptionPrompter(ObjectDescriptionPrompter):
    def __init__(self, base_folder, include_object_name: bool = True):
        super().__init__(base_folder, include_object_name)  # Calling the parent Prompter class __init__ method

    def get_prompt(self, text_query):

        category_name = text_query

        prompt = f"""
        Q: What are useful visual features for distinguishing a lemur in a photo?
        A: There are several useful visual features to tell there is a lemur in a photo:
        - four-limbed primate
        - black, grey, white, brown, or red-brown
        - wet and hairless nose with curved nostrils
        - long tail
        - large eyes
        - furry bodies
        - clawed hands and feet

        Q: What are useful visual features for distinguishing a television in a photo?
        A: There are several useful visual features to tell there is a television in a photo:
        - electronic device
        - black or grey
        - a large, rectangular screen
        - a stand or mount to support the screen
        - one or more speakers
        - a power cord
        - input ports for connecting to other devices
        - a remote control

        Q: What are useful features for distinguishing a {category_name} in a photo?
        A: There are several useful visual features to tell there is a {category_name} in a photo:
        -
        """

        prompt_gpt35 = f"""
        What are useful features for distinguishing a {category_name} in a photo?
        
        Here are few examples for you to consider: 
        Q: What are useful visual features for distinguishing a lemur in a photo?
        A: There are several useful visual features to tell there is a lemur in a photo:
        - It is a four-limbed primate
        - The color of the animal is black, grey, white, brown, or red-brown
        - The animal has wet and hairless nose with curved nostrils
        - The animal has long tail
        - The animal has large eyes
        - The animal has furry bodies
        - The animal has clawed hands and feet

        Q: What are useful visual features for distinguishing a television in a photo?
        A: There are several useful visual features to tell there is a television in a photo:
        - It is electronic device
        - The color of the device is black or grey
        - The device has a large, rectangular screen
        - The device has a stand or mount to support the screen
        - The device has one or more speakers
        - The device has a power cord
        - The device has input ports for connecting to other devices
        - The device has a remote control

        Based on these two examples, can you complete the following prompt?
        A: There are several useful visual features to tell there is a {category_name} in a photo:
        -
        """
        #return [prompt]
        return [{'role': 'user', 'content': prompt_gpt35}]

    def get_llm_params(self):
        params = {
            'model': 'gpt-3.5-turbo-1106',
            'temperature': 0.0,
            'max_tokens': 100,
            'top_p': 1.0,
            'stop': '.',
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        return params

    def convert_answer_to_description(self, answer):
        # break up the answer into sentences
        sentences = answer.split('\n')
        # if a sentence starts with "-" replace "":
        for sentence in sentences:
            if sentence.startswith("-"):
                sentence.replace("-", "")

        return sentences


