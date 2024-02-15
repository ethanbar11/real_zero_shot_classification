import os
import time
import json
import ovod.utils.logger as logging

logger = logging.get_logger(__name__)


def generate_object_code_by_description(object_name, description, prompter, llm_wrapper, context=None, category=None,
                                        failed_attempts_so_far=0):
    messages = prompter.get_prompt(description, context, category)
    messages = prompter.add_object_name(messages, object_name)
    logger.debug(f'Starting to create program out of description using {llm_wrapper.__class__.__name__}')
    code = llm_wrapper.forward(messages, failed_attempts_so_far)
    return code, messages


def generate_programs_from_descriptions(prompter, descriptions, output_dir, classifier=None, llm_wrapper=None,
                                        image_example=None, context=None, overwrite=False, website_category_hack=''):
    """ Generate programs for all objects in the dataset, given their descriptions """
    # model params:
    for category, description in descriptions:
        program_output_path = os.path.join(output_dir, f"{category.replace(' ', '_').lower()}")
        if overwrite or not os.path.exists(f"{program_output_path}.py"):
            # get code
            # time this section:
            start = time.time()
            is_valid_code = False
            tries = 0
            while not is_valid_code and tries < llm_wrapper.max_creation_times:
                code, messages = generate_object_code_by_description(object_name=category,
                                                                     description=description,
                                                                     prompter=prompter,
                                                                     llm_wrapper=llm_wrapper,
                                                                     context=context,
                                                                     category=category,
                                                                     failed_attempts_so_far=tries)
                if classifier:
                    is_valid_code = classifier.validate_program_is_valid(code, image_example)
                    tries += 1
                else:
                    # Means we don't check for validity
                    is_valid_code = True
            if tries == llm_wrapper.max_creation_times and not is_valid_code:
                raise Exception(f"Program for {category} could not be generated.")
            # Check code is valid.
            end = time.time()

            tries, is_valid_code = 0, False
            if website_category_hack:
                print('Applying website hack.. changing exists_full to include category name')
                messages.append({'role': 'user',
                                 'content': f"I have a generated python code that I need to fix. "
                                            f"Please fix the parts_to_detect list to include {website_category_hack} parts and not a general object class ."
                                            f"Answer directly with the code, no need for any other comments. \n Python code: \n {code}"
                                 })
                while not is_valid_code and tries < llm_wrapper.max_creation_times:
                    new_code = llm_wrapper.forward(messages, failed_attempts_so_far=tries)
                    if classifier:
                        is_valid_code = classifier.validate_program_is_valid(new_code, image_example)
                        tries += 1
                    else:
                        # Means we don't check for validity
                        is_valid_code = True
                    if tries == llm_wrapper.max_creation_times and not is_valid_code:
                        raise Exception(f"Program for {category} could not be generated.")
                    else:
                        code = new_code

            # write messages as json file:
            with open(f"{program_output_path}.json", 'w') as f:
                json.dump(messages, f, indent=2)
            # write code as python file:
            with open(f"{program_output_path}.py", 'w') as f:
                f.write(code)
            logger.info(
                f"Generated program for {category} in {end - start:.2f} seconds, saved to {program_output_path}.py")
        else:
            logger.debug(f"Program for {category} already exists, skipping.")

    return f"{program_output_path}.json", f"{program_output_path}.py"
