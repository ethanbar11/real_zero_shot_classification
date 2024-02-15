import json
from typing import List
import random

import openai
#from openai.error import RateLimitError
import os
import time

from ovod.utils.class_name_fix import fix_classname_special_chars
from projects.grounding.utils.dataitems import create_data_list_from_file

os.environ['REQUESTS_CA_BUNDLE'] = r"/etc/ssl/certs/ca-certificates.crt"
with open('api.key') as f:
    openai.api_key = f.read().strip()


def openai_request(messages, prompt_params):
    valid = False
    response = None
    print('Sending request to OpenAI API')
    while not valid:
        try:
            response = openai.ChatCompletion.create(
                model=prompt_params['llm_model'],
                messages=messages,
                temperature=prompt_params['temperature'],  # 0.99,
                max_tokens=prompt_params['max_tokens'],  # 50,
                n=prompt_params['n'],  # 10,
            )
            valid = True
        except (Exception, openai.error.Timeout) as e:
            time_to_sleep = 30
            # if type(e) == RateLimitError:
            #     print(f"Hit rate limit. Waiting {time_to_sleep} seconds.")
            # else:
            #     print(f"Timeout error. Waiting {time_to_sleep} seconds.")
            time.sleep(time_to_sleep)
        time.sleep(0.15)
    output = response.choices[0]['message']['content']
    return output


def parse_ann_file(filepath: str, limit: int = 0) -> List:
    if "lvis" in filepath:
        lvis_gt = LVIS(filepath)
        categories = sorted(lvis_gt.cats.values(), key=lambda x: x["id"])

        category_list = [c['synonyms'][0].replace('_', ' ') for c in categories]

    elif "activity" in filepath:
        with open(filepath, "r") as f:
            categories = json.loads(f.read())
        category_list = [c["nodeName"] for c in categories["taxonomy"]]

    else:
        category_list = create_data_list_from_file(textfile=filepath)

    if limit > 0:
        random.shuffle(category_list)
        category_list = category_list[limit]

    # category_list = [fix_cat(c) for c in category_list]

    return category_list
