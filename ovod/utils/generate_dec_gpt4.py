import os
import openai
import json
import timeit
import random

def get_api_key_from_file(filename):
    with open(filename, 'r') as f:
        content = f.readline().strip()
        # Split the line on the equals sign and retrieve the part after it
        # Then strip the quotes and split again to discard the comment
        key = content.split('=')[1].split('#')[0].strip()[1:-1]

        return key

def get_response_from_gpt4_no_context(instruction, input_text, context):
    response = openai.ChatCompletion.create(
        model="gpt-4",     # "gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{instruction}\n{input_text}"}
        ]
    )
    return response.choices[0]["message"]["content"]


def get_response_from_gpt4(instruction, input_text, context):
    messages = []
    for v in context:
        messages.append({"role": "user", "content": f"{v['instruction']}\n{v['input']}"})
        messages.append({"role": "assistant", "content": v['output']})
    messages.append({"role": "user", "content": f"{instruction}\n{input_text}"})

    response = openai.ChatCompletion.create(
        model="gpt-4",     # "gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0]["message"]["content"] #.text.strip()


def get_dummy_response_from_gpt4():
    response = openai.ChatCompletion.create(
        model="gpt-4",     # "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
    )
    return response.choices[0]["message"]["content"] #.text.strip()


if __name__ == "__main__":
    os.environ['REQUESTS_CA_BUNDLE'] = r"/etc/ssl/certs/ca-certificates.crt"
    # Read the API key from the file named "API.KEY"
    openai.api_key = get_api_key_from_file("api.key")

    train_filepath = "files/annots/explanation_transcript_2208_train_1_1.json"
    test_filepath = "files/annots/explanation_transcript_2208_test.json"
    num_examples_for_context = 10
    output_folder = "/shared-data5/guy/exps/echograph_nle/logs"

    # read content from in_context_filepath:
    with open(train_filepath, 'r') as f:
        context = json.load(f)
    # select random examples from context:
    random.shuffle(context)
    context = context[:num_examples_for_context]

    # instruction = "Explain why the ejection fraction is estimated as 49%."
    # input_text = ("In the echocardiography image, it is measured that there could be a small septal bulge, "
    #               "the shape of the left ventricle looks round, the movement of the segments is normal, "
    #               "the image quality is normal, the basal points move normal, the left ventricle is  fully visible, "
    #               "and the apex moves normal. The bulge value is 863.0, the height over width value is 1.00, "
    #               "the segment movement is normal, the apex moves 9.01%, the basal points move by 4.77%, contrast is 1.67.")

    with open(test_filepath, 'r') as f:
        test = json.load(f)

    # write response to json file:
    results = dict()

    # time the response:
    start = timeit.default_timer()
    for ii, v in enumerate(test):
        instruction = v['instruction']
        input_text = v['input']
        gt = v['output']
        response = get_response_from_gpt4(instruction, input_text, context)
        stop = timeit.default_timer()
        time_elapsed = stop - start

        results[f"test_example_{ii}"] = {"instruction": instruction, "input": input_text, gt: gt, "response": response}

        #response = get_dummy_response_from_gpt4()
        print(response)
        print(f"Time for response: {time_elapsed:.2f} seconds")

    out_filename = os.path.join(output_folder, f"GPT4_CoT_train_{os.path.splitext(os.path.basename(train_filepath))[0]}_test_{os.path.splitext(os.path.basename(test_filepath))[0]}_results.json")
    with open(out_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"file was saved to {out_filename}")
