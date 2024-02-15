import os
import sys
import timeit

from transformers import GenerationConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ovod.llm_wrappers.llama_2_code_phind import CodeLLamaV2Phind

if __name__ == '__main__':

    prompt_path = r'./prompt.txt'
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    model = r'/shared-data5/guy/modelzoo/llama2/Phind-CodeLlama-34B-v2'
    output_directory = r'./phind_tries'
    # If directory doesn't exist, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    load_in_8bit = True
    gen_config = GenerationConfig(max_new_tokens=1024, min_new_tokens=1, top_p=0.95, temperature=0.7, do_sample=True,
                                  top_k=40, repetition_penalty=1.0)
    model = CodeLLamaV2Phind(model, load_in_8bit, gen_config.max_new_tokens, gen_config.min_new_tokens,
                             gen_config.top_p, gen_config.temperature, 'auto', gen_config.top_k, gen_config.do_sample,
                             gen_config.repetition_penalty)
    temperatures = [1.0, 1.5, 1.2, 1.0, 0.7, 0.5, 0.2]
    rep_penalty = [1.0, 1.1, 1.2]
    for rep in rep_penalty:
        for temp in temperatures:
            gen_config.temperature = temp
            gen_config.repetition_penalty = rep
            output_path = os.path.join(output_directory, f'temp_{temp}_rep_{rep}.txt')
            if os.path.exists(output_path):
                print('Skipping experiment for ', output_path, 'as it already exists')
                continue

            model.generation_config = gen_config
            print('Starting experiment for ', output_path)
            start_time = timeit.default_timer()
            code = model.forward_prompt(prompt)
            print('Finished experiment for ', output_path, ' in ', timeit.default_timer() - start_time)
            # Save the code to a file
            with open(output_path, 'w') as f:
                f.write(code)
