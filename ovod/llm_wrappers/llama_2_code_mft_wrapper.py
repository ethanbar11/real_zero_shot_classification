import random

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from ovod.llm_wrappers.base_llm_wrapper import LLMWrapper
import numpy as np
import ovod.utils.logger as logging

logger = logging.get_logger(__name__)


class CodeFuseWrapper(LLMWrapper):
    def __init__(self, model, load_in_8bit, max_new_tokens, min_new_tokens, top_p, temperature, device, top_k,
                 do_sample,
                 repetition_penalty, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = model
        self.load_in_8bit = load_in_8bit
        self.device = device
        self.do_sample = do_sample
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_creation_times = 10
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        # Some shit that is specific for CodeFuse, not clear why.
        self.generation_config = GenerationConfig(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                                  top_p=top_p, temperature=temperature, do_sample=self.do_sample,
                                                  top_k=top_k,
                                                  repetition_penalty=repetition_penalty)
        self.init_model()
        self.model.eval()

    def init_model(self):
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<unk>")
        self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("</s>")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, load_in_8bit=self.load_in_8bit,
                                                          device_map=self.device, trust_remote_code=True)

    def convert_messages_to_prompt(self, messages):
        final_str = ''

        for msg in messages:
            role = msg['role']
            content = msg['content']
            final_str += f"<|im_start|>{role}\n"
            final_str += f'{content}<|im_end|>\n'
        final_str += f"<|im_start|>assistant"
        return final_str

    def extract_function_of_output(self, output):
        start_index = output.find('def execute_command(img, patch):')
        end_seq = 'return attributes, special_weights'
        end_index = output.find(end_seq) + len(end_seq)
        return output[start_index:end_index]

    def set_llm_params_based_on_failed_tries_amount(self, failed_amount):
        if failed_amount > 0:
            self.generation_config.temperature = self.temperature / (failed_amount + 1)
            # Notice 1.0 is the minimum value for repetition penalty, which means no penalty.
            diff = (self.repetition_penalty - 1.0) / (failed_amount + 1)
            self.generation_config.repetition_penalty = 1 + diff

    def forward(self, messages: list, failed_tries_so_far=0) -> str:
        """ messages should be of the format:
        [{'content' : 'value1', 'role': 'user'},
        {'content' : 'value1', 'role': 'user'}]
        """
        prompt = self.convert_messages_to_prompt(messages)
        self.set_llm_params_based_on_failed_tries_amount(failed_tries_so_far)
        # Save it to a file
        with open('prompt.txt', 'w') as f:
            f.write(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')  # self.device)
        input_token_amount = inputs['input_ids'][0].shape[0]
        print(f'Input token amount for generation : {input_token_amount}')
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            outputs = self.model.generate(**inputs, generation_config=self.generation_config,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          pad_token_id=self.tokenizer.pad_token_id)
            outputs = outputs[0][input_token_amount:]
        text_output = self.tokenizer.decode(outputs, skip_special_tokens=True)
        logger.debug(f'Generated code: {text_output}')
        return self.extract_function_of_output(text_output)
        # return text_output
