import os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM
import torch

from ovod.llm_wrappers.llama_2_code_mft_wrapper import CodeFuseWrapper


class CodeLLamaV2Phind(CodeFuseWrapper):
    def init_model(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlamaForCausalLM.from_pretrained(self.model_id, device_map=self.device,
                                                      load_in_8bit=self.load_in_8bit,
                                                      do_sample=self.do_sample,
                                                      trust_remote_code=True)

    def convert_messages_to_prompt(self, messages):
        final_str = ''

        for msg in messages:
            role = msg['role']

            if role == 'system' or role == 'System':
                role = 'System'
            elif role == 'user' or role == 'User':
                role = 'User'
            elif role == 'assistant' or role == 'Assistant':
                role = 'Assistant'
            else:
                raise Exception('Unknown role')
            content = msg['content']
            final_str += f"### {role} Prompt\n"
            final_str += f'{content}\n'
        final_str += f"### Assistant Prompt\n"
        return final_str

    def forward_prompt(self, prompt: str, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')  # self.device)
        input_token_amount = inputs['input_ids'][0].shape[0]
        print(f'Input token amount for generation : {input_token_amount}')
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            outputs = self.model.generate(**inputs, generation_config=self.generation_config,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          pad_token_id=self.tokenizer.pad_token_id)
            outputs = outputs[0][input_token_amount:]
        text_output = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return text_output
