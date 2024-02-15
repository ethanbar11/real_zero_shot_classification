import torch
from fvcore.common.config import CfgNode

from ovod.llm_wrappers.open_ai_wrapper import OpenAIWrapper
from ovod.llm_wrappers.llama_2_code_mft_wrapper import CodeFuseWrapper
from ovod.llm_wrappers.llama_2_code_phind import CodeLLamaV2Phind
from ovod.llm_wrappers.mock_llm_wrapper import MockLLMWrapper


def build_llm_wrapper(cfg: CfgNode) -> torch.nn.Module:
    name = cfg.LLM_WRAPPER.ARCH

    print('Loading LLM Wrapper {}.'.format(name))
    if name == 'openai_wrapper':
        print('Model: {}'.format(cfg.LLM_WRAPPER.PARAMS.model))
        llm_wrapper = OpenAIWrapper(**cfg.LLM_WRAPPER.PARAMS)
    elif name == 'llama_2_mft_coder':
        llm_wrapper = CodeFuseWrapper(**cfg.LLM_WRAPPER.PARAMS)
    elif name == 'llama_2_phind':
        llm_wrapper = CodeLLamaV2Phind(**cfg.LLM_WRAPPER.PARAMS)
    elif name == 'mock_llm':
        llm_wrapper = MockLLMWrapper(**cfg.LLM_WRAPPER.PARAMS)
    else:
        raise NotImplementedError("LLM Wrapper name is not supported..")

    return llm_wrapper
