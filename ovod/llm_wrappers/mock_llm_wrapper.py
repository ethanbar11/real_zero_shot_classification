from ovod.llm_wrappers.base_llm_wrapper import LLMWrapper


class MockLLMWrapper(LLMWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.max_creation_times = 3

    def forward(self, messages: list, failed_tries_so_far=None) -> str:
        code = """def execute_command(img, patch):
    patch.initialize_image(img)
    attributes = {}
    special_weights = {}
    
    attributes['medium sized'] = patch.clip_similarity('A really nice program','Oh yeah so nice and effective')
                                 
    
    return attributes, special_weights"""
        return code
