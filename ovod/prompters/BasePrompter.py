
class BasePrompter:
    def __init__(self, base_folder):
        self.base_folder = base_folder

    def get_llm_params(self) -> dict:
        raise NotImplementedError

    def get_prompt(self, query_text) -> list:
        raise NotImplementedError

