import torch.nn


class LLMWrapper(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.max_creation_times = None # Change it to an integer in derived classes.

    def forward(self, messages: list) -> str:
        pass
