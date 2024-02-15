from openai import OpenAI
import threading
from ovod.llm_wrappers.base_llm_wrapper import LLMWrapper
import os

os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'

# get API key:
with open('api.key') as f:
    api_key = f.read().strip()


class OpenAIWrapper(LLMWrapper):
    def __init__(self, model, temperature, top_p, frequency_penalty, presence_penalty, stop, **kwargs):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.model_id = model
        self.temperature = temperature
        self.max_creation_times = 3
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def forward(self, messages, failed_attempts_so_far=0):
        response = None
        while not response:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,  # 1.0, #config.codex.temperature,  0.
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop,
                    timeout=10
                )
            except Exception as e:
                # logger.error(
                print(f"Exception : {e} " +
                      f'Exception occurred when trying to generate code for an object, Starting over.')
                response = None
        output = response.choices[0].message.content
        return output

    def forward_threaded(self, messages, output_d, key, failed_attempts_so_far=0):
        output_d[key] = self.forward(messages, failed_attempts_so_far)

    def forward_batch(self, messages_batch, failed_attempts_so_far=0):
        # do it using threading
        threads = []
        outputs = {}
        for i, messages in enumerate(messages_batch):
            t = threading.Thread(target=self.forward_threaded, args=(messages, outputs, i, failed_attempts_so_far))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return [outputs[i] for i in range(len(messages_batch))]
