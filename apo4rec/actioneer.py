import random
from apo4rec.eval import eval_a_prompt_with_debate
from apo4rec.util import generate_responses_and_extract_outputs

from time import perf_counter






class Actioneer:
    def __init__(self,
                 config,
                 idx2str,
                 str2idx):
        self.config = config
        self.idx2str = idx2str
        self.str2idx = str2idx

    def debate(self, prompts, generator, data):
        """
        prompts: a list of dicts
        generator: the llm
        """
        print(' --------------- debating -----------------')

        for i, prompt in enumerate(prompts):
            eval_a_prompt_with_debate(prompt, data, generator, self.config, self.str2idx)

        return prompts
