import random
import math
from apo4rec.data import subsample_data
from apo4rec.eval import eval_a_prompt_with_debate
import numpy as np


class Evaluator:
    def __init__(self,
                 test_data,
                 config,
                 idx2title,
                 title2idx,
                 idx2attribute,
                 title2summary,
                 filter):
        self.test_data = test_data
        self.config = config
        self.idx2title = idx2title
        self.title2idx = title2idx
        self.idx2attribute = idx2attribute
        self.title2summary = title2summary
        self.filter = filter

    def full_eval(self, prompt_list: list, generator):
        rewards = [0] * len(prompt_list)
        for i, prompt in enumerate(prompt_list):
            print(f'*** eval prompt {i}')
            reward, _, _, _, _ = eval_a_prompt_with_debate(prompt,
                                                           self.test_data,
                                                           generator,
                                                           self.config,
                                                           self.title2idx,
                                                           self.idx2title,
                                                           self.idx2attribute,
                                                           self.title2summary,
                                                           self.filter,
                                                           additional_str=f'full eval prompt {i}')
            rewards[i] = reward
        rewards = np.array(rewards) * (-1)
        top_reward_indices = rewards.argsort()[:self.config['beam_width']]
        top_b_prompt = [prompt_list[i] for i in top_reward_indices]
        for i in top_reward_indices:
            print(f'eval select prompt {i} with score {rewards[i] * -1}: {prompt_list[i]}')
        return top_b_prompt

    def evaluate_prompt(self, prompt_list, generator):
        print(' ------------------------- evaluating prompts ---------------------------')
        top_b_prompt = self.full_eval(prompt_list, generator)

        return top_b_prompt





