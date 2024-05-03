import random
import math
from apo4rec.data import subsample_data
from apo4rec.eval import eval_a_prompt_with_debate
import numpy as np


class Selector:
    def __init__(self,
                 train_data,
                 config,
                 idx2title,
                 title2idx,
                 idx2attribute,
                 title2summary,
                 filter):
        self.train_data = train_data
        self.config = config
        self.idx2title = idx2title
        self.title2idx = title2idx
        self.idx2attribute = idx2attribute
        self.title2summary = title2summary
        self.filter = filter

    def ucb(self, prompt_list, generator):
        numbers_of_selections = [0] * len(prompt_list)
        sums_of_reward = [0] * len(prompt_list)
        index_list = [i for i in range(len(prompt_list))]

        for t in range(1, self.config['time_steps'] + 1):
            sample_data = subsample_data(self.train_data, self.config['select_sample_num'])
            if t == 1:
                select_prompt_index = random.choice(index_list)
            else:
                explore_param = self.config['explore_param']
                results = [q_value + explore_param * math.sqrt(math.log(t) / (n + 1)) for q_value, n in
                           zip(sums_of_reward, numbers_of_selections)]
                max_result = max(results)
                select_prompt_index = results.index(max_result)
            select_prompt = prompt_list[select_prompt_index]
            select_prompt_reward, _, _ = eval_a_prompt_with_debate(select_prompt, sample_data, generator, self.config, self.title2idx)

            # Update N and Q
            numbers_of_selections[select_prompt_index] += self.config['select_sample_num']
            sums_of_reward[select_prompt_index] += select_prompt_reward / numbers_of_selections[select_prompt_index]

        # Return top b prompts
        if self.config['beam_width'] > len(prompt_list):
            raise Exception("The value of beamwidth needs to be less than the length of the prompt list")
        else:
            # pairs = list(zip(sums_of_reward, prompt_list))
            # pairs.sort(reverse=True)
            sums_of_reward = np.array(sums_of_reward) * (-1)
            top_reward_indices = sums_of_reward.argsort()[:self.config['beam_width']]
            top_b_prompt = [prompt_list[i] for i in top_reward_indices]
            for i in top_reward_indices:
                print(f'select prompt {i} with score {sums_of_reward[i] * -1}: {prompt_list[i]}')

        return top_b_prompt

    def full_eval(self, prompt_list: list, generator):
        rewards = [0] * len(prompt_list)
        sample_data = subsample_data(self.train_data, 100)
        for i, prompt in enumerate(prompt_list):
            print(f'*** eval prompt {i}')
            reward, _, _, _, _ = eval_a_prompt_with_debate(prompt,
                                                     sample_data,
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
            print(f'train select prompt {i} with score {rewards[i] * -1}: {prompt_list[i]}')
        return top_b_prompt

    def select_prompt(self, prompt_list, generator):
        print(' ------------------------- selecting prompts ---------------------------')
        top_b_prompt = self.full_eval(prompt_list, generator)

        return top_b_prompt





