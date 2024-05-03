
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import fire

from llama import Llama
import random
from apo4rec.util import init_prompts
from apo4rec.filtre import Filter
from apo4rec.data import load_data_to_correct, create_split, init_item_summaries
from apo4rec.improve import Improver
from apo4rec.selector import Selector
from apo4rec.augment import Augmentor
from apo4rec.actioneer import Actioneer
from apo4rec.data import load_itemIdx2Str
from apo4rec.eval import correct_dataset


random.seed(123)

profiles =["As a skilled gamer and psychology expert, I examine the connections between video games by analyzing their core attributes and themes, "
           "such as gameplay mechanics, narrative, and player involvement. "
           "I comprehend users' tastes by examining their interaction data and applying psychological theories and principles to identify patterns and trends that shape the gaming experience and user behavior. "
           "By considering the interplay between game mechanics, narrative, and player engagement, I can offer insights into how video games affect user emotions, decision-making, and preferences."]

instructions = [
                # without planning
                "To determine the relevance of input video games to a target game, identify shared attributes such as genre, plot, theme, actor, director, and gameplay mechanics. "
                "Assign weights to these commonalities based on their importance, then classify input games as relevant or irrelevant to the target game based on their similarities and differences."
                "\nFor instance, if the target game is \"The Last of Us,\" input games with similar themes, such as post-apocalyptic survival, would be considered relevant. "
                "Similarly, games with similar gameplay mechanics, such as stealth and combat, would also be relevant. "
                "Conversely, games with vastly different genres, such as puzzle games, would be considered irrelevant."
                ]

restrictions = ["Please wrap relevant $item_type$s between '<answer>' and '</answer>', and separate $item_type$s with ';'."
               "If none of the input $item_type$s is relevant to the target $item_type$, then answer '<answer>NONE</answer>'."
                ]

# initial_memory = "Attributes like genre, plot, theme, and tone are most important for consideration."

examples =[
           # without plan
           "<example>\n"
           "<input_$item_type$s>\n"
           "* {[Mad Max 2]: ...}\n"
           "* {[Alien]: ...]}\n"
           "* {[The Shawshank Redemption]: ...}\n"
           "</input_$item_type$s>\n"
           "<target_$item_type$>\n"
           "* {[Terminator Salvation]: ...}\n"
           "</target_$item_type$>\n"
           "Answer: <answer>Mad Max 2; Alien</answer>\n"
           "</example>"
            ]

config = {
    # data settings
    'item_type': 'movie',
    'data_dir': './dataset/ml1m/',
    'file_name': 'synthetic_instances_simple.dat',
    # apo settings
    'select_sample_num': 20,
    'beam_width': 4,
    'explore_param': 2,
    'epoch_num': 5,
    'time_steps': 32,
    # llm settings
    'temperature': 0.01,
    'top_p': 0.1,
    'batch_size': 10,
    # test settings
    'verbose': True,
    # debate settings
    'agent_num': 1,
    'round_num': 1,
    'add_memory': True,
    'model_name': 'berd'
}


def main(
    ckpt_dir: str = 'llama-2-7b-chat/',
    tokenizer_path: str = 'tokenizer.model',
    max_seq_len: int = 2500,
    epoch_num=10
):
    batch_size = config['batch_size']
    prompts = init_prompts(profiles, instructions, restrictions, examples, config)

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )

    # prepare data
    idx2attributes, title2idx, idx2title = load_itemIdx2Str(config['data_dir'])
    title2summary, title2attribute = init_item_summaries(config['data_dir'], idx2attributes, idx2title, generator, config)
    filter = Filter(config['data_dir'])

    for file_id in range(20):
        file_name = '/' + config['model_name'] + f'/instances_to_correct_{file_id}.dat'
        raw_data = load_data_to_correct(config['data_dir'], file_name, config['item_type'], idx2title, title2summary, target_pos=-1)

        none_data_indices = correct_dataset(prompts[0], raw_data, generator, config, title2idx, idx2title, idx2attributes, title2summary, filter, config['model_name'], f'correct_input_{file_id}')
        for target_pos in range(3):
            replace_data = load_data_to_correct(config['data_dir'], file_name, config['item_type'], idx2title, title2summary, target_pos)
            instance_ids, inputs, input_items_list,  _, target_ids, target_titles, _ = replace_data
            selected_instance_ids = [instance_ids[i] for i in none_data_indices]
            selected_inputs = [inputs[i] for i in none_data_indices]
            selected_target_ids = [target_ids[i] for i in none_data_indices]
            selected_target_titles = [target_titles[i] for i in none_data_indices]
            selected_input_items_list = [input_items_list[i] for i in none_data_indices]
            selected_replace_data = [selected_instance_ids, selected_inputs, selected_input_items_list, None, selected_target_ids, selected_target_titles, None]
            correct_dataset(prompts[0], selected_replace_data, generator, config, title2idx, idx2title, idx2attributes, title2summary, filter, config['model_name'], f'correct_target_{file_id}_{target_pos}')


if __name__ == "__main__":
    fire.Fire(main)




