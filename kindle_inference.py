
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

profiles =["As an expert in the Kindle eBook domain, I can identify the relationships between eBooks based on their key attributes and topics. "
           "I am also knowledgeable in the psychology domain, allowing me to infer users' preferences based on their Kindle eBook interaction history. "
           ]

instructions = [
                # without planning
                "Classify each input Kindle eBook as either relevant or irrelevant to the target Kindle eBook based on the following criteria:\n"
                "To determine the relevance of an input eBook to a target eBook, I consider the following factors:\n"
                "1. Genre similarity: I assess the similarity between the genre of the input eBook and the target eBook. A higher similarity suggests a greater likelihood of relevance.\n"
                "2. Shared attributes: I evaluate the similarity between the shared attributes of the input and target eBooks, such as author, publisher, or keywords. A higher similarity indicates a greater likelihood of relevance.\n"
                "3. Series connection: I determine if the input eBook is part of the same series as the target eBook. If so, it may indicate a greater likelihood of relevance.\n"
                "4. User interaction history: I consider the user interaction history of the input and target eBooks, such as ratings, reviews, and reading history. A higher similarity in user interaction history may indicate a greater likelihood of relevance.\n\n"
                "By considering these factors, I can accurately determine the relevance of an input eBook to a target eBook with a high degree of accuracy."
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




