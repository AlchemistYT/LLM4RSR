
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import fire

from llama import Llama
import random
from apo4rec.util import init_prompts
from apo4rec.filtre import Filter
from apo4rec.data import load_instances, create_split, init_item_summaries
from apo4rec.improve import Improver
from apo4rec.selector import Selector
from apo4rec.evaluator import Evaluator
from apo4rec.augment import Augmentor
from apo4rec.actioneer import Actioneer
from apo4rec.data import load_itemIdx2Str


random.seed(123)

profiles =["You are an expert in the $item_type$ domain and good at telling the relationships between $item_type$s based on their key attributes and topics.",
           "You are an expert in the $item_type$ domain and good at telling the relationships between $item_type$s based on their key attributes and topics."\
           "You are also an expert in the psychology domain and good at telling users' preferences based on their $item_type$ interaction list."]

instructions = [
                # without planning
                "You will be given an input $item_type$ list and a target $item_type$, together with the summaries of these $item_type$s. "
                "You need to answer which input $item_type$(s) is/are relevant to the target $item_type$, w.r.t. attributes like genre, plot, theme, actor, director, etc.",
                # with plan
                "You will be given an input $item_type$ list and a target $item_type$, together with the summaries of these $item_type$s. "
                "You need to answer which input $item_type$(s) is/are relevant to the target $item_type$. "
                "'Relevant' means that the input and target $item_type$ reflect the same preference or intention of a user. "
                "You should solve the problem step by step as follows:\n"
                "Step 1: identify the commonness between each input item and the target pairwisely, w.r.t. attributes like genre, plot, theme, actor, director, etc.\n"
                "Step 2: determine whether the commonness is significant enough to indicate relevance. Attributes like genre, plot, theme, and tone are most important for consideration."
                ]

restrictions = ["Please wrap relevant $item_type$s between '<answer>' and '</answer>', and separate $item_type$s with ';'."
               "If none of the input $item_type$s is relevant to the target $item_type$, then answer '<answer>NONE</answer>'.",
               "Please wrap relevant $item_type$s between '<answer>' and '</answer>', and separate $item_type$s with ';'."
               "Please select relevant $item_type$s stringently. If none of the input $item_type$s is relevant to the target $item_type$, then answer '<answer>NONE</answer>'."
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
           "</example>\n"
           "\n"
            ]

config = {
    # data settings
    # video game, CD, kindle ebook, movie
    'item_type': 'movie',
    'data_dir': './dataset/ml1m/',
    'file_name': 'synthetic_instances_simple.dat',
    # apo settings
    'select_sample_num': 30,
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
    'agent_num': 3,
    'round_num': 2,
    'add_memory': True,
}

def main(
    ckpt_dir: str = 'llama-2-7b-chat/',
    tokenizer_path: str = 'tokenizer.model',
    max_seq_len: int = 3000,
    epoch_num=50
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
    full_data = load_instances(config['data_dir'], config['file_name'], config['item_type'], idx2title, title2summary, config['add_memory'])
    test_size = min(len(full_data[0]), 100)
    test_data, train_data = create_split(full_data, test_size)

    filter = Filter(config['data_dir'])

    improver = Improver(train_data, config, idx2title, title2idx, idx2attributes, title2summary, title2attribute, filter)
    selector = Selector(train_data, config, idx2title, title2idx, idx2attributes, title2summary, filter)
    augmentor = Augmentor(config)
    evaluator = Evaluator(test_data, config, idx2title, title2idx, idx2attributes, title2summary, filter)

    # train stage
    for agent_num in [1]:
        for round_num in [1]:
            print('###################################')
            print('####### apo training stage ########')
            print('###################################')
            config['agent_num'] = agent_num
            config['round_num'] = round_num
            print(config)
            for epoch in range(epoch_num):
                print(f' ================ epoch {epoch} training ========================')
                prompts = improver.improve_prompt(prompts, generator)
                prompts = selector.select_prompt(prompts, generator)
                prompts = augmentor.augment_prompt(prompts, generator)
                evaluator.evaluate_prompt(prompts, generator)

if __name__ == "__main__":
    fire.Fire(main)




