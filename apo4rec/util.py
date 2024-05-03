import fire

from llama import Llama, Dialog
from time import perf_counter
import re
import random
import math
import numpy as np
import difflib

def extract_matched_output(input_string):
    if 'None of the' in input_string:
        return 'NONE'
    else:
        pattern_1 = re.compile(r'<answer>(.*?)<answer>', re.DOTALL)
        matches_1 = pattern_1.findall(input_string)
        pattern_2 = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        matches_2 = pattern_2.findall(input_string)
        pattern_3 = re.compile(r'<answer>(.*?)\[answer\]', re.DOTALL)
        matches_3 = pattern_3.findall(input_string)
        pattern_4 = re.compile(r'<answer>(.*?)\[answer>', re.DOTALL)
        matches_4 = pattern_4.findall(input_string)
        matches = matches_1 + matches_2 + matches_3 + matches_4
        if matches:
            if len(matches) > 1:
                return matches[-1]
            else:
                return matches[0]
            # content_between_input_tags = match.group(1)
            # while content_between_input_tags[0] == ';':
            #     content_between_input_tags = content_between_input_tags[1:]
            # while content_between_input_tags[-1] == ';':
            #     content_between_input_tags = content_between_input_tags[0:-1]
            # return content_between_input_tags
        else:
            return 'NONE'


def extract_summary_1(input_string, config):
    item_type = config['item_type']
    pattern = re.compile(r'<summary_1>(.*?)</summary_1>', re.DOTALL)
    match = pattern.search(input_string)
    if match:
        content_between_input_tags = match.group(1)
        return content_between_input_tags
    else:
        return 'NONE'

def extract_summary_2(input_string, config):
    item_type = config['item_type']
    pattern = re.compile(r'summary_2>(.*?)</summary_2>', re.DOTALL)
    match = pattern.search(input_string)
    if match:
        content_between_input_tags = match.group(1)
        return content_between_input_tags
    else:
        return 'NONE'

def generate_responses(queries, generator, config):
    """
    queries: a list of input Strings
    """
    batch_size = config['batch_size']
    temperature = config['temperature']
    top_p = config['top_p']
    query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    all_output = []
    print(f'{len(queries)} queries is split into {len(query_batches)} batches with size {batch_size}')
    for i, query_batch in enumerate(query_batches):
        # print(f'batch {i} input: {query_batch}')
        start = perf_counter()
        response = generator.chat_completion(
            query_batch,
            max_gen_len=2000,
            temperature=temperature,
            top_p=top_p)
        batch_output = [result['generation']['content'].strip() for result in response]
        all_output += batch_output
        end = perf_counter()
        # print(f'batch {i} output: {batch_output}')
        print(f'batch {i} finished in {end-start} seconds')
    # if config['verbose']:
    #     print_inputs_outputs(inputs=[queries[0]], outputs=[all_output[0]])
    return all_output


def print_inputs_outputs(inputs: list, outputs: list):
    assert len(inputs) == len(outputs)
    for input, output in zip(inputs, outputs):
        print(f"> Input: {input[0]['content']}\n")
        print(f"> Output: {output}")
        print("\n==================================\n")


def prepare_inputs_for_prompt_strs(prompts: list):
    """
    prompts: list of strings
    """
    inputs = []
    for prompt in prompts:
        input = [{"role": "user",
            "content": f'{prompt}'
        }]
        inputs.append(input)

    return inputs


def generate_responses_and_extract_outputs(queries, generator, config):
    """
    queries: list of strings
    """
    inputs = prepare_inputs_for_prompt_strs(queries)
    responses = generate_responses(inputs, generator, config)
    outputs = []
    for response in responses:
        extracted_output = extract_matched_output(response)
        outputs.append(extracted_output)
    return outputs

def generate_responses_and_extract_outputs_for_debate(queries, input_items_list, target_titles, generator, config, filter, title2idx):
    """
    queries: list of strings
    """
    inputs = prepare_inputs_for_prompt_strs(queries)
    responses = generate_responses(inputs, generator, config)
    extracted_responses = []
    for query, response, target_title, input_items in zip(queries, responses, target_titles, input_items_list):
        extracted_output = extract_matched_output(response)
        predicted_items = get_item_title_set_for_eval(f'<answer>{extracted_output}<answer>', title2idx)
        if target_title in predicted_items:
            # print(f'target [{target_title}] is removed from the prediction set')
            predicted_items.remove(target_title)
        predicted_items = filter_prediction_set(predicted_items, target_title, filter)
        predicted_items = enrich_prediction_set(input_items, predicted_items, target_title, filter)
        predicted_items_str = itemList2Str(predicted_items)
        extracted_responses.append(f'<answer>{predicted_items_str}<answer>')
        # print('------------')
        # print(f'> INPUT: {query}')
        # print(f'> OUTPUT: {response}')
        # print(f'> Extracted Response: {predicted_items_str}')
        # print('------------')
    return extracted_responses, responses

def init_prompts(profiles, instructions, restrictions, examples, config):
    item_type = config['item_type']
    prompts = []
    for profile in profiles:
        for instruction in instructions:
            for example in examples:
                for restriction in restrictions:
                    prompt = {
                        'profile': profile.replace('$item_type$', item_type),
                        'instruction': instruction.replace('$item_type$', item_type),
                        'restriction': restriction.replace('$item_type$', item_type),
                        'example': example.replace('$item_type$', item_type),
                    }
                    prompts.append(prompt)
    return prompts


def construct_prompt(prompt_dict):

    # memory = prompt_dict['memory']
    example = prompt_dict['example']
    # if memory == '' and example == '':
    #     prompt = f'### Instruction:\n{get_overall_prompt(prompt_dict)}\n'
    # elif memory == '' and example != '': # example only
    #     prompt = f'### Instruction:\n{get_overall_prompt(prompt_dict)}\n' \
    #              f'### Examples:\n{example}\n'
    # elif memory != '' and example == '': # memory only
    #     prompt = f'### Instruction:\n{get_overall_prompt(prompt_dict)}\n' \
    #              f'### Notes:\n{memory}\n'
    # else:  # memory and example
    prompt = f'{get_prompt_profile(prompt_dict)}\n' \
             f'### Instruction:\n{get_overall_prompt(prompt_dict)}\n' \
             f'### Examples:\n{example}\n'
    return prompt

def get_overall_prompt(prompt_dict):
    # profile = prompt_dict['profile']
    instruction = prompt_dict['instruction']
    restriction = prompt_dict['restriction']

    return f'{instruction}\n{restriction}\n'


def get_prompt_profile(prompt_dict):
    profile = prompt_dict['profile']
    return f'\n{profile}\n'


def get_prompt_instruction(prompt_dict):
    instruction = prompt_dict['instruction']
    return f'\n{instruction}\n'


def get_prompt_memory(prompt_dict):
    memory = prompt_dict['memory']
    return f'\n{memory}\n'

def filter_prediction_set(predicted_items:set, target_title, filter):
    filtered_predicted_items = set()
    for input_item in predicted_items:
        if filter.check_mismatch(input_item, target_title) == 0:
            # print(f'the input [{input_item}] and target [{target_title}] are mismatched')
            pass
        else:
            filtered_predicted_items.add(input_item)
    return filtered_predicted_items

def enrich_prediction_set(input_items, predicted_items:set, target_title, filter):
    for item in input_items:
        if filter.check_match(item, target_title):
            predicted_items.add(item)
            # print(f'^^^ {item} is added for {target_title}')
    return predicted_items



def itemList2Str(items):
    if len(items) == 0:
        return 'NONE'
    else:
        output_str = ''
        for i, item in enumerate(items):
            if i == len(items) - 1:
                output_str += f'{item}'
            else:
                output_str += f'{item};'
        return output_str


def get_item_title_set_for_eval(input_str, title2idx):
    title_list = convert_response_to_item_titles(extract_matched_output(input_str), title2idx)
    if len(title_list) == 1 and title_list[0] in ['None', 'NONE']:
        title_set = set()
    else:
        title_set = set(title_list)
    return title_set



def convert_response_to_item_titles(input_str: str, title2idx: dict):
    input_str = input_str.replace('[', '').replace(']', '')
    if input_str in ['None', 'NONE']:
        return ['NONE']
    else:
        if ';' in input_str:
            separator = ';'
        elif ';;' in input_str:
            separator = ';;'
        else:
            # print(f'no appropriate separator for {input_str}')
            separator = ';'
        # print(f'items before split: {input_str}')
        items = input_str.strip().split(separator)
        if items[-1] == '':
            items = items[0:-1]

        valid_item_strs = title2idx.keys()
        # print(f'items after split: {items}')
        for i, item in enumerate(items):
            item = item.strip()
            if item not in valid_item_strs:
                score = -10
                similar_match = ''
                for valid_item_str in valid_item_strs:
                    current_score = difflib.SequenceMatcher(None, item, valid_item_str).ratio()
                    if current_score > score:
                        score = current_score
                        similar_match = valid_item_str
                print(f'{item} is not in the dict, find match {similar_match}')
                print(f'items: {items}')
                print(f'input_str: {input_str}')
                items[i] = similar_match
            else:
                items[i] = item.strip()

    return items