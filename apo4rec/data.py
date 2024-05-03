import random
from time import perf_counter
from apo4rec.util import construct_prompt
from apo4rec.util import generate_responses_and_extract_outputs
import os


def load_item_attributes(data_dir):
    file_path = data_dir + '/i_idx2attr.dat'

def load_itemIdx2Str(data_dir):
    idx2attributes = {}
    title2idx = {}
    idx2title = {}
    file_path = data_dir + '/i_idx2str.dat'
    line_count = 0
    attribute_types = []
    with open(file_path) as fin:
        for line in fin:
            if line_count == 0:
                attribute_types = line.strip().split(':::')[2:]  # itemIdx:::title:::year:::genre:::director:::actor:::tag:::country
            else:
                records = line.strip().split(':::')
                itemIdx = records[0]
                title = records[1].strip().replace(';', '').replace('&amp', '')
                title2idx[title] = itemIdx
                idx2title[itemIdx] = title
                idx2attributes[itemIdx] = {}
                for i, record in enumerate(records[2:]):
                    attribute_type = attribute_types[i]
                    attribute_value = record
                    if len(attribute_value) > 800:
                        attribute_value = attribute_value[0:800]
                    idx2attributes[itemIdx][attribute_type] = attribute_value
            line_count += 1
    return idx2attributes, title2idx, idx2title


def init_item_summaries(data_dir, idx2attributes, idx2title, generator, config):
    idx2summary = {}
    title2summary = {}
    title2attribute = {}
    file_path = data_dir + '/i_idx2summary.dat'
    # if file exists, read summary
    if os.path.exists(file_path):
        print('loading item summaries')
        with open(file_path) as fin:
            for line in fin:
                itemIdx, title, summary = line.strip().split(':::')
                title = title.strip().replace(';', '').replace('&amp', '')
                if len(summary) > 500:
                    summary = summary[0: 500] + '.'
                attribute = idx2attributes[itemIdx]
                idx2summary[itemIdx] = summary
                if title in title2summary:
                    print(f'title {title} is duplicated')
                title2summary[title] = summary
                title2attribute[title] = attribute

    else:  # if file does not exist, generate summary based on llm
        print('generating item summaries')
        output_lines = []
        unsuccess_item_ids = set(idx2attributes.keys())
        additional_str = 'You should wrap your summary between \'<answer>\' and \'</answer>\'. '
        repeat_num = 0
        while len(unsuccess_item_ids) > 0 and repeat_num < 10:
            idxs, titles, queries = prepare_queries_for_item_summary(unsuccess_item_ids, idx2attributes, idx2title, config, additional_str)
            generated_summaries = generate_responses_and_extract_outputs(queries, generator, config)
            success_item_ids = set()
            for itemIdx, title, summary in zip(idxs, titles, generated_summaries):
                if summary == 'NONE':
                    print(f'summary for item {itemIdx}: {title} is {summary}, re-generate later')
                    continue
                else:
                    summary = summary.replace('\n', ' ')
                    idx2summary[itemIdx] = summary
                    title2summary[title] = summary
                    attribute = idx2attributes[itemIdx]
                    title2attribute[title] = attribute
                    outputline = f'{itemIdx}:::{title}:::{summary}\n'
                    output_lines.append(outputline)
                    success_item_ids.add(itemIdx)
                if summary == 'NONE' and repeat_num == 9:
                    summary = title
                    idx2summary[itemIdx] = summary
                    title2summary[title] = summary
                    attribute = idx2attributes[itemIdx]
                    title2attribute[title] = attribute
                    outputline = f'{itemIdx}:::{title}:::{summary}\n'
                    output_lines.append(outputline)
                    success_item_ids.add(itemIdx)
            unsuccess_item_ids = unsuccess_item_ids - success_item_ids
            item_type = config['item_type']
            additional_str += f'\n# Generate the summary for {item_type} {title}, and wrap your answer between \'<answer>\' and \'</answer>\'.\n'
            repeat_num += 1

        with open(file_path, 'w') as fout:
            fout.writelines(output_lines)
        print(f'lens of ldxs, titles, queries: {len(idxs)}, {len(titles)}, {len(queries)}')
    print(f'lens of idx2summary, title2summary: {len(idx2summary)}, {len(title2summary)}')
    # assert len(idx2summary) == len(title2summary)

    return title2summary, title2attribute


def string_seq2id_seq(str2idx: dict, input_str_seq: str, separator:str=';'):
    output_id_seq = ''
    item_strs = input_str_seq.split(separator)
    for i, item_str in enumerate(item_strs):
        item_str = item_str.strip()
        item_id = str2idx[item_str]
        if i == len(item_strs) - 1:
            output_id_seq += f'{item_id}'
        else:
            output_id_seq += f'{item_id},'
    return output_id_seq

def load_instances(data_dir: str, file_name: str, item_type, idx2title, title2summary, add_memory=False):
    inputs = []
    outputs = []
    instance_ids = []
    target_ids = []
    involved_itemIds = []
    target_titles = []
    input_items_list = []
    with open(data_dir + file_name) as fin:
        for line in fin:
            line = line.strip().split(':::')
            instance_ids.append(line[0])
            input_items = line[1]
            input_items_list.append(itemIdList2titleList(input_items, idx2title))
            target_itemId = line[2]
            target_item_title_summary = itemIds2titles_and_summary(target_itemId, idx2title, title2summary)
            target_title = idx2title[target_itemId]
            target_titles.append(target_title)
            target_ids.append(target_itemId)
            matched_historical_movie = line[3]
            involved_itemIds.append(input_items.strip().split(',') + [target_itemId])
            if add_memory:
                inputs.append(f'<input_{item_type}s>\n{itemIds2titles_and_summary(input_items, idx2title, title2summary)}</input_{item_type}s>\n'
                          f'<target_{item_type}>\n{target_item_title_summary}</target_{item_type}>')
            else:
                inputs.append(
                    f'<input_{item_type}s>\n{itemIds2titles(input_items, idx2title)}\n</input_{item_type}s>\n'
                    f'<target_{item_type}>\n{target_title}\n</target_{item_type}>')
            outputs.append(f'<answer>{itemIds2titles(matched_historical_movie, idx2title)}<answer>')
    return instance_ids, inputs, input_items_list, outputs, target_ids, target_titles, involved_itemIds

def load_data_to_correct(data_dir: str, file_name: str, item_type, idx2title, title2summary, target_pos=-1):
    inputs = []
    outputs = []
    instance_ids = []
    target_ids = []
    target_titles = []
    input_items_list = []
    with open(data_dir + file_name) as fin:
        for line in fin:
            line = line.strip().split(':::')
            instance_ids.append(line[0])
            input_items = line[3]
            input_items_list.append(itemIdList2titleList(input_items, idx2title))
            if target_pos == -1:
                target_itemId = line[4]
            else:
                replace_candidates = line[5].replace('[', '').replace(']', '').strip().split(', ')
                target_itemId = replace_candidates[target_pos]
            target_item_title_summary = itemIds2titles_and_summary(target_itemId, idx2title, title2summary)
            target_title = idx2title[target_itemId]
            target_titles.append(target_title)
            target_ids.append(target_itemId)

            inputs.append(
                    f'<input_{item_type}s>\n{itemIds2titles_and_summary(input_items, idx2title, title2summary)}</input_{item_type}s>\n'
                    f'<target_{item_type}>\n{target_item_title_summary}</target_{item_type}>')

    return instance_ids, inputs, input_items_list, None, target_ids, target_titles, None


def itemIds2titles_and_summary(itemIds, idx2title, title2summary):
    output_str = ''
    itemIds = itemIds.replace('[', '').replace(']', '')
    if ', ' in itemIds:
        itemIds = itemIds.strip().split(', ')
    else:
        itemIds = itemIds.strip().split(',')
    num_item = len(itemIds)
    for i, itemIdx in enumerate(itemIds):
        if itemIdx in idx2title:
            title = idx2title[itemIdx]
            summary = title2summary[title]
            output_str += f'* {{[{title}]: {summary.strip()}}}\n'
    return output_str

# def itemIds2titles(itemIds, idx2title):
#     output_str = ''
#     itemIds = itemIds.strip().split(',')
#     num_item = len(itemIds)
#     for i, itemIdx in enumerate(itemIds):
#         if itemIdx in idx2title:
#             title = idx2title[itemIdx]
#             summary = title2summary[title]
#             output_str += f'[{title}]: {summary}\n'
#     return output_str

def itemIds2titles(itemIds, idx2title):
    output_str = ''
    itemIds = itemIds.replace('[', '').replace(']', '')
    if ', ' in itemIds:
        itemIds = itemIds.strip().split(', ')
    else:
        itemIds = itemIds.strip().split(',')
    num_item = len(itemIds)
    for i, itemIdx in enumerate(itemIds):
        if itemIdx in idx2title:
            title = idx2title[itemIdx]
            if i == num_item - 1:
                output_str += f'{title}'
            else:
                output_str += f'{title}; '
        else:
            output_str = 'NONE'
    return output_str


def itemIdList2titleList(itemIds, idx2title):
    output_list = []
    itemIds = itemIds.replace('[', '').replace(']', '')
    if ', ' in itemIds:
        itemIds = itemIds.strip().split(', ')
    else:
        itemIds = itemIds.strip().split(',')
    num_item = len(itemIds)
    for i, itemIdx in enumerate(itemIds):
        if itemIdx in idx2title:
            title = idx2title[itemIdx]
            output_list.append(title)
    return output_list

def itemIds2descriptions(itemIds, idx2title, idx2str):
    output_str = ''
    itemIds = itemIds.strip().split(',')
    num_item = len(itemIds)
    for i, itemIdx in enumerate(itemIds):
        title = idx2title[itemIdx]
        detail = idx2str[itemIdx]
        output_str += f'<{title}>\n{detail}\n<{title}>\n'
    return output_str


def subsample_data(data, subsample_size):
    """
    Subsample data. Data is in the form of a triplet of lists, i.e., instance_ids, inputs and labels.
    """
    instance_ids, inputs, input_items_list, outputs, target_ids, target_titles, involved_itemIds = data
    assert len(instance_ids) == len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), subsample_size)
    sampled_instance_ids = [instance_ids[i] for i in indices]
    sampled_inputs = [inputs[i] for i in indices]
    sampled_input_items_list = [input_items_list[i] for i in indices]
    sampled_outputs = [outputs[i] for i in indices]
    sampled_target_ids = [target_ids[i] for i in indices]
    sampled_involved_itemIds = [involved_itemIds[i] for i in indices]
    sampled_target_titles = [target_titles[i] for i in indices]
    return sampled_instance_ids, sampled_inputs, sampled_input_items_list, sampled_outputs, sampled_target_ids, sampled_target_titles, sampled_involved_itemIds


def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a triplet of lists, i.e., instance_ids, inputs and labels.
    """
    instance_ids, inputs, input_items_list, outputs, target_ids, target_titles, involved_itemIds = data
    assert len(instance_ids) == len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), split_size)
    inputs1 = [inputs[i] for i in indices]
    input_items_list1 = [input_items_list[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    target_ids1 = [target_ids[i] for i in indices]
    involved_itemIds1 = [involved_itemIds[i] for i in indices]
    instance_ids1 = [instance_ids[i] for i in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    input_items_list2 = [input_items_list[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    instance_ids2 = [instance_ids[i] for i in range(len(inputs)) if i not in indices]
    target_ids2 = [target_ids[i] for i in range(len(inputs)) if i not in indices]
    involved_itemIds2 = [involved_itemIds[i] for i in range(len(inputs)) if i not in indices]
    target_titles1 = [target_titles[i] for i in indices]
    target_titles2 = [target_titles[i] for i in range(len(inputs)) if i not in indices]

    return (instance_ids1, inputs1, input_items_list1, outputs1, target_ids1, target_titles1, involved_itemIds1), (instance_ids2, inputs2, input_items_list2, outputs2, target_ids2, target_titles2, involved_itemIds2)


def prepare_queries_for_one_prompt(a_prompt: dict, inputs, item_type, idx2title=None, idx2attribute=None, title2summary=None, other_responses=None):
    instance_num = len(inputs)
    msgs = []
    for i in range(instance_num):
        input = inputs[i]
        if other_responses is not None:
            other_response = other_responses[i]
        else:
            other_response = None
        query = generate_a_query(a_prompt, input, item_type, idx2title, idx2attribute, title2summary, other_response)
        msgs.append(query)
    return msgs

def generate_a_query(prompt_dict, input, item_type, idx2title=None, idx2attribute=None, title2summary=None, other_response=None):

    profile = prompt_dict['profile']
    instruction = prompt_dict['instruction']
    restriction = prompt_dict['restriction']
    example = prompt_dict['example']
    appended_str = ''

    if other_response is not None:
        appended_str += f'\n### Other agents\' answer:\n{other_response}'
    query = f'<instructions>\n' \
            f'{profile}\n' \
            f'{instruction}\n' \
            f'</instructions>\n' \
            f'<restrictions>\n' \
            f'{restriction}\n' \
            f'</restrictions>\n'\
            f'### Here is an example for you to better understand the task:\n{example}\nDo not use any {item_type}s in the above examples.\n'\
            f'### Here is your task:\n{input}'\
            f'{appended_str}' \
            f'\nRemember you must wrap relevant {item_type}s between \'<answer>\' and \'</answer>\', and separate {item_type}s with \';\''

    return query

def itemIds2attributes(involved_itemIds, idx2title, idx2attribute):
    output_str = ''
    for itemIdx in involved_itemIds:
        title = idx2title[itemIdx]
        attribute = idx2attribute[itemIdx]
        output_str += f'{title}: {attribute}\n'
    return output_str

def itemIds2summary(involved_itemIds, idx2title, title2summary):
    output_str = ''
    for itemIdx in involved_itemIds:
        title = idx2title[itemIdx]
        summary = title2summary[title]
        output_str += f'* {title}: {summary}\n'
    return output_str


def prepare_queries_for_item_summary(required_item_ids, idx2attributes, idx2title, config, additional_str=''):
    instance_num = len(idx2title)
    queries = []
    idxs = []
    titles = []
    for itemIdx in required_item_ids:
        attributes = idx2attributes[itemIdx]
        title = idx2title[itemIdx]
        item_type = config['item_type']
        msg = f'### Instruction: please generate a summary of a {item_type}, with no more than 30 words. '\
              f'The summary should summarize the plot and the most representative attributes.'\
              f'Please wrap your answer between \'<answer>\' and \'</answer>\'.\n' \
              f'### Example:\n' \
              f'The summary of the movie [Ever After] is as follows: ' \
              f'<answer>Ever After is a 1998 romantic comedy-drama film about a commoner who marries a prince and must navigate royal politics and gender roles. Starring Anjelica Huston and Dougray Scott.<answer>\n' \
              f'### Your task: referring to the above example, you have to generate the summary of {item_type} [{title}], based on its attributes: {attributes}.\n' \
              f'Note that you should not completely copy the attributes above. You should summarize the attributes and wrap your answer between \'<answer>\' and \'</answer>\' with no more than 30 words. {additional_str}'
        idxs.append(itemIdx)
        queries.append(msg)
        titles.append(title)
    return idxs, titles, queries





