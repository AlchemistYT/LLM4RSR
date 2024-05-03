

from llama import Llama, Dialog
from time import perf_counter
import random
import os
import math
from time import perf_counter
import numpy as np
from apo4rec.data import prepare_queries_for_one_prompt, load_itemIdx2Str, string_seq2id_seq
from apo4rec.util import generate_responses, extract_matched_output, generate_responses_and_extract_outputs, generate_responses_and_extract_outputs_for_debate, construct_prompt, itemList2Str, filter_prediction_set, get_item_title_set_for_eval, convert_response_to_item_titles




def correct_dataset(prompt, data, generator, config, title2idx, idx2title, idx2attribute, title2summary, filter, model_name, file_name):
    agent_num = config['agent_num']
    round_num = config['round_num']
    start = perf_counter()
    instance_ids, inputs, input_items_list, _, target_ids, target_titles, _ = data
    num_data = len(instance_ids)
    last_debate_responses = [None for i in range(agent_num)]  # agent_num, data_num
    debate_responses = None
    for debate_round in range(round_num):
        for current_agent_id in range(agent_num):
            other_responses = get_debate_output(agent_num, num_data, last_debate_responses, title2idx, debate_round)
            # [data_num]
            queries = prepare_queries_for_one_prompt(prompt, inputs, config['item_type'], idx2title, idx2attribute, title2summary, other_responses)
            # data_num of: <answer>{extracted_output}</answer>
            current_responses, raw_responses = generate_responses_and_extract_outputs_for_debate(queries, input_items_list, target_titles, generator, config, filter, title2idx)
            last_debate_responses[current_agent_id] = current_responses
        print(f'debate round {debate_round} result')
        # list of strings: item; item; item
        debate_responses = get_debate_output(agent_num, num_data, last_debate_responses, title2idx, debate_round + 1)

    complete_mismatch_data_ids = []
    for i, debate_response in enumerate(debate_responses):
        if 'NONE' in debate_response:
            complete_mismatch_data_ids.append(i)

    end = perf_counter()
    print(f'*** prompt inference finished in {end-start} seconds')
    generate_corrected_data(instance_ids, debate_responses, target_ids, config, idx2title, title2idx, model_name, file_name)

    return complete_mismatch_data_ids


def eval_a_prompt_with_debate(prompt, data, generator, config, title2idx, idx2title, idx2attribute, title2summary, filter, additional_str='', print_output=True):
    agent_num = config['agent_num']
    round_num = config['round_num']
    print(f'*** eval prompt: {prompt} with debate')
    start = perf_counter()
    instance_ids, inputs, input_items_list, groundtruth_outputs, target_ids, target_titles, involved_itemIds = data
    num_data = len(instance_ids)
    last_debate_responses = [None for i in range(agent_num)]  # agent_num, data_num
    missing_items_list, redundant_items_list = [], []
    for debate_round in range(round_num):
        for current_agent_id in range(agent_num):
            other_responses = get_debate_output(agent_num, num_data, last_debate_responses, title2idx, debate_round)
            # [data_num]
            queries = prepare_queries_for_one_prompt(prompt, inputs, config['item_type'], idx2title, idx2attribute, title2summary, other_responses)
            # data_num of: <answer>{extracted_output}</answer>
            current_responses, raw_responses = generate_responses_and_extract_outputs_for_debate(queries, input_items_list, target_titles, generator, config, filter, title2idx)
            last_debate_responses[current_agent_id] = current_responses
            print(f'debate round {debate_round}, agent {current_agent_id} result')
            get_eval_scores(current_responses, groundtruth_outputs, target_titles, title2idx, instance_ids, queries, raw_responses, additional_str=additional_str + f', debate {debate_round}, agent {current_agent_id}')
        print(f'debate round {debate_round} result')
        # list of strings: item; item; item
        debate_responses = get_debate_output(agent_num, num_data, last_debate_responses, title2idx, debate_round + 1)
        score, missing_items_list, redundant_items_list = get_eval_scores(debate_responses, groundtruth_outputs, target_titles, title2idx, instance_ids, additional_str=additional_str + f', debate final')
    end = perf_counter()
    print(f'*** prompt eval finished in {end-start} seconds')
    should_be_relevant_pairs_title, should_be_irrelevant_pairs_title, should_be_irrelevant_pairs_title_list, should_be_relevant_pairs_title_list= \
        get_errors(target_titles, missing_items_list, redundant_items_list, config['item_type'])
    assert len(inputs) == len(missing_items_list) == len(redundant_items_list)

    return score, should_be_relevant_pairs_title, should_be_irrelevant_pairs_title, should_be_irrelevant_pairs_title_list, should_be_relevant_pairs_title_list


def get_eval_scores(model_predictions, groundtruth_outputs, targets, str2idx, instance_idxs, queries=None, raw_responses=None, additional_str=''):
    if type(model_predictions) is not list:
        model_predictions = [model_predictions]
    if type(groundtruth_outputs) is not list:
        groundtruth_outputs = [groundtruth_outputs]

    no_f_precisions = []
    no_f_recalls = []
    no_f_f1s = []
    no_f_corrects = []

    missing_items_list = []
    redundant_items_list = []

    for i in range(len(model_predictions)):
        prediction = model_predictions[i]
        groundtruth = groundtruth_outputs[i]
        target = targets[i]
        instance_idx = instance_idxs[i]
        if queries is None:
            query = None
            response = None
        else:
            query = queries[i]
            response = raw_responses[i]

        no_f_precision, no_f_recall, no_f_f1, no_f_correct, missing_items, redundant_items = get_rec_matching_score(prediction, groundtruth, target, str2idx, instance_idx, query, response)

        no_f_precisions.append(no_f_precision)
        no_f_recalls.append(no_f_recall)
        no_f_f1s.append(no_f_f1)
        no_f_corrects.append(no_f_correct)

        missing_items_list.append(missing_items)
        redundant_items_list.append(redundant_items)

    no_f_precisions = np.array(no_f_precisions)
    no_f_recalls = np.array(no_f_recalls)
    no_f_f1s = np.array(no_f_f1s)
    no_f_corrects = np.array(no_f_corrects)

    print(f'*** {additional_str} overall result')
    print(f'Precision: {no_f_precisions.mean()}')
    print(f'Recall: {no_f_recalls.mean()}')
    print(f'F1: {no_f_f1s.mean()}')
    print(f'Correct: {no_f_corrects.mean()}')


    return no_f_precisions.mean(), missing_items_list, redundant_items_list


def get_rec_matching_score(prediction, ground_truth, target_title, title2idx, instance_idx, query=None, response=None):

    if '<answer>' not in prediction:
        prediction = f'<answer>{prediction}</answer>'

    predicted_items = get_item_title_set_for_eval(prediction, title2idx)
    groundtruth_items = get_item_title_set_for_eval(ground_truth, title2idx)

    if len(predicted_items) == 0 and len(groundtruth_items) == 0:
        precision, recall, f1, correct = 1, 1, 1, 1
        num_same = 0
    else:
        num_same = len(predicted_items & groundtruth_items)
        if num_same == 0:
            precision, recall, f1, correct = 0, 0, 0, 0
        else:
            precision = 1.0 * num_same / len(predicted_items)
            recall = 1.0 * num_same / len(groundtruth_items)
            f1 = (2 * precision * recall) / (precision + recall)
            if predicted_items <= groundtruth_items:
                correct = 1
            else:
                correct = 0

    missing_items = groundtruth_items - predicted_items
    redundant_items = predicted_items - groundtruth_items

    # if precision < 0.5 and query is not None:
    #     print('----------------')
    #     print(f'> Input: {query}')
    #     print(f'> Output: {response}')
    #     print('----------------')
    # print(f'instance_idx: {instance_idx}')
    # print(f'prediction: {predicted_items}')
    # print(f'groundtruth: {groundtruth_items}')
    # print(f'target: {target_title}')
    # print(f'num_same: {num_same}')
    # print(f'precision: {precision}')
    # print(f'recall: {recall}')
    # print(f'correct: {correct}')
    # print(f'missing_items: {missing_items}')
    # print(f'redundant_items: {redundant_items}')
    # print('==============================')


    return precision, recall, f1, correct, missing_items, redundant_items




def ids2str(items: list):
    if len(items) == 0:
        return 'None'
    else:
        items_str = ''
        for i, item in enumerate(items):
            if i < len(items) - 1:
                items_str += item
                items_str += ';;'
            else:
                items_str += item
        return items_str

def get_errors(targets, missing_items_list, redundant_items_list, item_type, max_error_count=5):
    should_be_irrelevant_pairs_title = ''
    should_be_irrelevant_count = 0
    should_be_relevant_pairs_title = ''
    should_be_relevant_count = 0
    should_be_irrelevant_pairs_title_list = []
    should_be_relevant_pairs_title_list = []
    for target_item, missing_items, redundant_items in zip(targets, missing_items_list, redundant_items_list):
        for missing_item in missing_items:
            should_be_relevant_pairs_title_list.append((missing_item, target_item))
            if should_be_relevant_count < max_error_count:
                should_be_relevant_pairs_title += f'({missing_item};;{target_item}), '
                should_be_relevant_count += 1
        for redundant_item in redundant_items:
            should_be_irrelevant_pairs_title_list.append((redundant_item, target_item))
            if should_be_irrelevant_count < max_error_count:
                should_be_irrelevant_pairs_title += f'({redundant_item};;{target_item}), '
                should_be_irrelevant_count += 1

    return should_be_relevant_pairs_title, should_be_irrelevant_pairs_title, should_be_irrelevant_pairs_title_list, should_be_relevant_pairs_title_list

def debate(prompt, data, generator, config, str2idx, idx2str):
    agent_num = config['agent_num']
    round_num = config['round_num']
    print(f'*** eval prompt: {prompt} with debate')
    start = perf_counter()
    instance_ids, inputs, groundtruth_outputs, targets = data
    debate_responses = [[] for i in range(agent_num)]  # agent_num, round_num, data_num

    # debating
    # for debate_round in range(round_num):
    #     # print(f'*** debating round {debate_round}')
    #     for current_agent_id in range(agent_num):
    #         if debate_round == 0:
    #             queries = prepare_queries_for_one_prompt(prompt, data, config['item_type'])
    #         else:
    #             other_agent_id = random.randint(0, agent_num-1)
    #             other_responses = debate_responses[other_agent_id][-1]
    #             queries = prepare_queries_for_one_prompt_with_other_responses(prompt, data, config['item_type'], other_responses)
    #         extracted_responses = generate_responses_and_extract_outputs_for_debate(queries, generator, config)
    #         # current_score, _, _ = get_eval_scores(extracted_responses, groundtruth_outputs, str2idx)
    #         # for query, response, groundtruth in zip(queries, current_responses, groundtruth_outputs):
    #         #     print('========================================')
    #         #     print(f'query: {query}')
    #         #     print(f'groundtruth: {groundtruth}')
    #         #     print(f'response: {response}')
    #         #     score = get_eval_scores(response, groundtruth, str2idx)
    #         debate_responses[current_agent_id].append(extracted_responses)
    #     print(f'*** eval with debate round {debate_round}')
    #     corrected_inputs = eval_debate(agent_num, debate_responses, len(instance_ids), groundtruth_outputs, str2idx, config)
    #     generate_corrected_data(instance_ids, corrected_inputs, targets, config, str2idx)
    # if config['verbose']:
    #     print_debate_responses(debate_responses, groundtruth_outputs, inputs, str2idx)


# def print_debate_responses(debate_responses, groundtruth_outputs, inputs, str2idx):
#     # agent_num, round_num, data_num
#     agent_num = len(debate_responses)
#     round_num = len(debate_responses[0])
#     data_num = len(debate_responses[0][0])
#     threshold = math.ceil(agent_num / 2)
#     print(f'agent_num: {agent_num}')
#     print(f'round_num: {round_num}')
#     print(f'data_num: {data_num}')
#     for data_id in range(data_num):
#         for debate_round in range(round_num):
#             item_count_dict = {}
#             majority_list = []
#             intersection_list = []
#             for agent_id in range(agent_num):
#                 response = debate_responses[agent_id][debate_round][data_id]
#                 items = convert_response_to_item_titles(response, str2idx)
#                 score, missing_items_list, redundant_items_list = get_eval_scores(response, groundtruth_outputs[data_id], str2idx)
#                 print(f'***\n'
#                       f'round_{round_num}, agent_{agent_id}, score {score}, \n'
#                       f'inputs: {inputs[data_id]}\n'
#                       f'groundtruth: {groundtruth_outputs[data_id]}\n'
#                       f'prediction: {items}\n'
#                       f'missing_items_list: {missing_items_list}\n'
#                       f'redundant_items_list: {redundant_items_list}')
#                 for item in items:
#                     if item not in item_count_dict:
#                         item_count_dict[item] = 0
#                     item_count_dict[item] += 1
#             for item, count in item_count_dict.items():
#                 if count >= threshold:
#                     majority_list.append(item)
#                 if count == agent_num:
#                     intersection_list.append(item)
#             majority_score, majority_missing_list, majority_redundant_list = get_eval_scores(majority_list, groundtruth_outputs[data_id], str2idx)
#             intersection_score, intersection_missing_list, intersection_redundant_list = get_eval_scores(intersection_list, groundtruth_outputs[data_id], str2idx)
#             print(f'*** round {debate_round} majority: \n'
#                   f'inputs: {inputs[data_id]}\n'
#                   f'groundtruth: {groundtruth_outputs[data_id]}\n'
#                   f'majority score: {majority_score}\n'
#                   f'majority list: {majority_list}\n'
#                   f'majority_missing_list: {majority_missing_list}\n'
#                   f'majority_redundant_list: {majority_redundant_list}')
#             print(f'*** round {debate_round} intersection: \n'
#                   f'inputs: {inputs[data_id]}\n'
#                   f'groundtruth: {groundtruth_outputs[data_id]}\n'
#                   f'intersection score: {intersection_score}\n'
#                   f'intersection list: {intersection_list}\n'
#                   f'intersection_missing_list: {intersection_missing_list}\n'
#                   f'intersection_redundant_list: {intersection_redundant_list}')

def get_debate_output(agent_num, num_data, debate_responses, title2idx, round_idx):
    # debate_responses: [num_agent, num_data] of <START>{extracted_output}<END>, or [num_agent] of None
    print('getting debate output')
    if round_idx == 0:
        print('obtain debate output')
        return None
    else:
        threshold = math.ceil(agent_num / 2)
        candidates_list = [{} for i in range(num_data)]
        for data_idx in range(num_data):
            for agent_idx in range(agent_num):
                one_response = debate_responses[agent_idx][data_idx]
                items = convert_response_to_item_titles(extract_matched_output(one_response), title2idx)
                for item in items:
                    if item not in candidates_list[data_idx]:
                        candidates_list[data_idx][item] = 0
                    candidates_list[data_idx][item] += 1
        # collecting corrected inputs with majority voting and intersection of sets
        majority_voting_candidates_list = [[] for i in range(num_data)]
        for data_idx, candidates in enumerate(candidates_list):
            for item, count in candidates.items():
                if count >= threshold:
                    majority_voting_candidates_list[data_idx].append(item)
        majority_voting_result = [[] for i in range(num_data)]
        for data_idx, candidates in enumerate(majority_voting_candidates_list):
            majority_voting_result[data_idx] = itemList2Str(candidates)
        # list of strings: item;item;item or NONE
        print('obtain debate output')
        return majority_voting_result


# def eval_debate(agent_num, debate_responses, num_data, groundtruth_outputs, str2idx, config):
#     # collecting results
#     threshold = math.ceil(agent_num / 2)
#     candidates_list = [{} for i in range(num_data)]
#     for agent_id in range(agent_num):
#         last_responses = debate_responses[agent_id][-1]
#         for data_idx, response in enumerate(last_responses):
#             items = convert_response_to_item_titles(response, str2idx)
#             for item in items:
#                 if item not in candidates_list[data_idx]:
#                     candidates_list[data_idx][item] = 0
#                 candidates_list[data_idx][item] += 1
#
#     # collecting corrected inputs with majority voting and intersection of sets
#     majority_voting_candidates_list = [[] for i in range(num_data)]
#     intersection_candidates_list = [[] for i in range(num_data)]
#     for data_idx, candidates in enumerate(candidates_list):
#         for item, count in candidates.items():
#             if count >= threshold:
#                 majority_voting_candidates_list[data_idx].append(item)
#             if count == agent_num:
#                 intersection_candidates_list[data_idx].append(item)
#
#     for data_idx, candidates in enumerate(candidates_list):
#         majority_voting_candidates_list[data_idx] = ids2str(majority_voting_candidates_list[data_idx])
#         intersection_candidates_list[data_idx] = ids2str(intersection_candidates_list[data_idx])
#     print('######## majority voting eval')
#     if config['verbose']:
#         print(majority_voting_candidates_list)
#     get_eval_scores(majority_voting_candidates_list, groundtruth_outputs, str2idx)
#     print('######## intersection eval')
#     if config['verbose']:
#         print(intersection_candidates_list)
#     get_eval_scores(intersection_candidates_list, groundtruth_outputs, str2idx)
#     end = perf_counter()
#
#     return intersection_candidates_list


def generate_corrected_data(instance_ids, corrected_inputs, target_ids, config, idx2title, title2idx, model_name, file_name):
    # corrected_inputs: list of strings: item; item; item
    output_ids = []
    output_strs = []
    assert len(corrected_inputs) == len(target_ids)
    for instance_idx, relevant_input_seq, target_id in zip(instance_ids, corrected_inputs, target_ids):
        target_str = idx2title[target_id]
        if relevant_input_seq in ['None', 'NONE']:
            output_strs.append(f'{instance_idx}:::NONE:::{target_str}\n')
            output_ids.append(f'{instance_idx}:::NONE:::{target_id}\n')
        else:
            input_id_seq = string_seq2id_seq(title2idx, relevant_input_seq)
            output_ids.append(f'{instance_idx}:::{input_id_seq}:::{target_id}\n')
            output_strs.append(f'{instance_idx}:::{relevant_input_seq}:::{target_str}\n')

    corrected_dir = config['data_dir'] + f'/{model_name}/corrected/'
    if not os.path.exists(corrected_dir):
        os.makedirs(corrected_dir)

    path_for_output_ids = f'{corrected_dir}/{file_name}'
    with open(path_for_output_ids, 'w') as fout:
        fout.writelines(output_ids)
    print(f'output ids saved at {path_for_output_ids}')

    path_for_output_strs = f'{corrected_dir}/detail_{file_name}'
    with open(path_for_output_strs, 'w') as fout:
        fout.writelines(output_strs)
    print(f'output strs saved at {path_for_output_strs}')
