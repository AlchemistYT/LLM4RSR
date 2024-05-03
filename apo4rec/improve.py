import random
from apo4rec.eval import eval_a_prompt_with_debate
from apo4rec.util import generate_responses_and_extract_outputs, generate_responses, extract_summary_1, extract_summary_2
from apo4rec.data import subsample_data

from time import perf_counter

infer_error_reason_instruction = "I'm trying to write an instruction for an LLM agent to help it select relevant input $item_type$s for a target $item_type$.\n" \
                                 "My current instruction is \"$instruction$\"\n" \
                                 "But this instruction gets the following errors: " \
                                 "Some irrelevant item pairs are misclassified as relevant: $should_be_irrelevant_pairs$\n" \
                                 "Some relevant item pairs are misclassified as irrelevant: $should_be_relevant_pairs$\n" \
                                 "Summarize the reasons why the instruction could have gotten these errors with no more than 150 words.\n" \
                                 "Wrap reasons with '<answer>' and '</answer>'"

refine_plan = "I'm trying to write a prompt to select relevant input $item_type$s for a target $item_type$.\n" \
                     "My current prompt is \"$instruction$\"\n" \
                     "But this prompt gets the following errors: \n" \
                     "Some irrelevant item pairs are misclassified as relevant: $should_be_irrelevant_pairs$\n" \
                     "Some relevant item pairs are misclassified as irrelevant: $should_be_relevant_pairs$\n" \
                     "Based on these errors, the problem with this prompt is that $reasons$.\n" \
                     "Based on the above information, please write an improved prompt with no more than 150 words.\n" \
                     "The prompt should be wrapped with '<answer>' and '</answer>'.\n" \
                     "The new prompt is:"

infer_error_reason_profile = "I'm trying to write a profile (i.e., a system prompt) for an LLM agent to help it select relevant input $item_type$s for a target $item_type$.\n" \
                             "An ideal profile should ask the agent to act as a domain expert when solving the problem." \
                             "My current profile is \"$profile$\"\n" \
                             "But this profile gets the following errors:\n" \
                             "Some irrelevant item pairs are misclassified as relevant: $should_be_irrelevant_pairs$\n" \
                             "Some relevant item pairs are misclassified as irrelevant: $should_be_relevant_pairs$\n" \
                             "Summarize the reasons why the profile could have gotten these errors with no more than 150 words.\n" \
                             "Wrap the reasons with '<answer>' and '</answer>'"

refine_profile = "I'm trying to write a profile (i.e., a system prompt) for an LLM agent to help it select relevant input $item_type$s for a target $item_type$.\n" \
                 "An ideal profile should ask the agent to act as a domain expert when solving the problem." \
                 "My current profile is \"$profile$\"\n" \
                 "But this profile gets the following errors:\n" \
                 "Some irrelevant item pairs are misclassified as relevant: $should_be_irrelevant_pairs$\n" \
                 "Some relevant item pairs are misclassified as irrelevant: $should_be_relevant_pairs$\n" \
                 "Based on these examples, the problem with this profile is that $reasons$.\n" \
                 "Based on the above information, please write an improved profile with no more than 150 words.\n" \
                 "The profile should be wrapped with '<answer>' and '</answer>'.\n" \
                 "The new profile is:"

infer_error_reason_memory = "I'm trying to update the summary of $item_type$s for a recommender to find relevant $item_type$s.\n" \
                            "### The current summaries for [$item_type$_1: $input_item$] and [$item_type$_2: $target_item$] are as follows: \n" \
                            "<$item_type$_1_summary>$summary_1$</$item_type$_1_summary>\n" \
                            "<$item_type$_2_summary>$summary_2$</$item_type$_2_summary>\n" \
                            "These two $item_type$s should be $groundtruth$, but are misclassified as $prediction$\n" \
                            "Please infer the reason why the summary lead to the misclassification.\n" \
                            "The possible reasons can be: containing incorrect information, containing redundant information, or missing important information." \
                            "Please wrap all your inferred reasons between '<answer>' and '</answer>'"

refine_memory = "I'm trying to update the description of $item_type$s for a recommender to find relevant $item_type$s.\n" \
                "The current summaries for [$item_type$_1: $input_item$] and [$item_type$_2: $target_item$] are as follows: \n" \
                "<$item_type$_1>$summary_1$</$item_type$_1>\n" \
                "<$item_type$_2>$summary_2$</$item_type$_2>\n" \
                "These two $item_type$s should be $groundtruth$, but are misclassified as $prediction$\n" \
                "The reasons why the summary lead to the misclassification are as follows:\n" \
                "$reasons$\n" \
                "Based on these reasons, please write improved summaries for [$item_type$_1: $input_item$] and [$item_type$_2: $target_item$], respectively." \
                "Please wrap the summary for [$item_type$_1: $input_item$] between <summary_1> and </summary_1>, " \
                "and wrap the summary for [$item_type$_2: $input_item$] between <summary_2> and </summary_2>"



class Improver:
    def __init__(self,
                 train_data,
                 config,
                 idx2title,
                 title2idx,
                 idx2attribute,
                 title2summary,
                 title2attribute,
                 filter):
        self.train_data = train_data
        self.config = config
        self.idx2title = idx2title
        self.title2idx = title2idx
        self.idx2attribute = idx2attribute
        self.title2summary = title2summary
        self.title2attribute = title2attribute
        self.filter = filter

    def replace_placeholders(self, template, placeholder, prompt, should_be_irrelevant_pairs, should_be_relevant_pairs,
                             reasons=''):
        replaced_template = template \
            .replace("$item_type$", self.config['item_type']) \
            .replace(placeholder, prompt) \
            .replace("$should_be_irrelevant_pairs$", should_be_irrelevant_pairs) \
            .replace("$should_be_relevant_pairs$", should_be_relevant_pairs) \
            .replace("$reasons$", reasons)
        return replaced_template

    def construct_memory_prompt_infer_reason(self, item_pair, groundtruth, prediction):
        input_item, target_item = item_pair
        input_item = input_item.strip()
        target_item = target_item.strip()
        input_summary = self.title2summary[input_item]
        input_attribute = self.title2attribute[input_item]
        target_summary = self.title2summary[target_item]
        target_attribute =  self.title2attribute[target_item]
        replaced_template = infer_error_reason_memory \
            .replace("$item_type$", self.config['item_type']) \
            .replace("$input_item$", input_item) \
            .replace("$target_item$", target_item) \
            .replace("$summary_1$", input_summary) \
            .replace("$summary_2$", target_summary) \
            .replace("$groundtruth$", groundtruth) \
            .replace("$prediction$", prediction)

        return replaced_template

    def construct_memory_prompt_refine(self, item_pair, groundtruth, prediction, inferred_reason):
        input_item, target_item = item_pair
        input_summary = self.title2summary[input_item]
        target_summary = self.title2summary[target_item]
        replaced_template = refine_memory \
            .replace("$item_type$", self.config['item_type']) \
            .replace("$input_item$", input_item) \
            .replace("$target_item$", target_item) \
            .replace("$summary_1$", input_summary) \
            .replace("$summary_2$", target_summary) \
            .replace("$groundtruth$", groundtruth) \
            .replace("$prediction$", prediction) \
            .replace("$reasons$", inferred_reason)
        return replaced_template

    def update_memory(self, pairs, groundtruth, prediction, generator, config):
        # infer reasons
        infer_reason_prompts = []
        for pair in pairs:
            infer_reason_prompt = self.construct_memory_prompt_infer_reason(pair, groundtruth, prediction)
            infer_reason_prompts.append(infer_reason_prompt)
        inferred_reasons = generate_responses_and_extract_outputs(infer_reason_prompts, generator, config)
        # get new summaries based on reasons
        refine_prompts = []
        for pair, reason in zip(pairs, inferred_reasons):
            refine_prompt = self.construct_memory_prompt_refine(pair, groundtruth, prediction, reason)
            refine_prompts.append(refine_prompt)
        updated_memory_responses = generate_responses_and_extract_outputs(infer_reason_prompts, generator, config)
        # update memory based on new summaries
        for pair, updated_memory_response in zip(pairs, updated_memory_responses):
            input_item, target_item = pair
            summary_1 = extract_summary_1(updated_memory_response, self.config)
            summary_2 = extract_summary_2(updated_memory_response, self.config)
            if summary_1 in ['NONE', 'None']:
                print(f'new summary for {input_item} is None')
            else:
                print(
                    f'the summary of {input_item} is updated to \n new: [{summary_1}] \n old: [{self.title2idx[input_item]}]')
                self.title2idx[input_item] = summary_1
            if summary_2 in ['None', 'None']:
                print(f'new summary for {target_item} is None')
            else:
                print(
                    f'the summary of {input_item} is updated to \n new: [{summary_1}] \n old: [{self.title2idx[input_item]}]')
                self.title2idx[target_item] = summary_2


    def improve_memory(self, full_should_be_relevant_pairs, full_should_be_irrelevant_pairs, generator, config):
        print('improving memory')
        self.update_memory(full_should_be_relevant_pairs, groundtruth='relevant', prediction='irrelevant', generator=generator, config=config)
        self.update_memory(full_should_be_irrelevant_pairs, groundtruth='irrelevant', prediction='relevant', generator=generator, config=config)

    def improve_prompt(self, prompts, generator):
        """
        prompts: a list of dicts
        generator: the llm
        """
        print(' --------------- improving profile and plan -----------------')
        # infer error reasons
        print('# infer error reasons')
        infer_start = perf_counter()
        profile_reason_prompts = []
        plan_reason_prompts = []
        memory_reason_prompts = []
        should_be_relevant_pairs_list = []
        should_be_irrelevant_pairs_list = []
        full_should_be_relevant_pairs = []
        full_should_be_irrelevant_pairs = []
        # collet errors for each prompt (each profile and plan), and memory

        sample_data = subsample_data(self.train_data, self.config['select_sample_num'])
        for i, prompt in enumerate(prompts):
            score, should_be_relevant_pairs, should_be_irrelevant_pairs, \
                should_be_irrelevant_pairs_title_list, should_be_relevant_pairs_title_list = eval_a_prompt_with_debate(prompt,
                                                                                                                       sample_data,
                                                                                                                       generator,
                                                                                                                       self.config,
                                                                                                                       self.title2idx,
                                                                                                                       self.idx2title,
                                                                                                                       self.idx2attribute,
                                                                                                                       self.title2summary,
                                                                                                                       self.filter,
                                                                                                                       additional_str=f'improve eval, collecting errors for prompt {i}/{len(prompts)}')
            full_should_be_relevant_pairs.extend(should_be_relevant_pairs_title_list)
            full_should_be_irrelevant_pairs.extend(should_be_irrelevant_pairs_title_list)
            prompt_for_inferring_reasons_instruction = self.replace_placeholders(infer_error_reason_instruction,
                                                                                 '$instruction$',
                                                                                 prompt['instruction'],
                                                                                 should_be_irrelevant_pairs,
                                                                                 should_be_relevant_pairs)
            prompt_for_inferring_reasons_profile = self.replace_placeholders(infer_error_reason_profile,
                                                                             '$profile$',
                                                                             prompt['profile'],
                                                                             should_be_irrelevant_pairs,
                                                                             should_be_relevant_pairs)
            # prompt_for_inferring_reasons_memory = self.replace_placeholders(infer_error_reason_memory,
            #                                                                 '$memory$',
            #                                                                 prompt['memory'],
            #                                                                 should_be_irrelevant_pairs,
            #                                                                 should_be_relevant_pairs)
            plan_reason_prompts.append(prompt_for_inferring_reasons_instruction)
            profile_reason_prompts.append(prompt_for_inferring_reasons_profile)
            # memory_reason_prompts.append(prompt_for_inferring_reasons_memory)
            should_be_relevant_pairs_list.append(should_be_relevant_pairs)
            should_be_irrelevant_pairs_list.append(should_be_irrelevant_pairs)
        print('## infer error reasons for profile')
        inferred_reasons_profile = generate_responses_and_extract_outputs(profile_reason_prompts, generator, self.config)
        print('## infer error reasons for planning')
        inferred_reasons_plan = generate_responses_and_extract_outputs(plan_reason_prompts, generator, self.config)

        # print('## infer error reasons for memory')
        # inferred_reasons_memory = generate_responses_and_extract_outputs(memory_reason_prompts, generator, self.config)
        infer_end = perf_counter()
        print(f'# get error reasons in {infer_end - infer_start} seconds')

        assert len(inferred_reasons_plan) == len(inferred_reasons_profile) == len(prompts) == len(should_be_relevant_pairs_list) == len(
            should_be_irrelevant_pairs_list)

        # improve prompts according to error reasons
        print('# improve prompts w.r.t. error reasons')
        improve_start = perf_counter()
        prompts_for_improving_instruction = []
        prompts_for_improving_profile = []
        # prompts_for_improving_memory = []
        for prompt, inferred_reason_instruction, inferred_reason_profile, should_be_relevant_pairs, should_be_irrelevant_pairs in \
                zip(prompts, inferred_reasons_plan, inferred_reasons_profile,
                    should_be_relevant_pairs_list, should_be_irrelevant_pairs_list):
            prompt_for_improving_plan = self.replace_placeholders(refine_plan,
                                                                         '$instruction$',
                                                                  prompt['instruction'],
                                                                  should_be_irrelevant_pairs,
                                                                  should_be_relevant_pairs,
                                                                  inferred_reason_instruction)
            prompt_for_improving_profile = self.replace_placeholders(refine_profile,
                                                                     '$profile$',
                                                                     prompt['profile'],
                                                                     should_be_irrelevant_pairs,
                                                                     should_be_relevant_pairs,
                                                                     inferred_reason_instruction)
            # prompt_for_improving_memory = self.replace_placeholders(refine_memory,
            #                                                         '$memory$',
            #                                                         prompt['memory'],
            #                                                         should_be_irrelevant_pairs,
            #                                                         should_be_relevant_pairs,
            #                                                         inferred_reason_instruction)
            prompts_for_improving_instruction.append(prompt_for_improving_plan)
            prompts_for_improving_profile.append(prompt_for_improving_profile)
            # prompts_for_improving_memory.append(prompt_for_improving_memory)
        print('## improving instructions')
        improved_instructions = generate_responses_and_extract_outputs(prompts_for_improving_instruction, generator,
                                                                       self.config)
        print('## improving profiles')
        improved_profiles = generate_responses_and_extract_outputs(prompts_for_improving_profile, generator,
                                                                   self.config)
        # print('## improving memories')
        # improved_memories = generate_responses_and_extract_outputs(prompts_for_improving_memory, generator, self.config)

        improved_prompts = []
        for prompt, improved_instruction, improved_profile in zip(prompts, improved_instructions, improved_profiles):
            new_prompt = {
                'profile': improved_profile,
                'instruction': improved_instruction,
                'restriction': prompt['restriction'],
                'example': prompt['example'],
            }
            improved_prompts.append(new_prompt)

        prompts.extend(improved_prompts)
        improve_end = perf_counter()
        print(f'### improve prompts in {improve_end - improve_start} seconds')

        self.improve_memory(full_should_be_relevant_pairs, full_should_be_irrelevant_pairs, generator, self.config)

        return prompts
