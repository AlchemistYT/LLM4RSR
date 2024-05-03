from apo4rec.util import generate_responses_and_extract_outputs

augment_prompt = "Generate a variation of the following original content while keeping the semantic meaning with no more than 150 words.\n" \
                 "Your generated content should be wrapped with <answer> and </answer>\n" \
                 "<original_content>$input$</original_content>\n" \
                 "Output:"


class Augmentor:

    def __init__(self, config):
        self.config = config

    def replace_placeholders(self, template, prompt):
        replaced_template = template.replace("$input$", prompt)
        return replaced_template

    def augment_prompt(self, prompts, generator):
        print(' ------------------------- augmenting prompts ------------------------- ')
        prompts_for_aug_instruction = []
        prompts_for_aug_profile = []
        prompts_for_aug_memory = []
        for prompt in prompts:
            prompt_for_aug_instruction = self.replace_placeholders(augment_prompt, prompt['instruction'])
            prompt_for_aug_profile = self.replace_placeholders(augment_prompt, prompt['profile'])
            # prompt_for_aug_memory = self.replace_placeholders(augment_prompt, prompt['memory'])
            prompts_for_aug_instruction.append(prompt_for_aug_instruction)
            prompts_for_aug_profile.append(prompt_for_aug_profile)
            # prompts_for_aug_memory.append(prompt_for_aug_memory)
        augmented_instructions = generate_responses_and_extract_outputs(prompts_for_aug_instruction, generator, self.config)
        augmented_profiles = generate_responses_and_extract_outputs(prompts_for_aug_profile, generator, self.config)
        # augmented_memories = generate_responses_and_extract_outputs(prompts_for_aug_memory, generator, self.config)

        augmented_prompts = []
        for prompt, augmented_instruction, augmented_profile in zip(prompts, augmented_instructions, augmented_profiles):
            new_prompt = {
                'profile': augmented_profile,
                'instruction': augmented_instruction,
                'restriction': prompt['restriction'],
                'example': prompt['example'],
            }
            augmented_prompts.append(new_prompt)

        return prompts + augmented_prompts
