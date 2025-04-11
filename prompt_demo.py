import json
import os
from typing import List, Dict, Any, Tuple, Union

from llama3_utils import construct_prompt, Message, Dialog


##############################################
#                   Main                     #
##############################################

def build_demo(
        demo_indices: List[int],
        demonstration_path: str,
        dataset_name: str
):
    ########## Add new dataset here ##########
    dataset_to_build_demo = {
        "hate-speech-dataset": build_demo_hate
    }
    ###########################################
    prompt_demo = dataset_to_build_demo[dataset_name](
        demo_indices=demo_indices,
        demonstration_path=demonstration_path,
    )
    return prompt_demo


def build_prompt(
        example: Dict[str, Any],
        prompt_demo: str,
        dataset_name: str,
):
    ########## Add new dataset here ##########
    dataset_to_build_prompt = {
        "hate-speech-dataset": build_prompt_hate
    }
    ###########################################
    prompt_text = dataset_to_build_prompt[dataset_name](
        example=example,
        prompt_demo=prompt_demo,
        instruct=True
    )
    return prompt_text


##############################################
#             hate-speech-dataset            #
##############################################

def build_demo_hate(
        demo_indices: List[int],
        demonstration_path: str,

):
    import llm_prompts_related.prompt_templates.template_hate as template
    # load demonstrations
    demos = []
    for i, index in enumerate(demo_indices):
        curr = dict()
        with open(os.path.join(demonstration_path, f"raw_{index}.txt"), 'r', encoding='utf-8') as f:
            curr['raw'] = f.read().strip()
        with open(os.path.join(demonstration_path, f"triple_{index}.txt"), 'r', encoding='utf-8') as f:
            curr['triple'] = f.read().strip()
        demos.append(curr)
    return demos


def build_prompt_hate(
        example: Dict[str, Any],
        prompt_demo: str,
        instruct: bool,
):
    import llm_prompts_related.prompt_templates.template_hate as template


    INSTRUCTION = template.INSTRUCTION_COT
    PLACE_HOLDER = "Sure! I am happy to help you solve this problem. Here is the answer:\n{}\n"
    system_message = "You are a helpful assistant that helps people solve problems.\n"
    problem_text = example['raw']

    if instruct:
        dialog = []
        dialog.append({"role": "system", "content": system_message})
        dialog.append({
            "role": "user", "content": f"{INSTRUCTION}\n\nHere is the problem:\n\n{prompt_demo[0]['raw']}\n"
        })
        dialog.append({
            "role": 'assistant', "content": PLACE_HOLDER.format(prompt_demo[0]['triple'])
        })
        for i in range(1, len(prompt_demo)):
            dialog.append({
                "role": 'user',
                "content": f"Excellent work! Here is another problem for you to solve. Please apply the same approach you used for the previous one(s) to tackle this new one. \nProblem:\n{prompt_demo[i]['raw']}\n"
            })
            dialog.append({
                "role": 'assistant', "content": PLACE_HOLDER.format(prompt_demo[i]['triple'])
            })
        dialog.append({
            'role': "user",
            "content": f"Excellent work! Here is another problem for you to solve. Please apply the same approach you used for the previous one(s) to tackle this new one. \nProblem:\n{problem_text}\n"
        })

        dialog_texts = construct_prompt(dialog)
        # output_prompt = '\n'.join(dialog_texts)
        output_prompt = dialog_texts
        output_prompt += "\n### Problem No.{}\n\n```{}```\n".format(len(prompt_demo) + 1, problem_text)
    else:
        pass


    return output_prompt



##############################################
#                crows_pairs                 #
##############################################

def build_demo_crowspairs():
    raise NotImplementedError


def build_prompt_crowspairs():
    raise NotImplementedError




if __name__ == '__main__':
    from pathlib import Path

    project_root = Path('.')
    project_root = project_root.cwd()
    relative_path = Path('llm_prompts_related/icl_demonstrations/hate-speech-dataset')

    demo = build_demo([0, 1, 2], str(project_root / relative_path), 'hate-speech-dataset')
    # print(demo)
    example = {"raw": "There is nothing I would love to see more than the arrest , trial and execution of these murderous and genocidal Zionists !"}
    # print(example)
    prompt = build_prompt(example, demo, "hate-speech-dataset")
    print(prompt)
