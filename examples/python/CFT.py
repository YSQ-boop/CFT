

import re
import json
import os
from collections import Counter

import torch
from rouge_score import rouge_scorer


rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order."""
    if list1 == list2:
        return 1.0
    
    if os.getenv("REFINEDREWARD", 0) == "1":
        print("REFINEDREWARD is set to 1, so strict match is used")
        if list1 != list2:
            return 0.0
    
    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)  # Frequency count for list1
    count2 = Counter(list2)  # Frequency count for list2

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0
    

# custoimzed reward functions: format
def customize_format_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step >= 30:
            max_possible_reward = max_possible_reward / 2
            min_possible_reward = min_possible_reward / 2
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    # schedule reward
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = 2 - (2 - max_possible_reward) * step / 50
        min_possible_reward = -2 + (2 + min_possible_reward) * step / 50
        if max_possible_reward < 1.0:
            max_possible_reward = 1.0
        if min_possible_reward > -1.0:
            min_possible_reward = -1.0
    
    rewards = []
    responses = [completion[0]['content'] for completion in completions]


    for response, ans in zip(responses, answer):
        reward = min_possible_reward
        if "<response>" in ans and "<tool_call>" not in ans:
            pattern = r"^<think>\n[\s\S]*?\n</think>\n\n<response>\n.*?\n</response>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<response>") == 1 and response.count("</response>") == 1:
                reward = max_possible_reward
        elif "<response>" not in ans and "<tool_call>" in ans:
            pattern = r"^<think>\n[\s\S]*?\n</think>\n\n<tool_call>\n.*?\n</tool_call>(?:\n<tool_call>\n.*?\n</tool_call>)*$"
            if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == response.count("</tool_call>"):
                reward = max_possible_reward
        
        rewards.append(reward)
        
    print("\n======= Reward for <format> =======")
    print("Reward function for <format> is called ...")
    print(rewards)
    return rewards


# customized reward functions: length
def customize_length_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    # schedule length
    if os.getenv("SCHEDULELENGTH", 0) == "1":
        print("SCHEDULELENGTH is set to 1, so schedule max reward for length is used")
        max_reward_len = (640 - 384) * step / 105 + 384
    else:
        max_reward_len = 512
    
    """Reward function that gives higher scores to longer completions."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        if "<think>" not in response or "</think>" not in response:
            rewards.append(min_possible_reward)
            continue
        think_responses = response.split("<think>")[-1].split("</think>")[0].strip()
        reward = round(len(think_responses.split()) / max_reward_len, 2)
        if reward > 1.0:
            reward = 1.0
        
        final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
        rewards.append(final_reward)
    
    print("\n======= Reward for <length> =======")
    print("Reward function for <length> is called ...")
    print(rewards)
    return rewards
                

def get_rouge_score(p_value: str, gt_value: str) -> float:
    if p_value == gt_value:
        return 1.0
    if gt_value.strip() == "" or p_value.strip() == "":
        return 0.0
    return rouge.score(gt_value, p_value)["rougeL"].fmeasure


def compute_param_value_score(gt_params, pd_params):
    score = 0.0
    for key, gt_value in gt_params.items():
        if key not in pd_params:
            continue

        pd_value = pd_params[key]
        if isinstance(pd_value, str) and isinstance(gt_value, str):
            score += get_rouge_score(pd_value, gt_value)
        elif isinstance(pd_value, (int, float, bool)) and isinstance(gt_value, (int, float, bool)):
            score += 1.0 if pd_value == gt_value else 0.0
        else:
            score += 1.0 if str(pd_value) == str(gt_value) else 0.0

    return score


def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    if gt_tools == pd_tools:
        print("Max possible score:", "Exact Match!")
        print("Score:", max_possible_reward)
        return max_possible_reward
    
    if os.getenv("COARSEREWARD", 0) == "1":
        print("COARSEREWARD is set to 1, so coarse reward is used")
        if gt_tools != pd_tools:
            return min_possible_reward

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))
    
    local_max_possible = 1.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["arguments"]
        
        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            print("INTERMEDIATEREWARD is set to 1, so local max possible is changed")
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0 + len(gt_params)
        
        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue
            
            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if gt_tool == pd_tool:
                    best_match = pd_tool
                    best_match_index = i
                    best_match_score = 1.0
                    break
                else:
                    continue
            
            pd_params = pd_tool["arguments"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            
            # Calculate correctness score for parameter values
            correctness_score = compute_param_value_score(gt_params, pd_params)

            total_score = param_score + correctness_score
            
            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    print()
    print("Max possible score:", local_max_possible)
    print("Score:", score)
    
    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


# custoimzed reward functions: tool call correctness
def customize_correctness_reward_tool(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = (max_possible_reward - 2) * step / 50 + 2
        min_possible_reward = (min_possible_reward + 2) * step / 50 - 2
        if max_possible_reward > 3.0:
            max_possible_reward = 3.0
        if min_possible_reward < -3.0:
            min_possible_reward = -3.0
    
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        reward = 0.0
        
        if "<tool_call>" not in ans:
            # if "<tool_call>" not in response and "</tool_call>" not in response:
            #     reward = max_possible_reward
            # else:
            #     reward = min_possible_reward
            rewards.append(reward)
            continue

        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, ans, re.DOTALL)
        gt_tools = []
        for match in matches:
            try:
                gt_tools.append(json.loads(match.strip()))
            except json.JSONDecodeError as e:
                print(f"JSON decode error for gt_tools: {e}")
        
        try:
            # Change here as a constrint in training: if the format is not correct, directly give the lowest possible score
            assert "<tool_call>" in response
            assert "</tool_call>" in response
            matches = re.findall(pattern, response, re.DOTALL)
            pd_tools = []
            for match in matches:
                pd_tools.append(json.loads(match.strip()))
            reward = compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward) # top reward is 2
        except:
            reward = min_possible_reward
        
        rewards.append(reward)
    
    print("\n======= Reward for <tool call> =======")
    print("Reward function for <tool call> correctness is called ...")
    print(rewards)
    return rewards


def reward_func(queries, prompts, labels, **kwargs):
    """
    Reward function for calculating rewards of model outputs.

    Args:
        queries (torch.Tensor): Complete text sequences containing prompts and responses
        prompts (torch.Tensor): Input prompt sequences
        labels (torch.Tensor): Ground truth answer sequences
        **kwargs: Additional optional arguments

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - rewards: Reward values used for calculating advantage function
            - scores: Reward values in range [0,1] used for dynamic filtering
            - extra_logs: Additional information to be logged in wandb
    """

    step = 50
    exp_name = str(os.getenv("EXPERIMENT_NAME", ""))
    if "llama" in exp_name:
        pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>'
    elif "qwen" in exp_name:
        pattern = r'<\|im_start\|>assistant(.*?)<\|im_end\|>'
    completions = []
    for query in queries:
        matches = re.findall(pattern, query, re.DOTALL)
        
        if not matches:  # 无匹配时可选操作
            completions.append([{"role": "assistant", "content": ""}])
        else:
            completions.append([{"role": "assistant", "content": matches[-1].strip()}])
    
    if str(os.getenv("CORRECTMAX1", 0)) == "1":
        print("CORRECTMAX1 is set to 1, so max score is set to 1")
        tool_max_possible = 1.0
        tool_min_possible = -1.0
    else:
        tool_max_possible = 3.0
        tool_min_possible = -3.0
    
    format_max_possible = 1.0
    format_min_possible = 0.0
    
    length_max_possible = 1.0
    length_min_possible = 0.0
    
    
    fomrat_score = customize_format_reward_func(completions, labels, step, format_max_possible, format_min_possible)
    correctness_score = customize_correctness_reward_tool(completions, labels, step, tool_max_possible, tool_min_possible)
    
    if str(os.getenv("WITHLENGTH", 0)) == "1":
        print("WITHLENGTH is set to 1, so length score is set!")
        length_score = customize_length_reward_func(completions, labels, step, length_max_possible, length_min_possible)
    else:
        length_score = 0.0
    if not isinstance(length_score, list):
        length_score = [length_score] * len(queries)
    fomrat_score = torch.tensor(fomrat_score)
    correctness_score = torch.tensor(correctness_score)
    length_score = torch.tensor(length_score)
    score = fomrat_score + correctness_score + length_score
    return {
        "rewards": score,  
        "scores": score,  
        "extra_logs": {
                        "dummy_format_score": fomrat_score,
                        "dummy_correctness_score": correctness_score, 
                        "dummy_length_score": length_score
                        }
    }

