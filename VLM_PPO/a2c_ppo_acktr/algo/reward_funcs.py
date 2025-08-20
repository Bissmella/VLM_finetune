import json
from math_verify import LatexExtractionConfig, parse, verify, StringExtractionConfig 
import torch
import re

def format_reward(completions, possible_actions):
    """Reward function that checks if the completion has the required JSON keys."""
    completion_contents = completions
    rewards_list = []
    for content in completion_contents:
        content = content.lower()
        match = re.search(r"```(?:json)?\n(.*?)\n```", content, re.DOTALL)
        if match:
            content = match.group(1)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "thoughts" in parsed and "action" in parsed:
                if parsed["action"] in possible_actions:
                    rewards_list.append(0.0)
                else:
                    rewards_list.append(-0.5)
            else:
                rewards_list.append(-0.7)
        except json.JSONDecodeError:
            rewards_list.append(-1.0)
    return torch.tensor(rewards_list)




def accuracy_reward(completions, solutions):
    """Reward function that compares JSON answer field with the ground truth."""
    completion_contents = completions #[completion[0]["content"] for completion in completions]
    rewards = []
    ncorrects = 0

    for content, solution in zip(completion_contents, solutions):
        content = content.lower()
        match = re.search(r"```(?:json)?\n(.*?)\n```", content, re.DOTALL)
        if match:
            content = match.group(1)
        try:
            parsed = json.loads(content)
            answer = parsed.get("action", "")
        except json.JSONDecodeError:
            rewards.append(0.0)
            continue

        gold_parsed = [solution.lower()]#parse(solution, extraction_mode="first_match", extraction_config=[StringExtractionConfig ()])
        answer_parsed = [answer.lower()] #parse(answer, extraction_mode="first_match", extraction_config=[StringExtractionConfig ()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(answer_parsed == gold_parsed))
                ncorrects += int(answer_parsed == gold_parsed)
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return torch.tensor(rewards), ncorrects