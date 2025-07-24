import torch
import random
from typing import List
import re
import json

def get_prompt(env_name, action_only,sampling = False, infos = None):
    """
        This function defines the prompt for the text-to-action task, depending on the environments
        env_name: determines the prompts for each environment
        info: additional information that can be added to the prompt, if none, then use the default prompt
    """
    if env_name == 'gym_cards/NumberLine-v0':
        qs = "You are playing a game called number line. You will see a target number and a current number in the image. "
        qs = qs + "And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. "
        qs = qs + "Your response should be a valid json file in the following format: \n{\n "
        if not action_only:
            qs = qs + "\"current number\": \"x\", \n"
            qs = qs + "\"target number\": \"x\", \n"
            qs = qs + "\"thoughts\": \"{first read out the current and target number, then think carefully about which action to choose}\", \n"
        qs = qs + "\"action\": \"-\" or \"+\" \n}"
    elif env_name == 'gym_cards/Blackjack-v0':
        qs = "You are a blackjack player. You are observing the current game state, you can choose between ['stand', 'hit']. "
        qs = qs + "Your response should be a valid json file in the following format: \n{\n "
        if not action_only:
            qs = qs + "\"thoughts\": \"{first describe your total points and the dealer's total points then think about which action to choose}\", \n"
        qs = qs + "\"action\": \"stand\" or \"hit\" \n}"
    elif env_name == 'gym_cards/EZPoints-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert card game player. You are observing two cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 12, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: \{\n"
        if not action_only:
            qs = qs + " \"cards\": [x, y], \n"
            qs = qs + f"\"current formula\": {text_formula}, \n"
            qs = qs + "\"thoughts\": {First check whether the current formula 'z' is complete. "
            qs = qs + "If the current formula 'z' is complete, output '='. "
            qs = qs + "Otherwise consider which number or operator should be appended to the current formula to make it equal 12.} \n"
        qs = qs + "\"action\": \"{number}\" or \"{operator}\" \n \}"
    elif env_name == 'gym_cards/Points24-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert 24 points card game player. You are observing thee four cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 24, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: \{\n"
        if not action_only:
            qs = qs + " \"cards\": [x, y, z, w], \n"
            qs = qs + f"\"current formula\": {text_formula}, \n"
            qs = qs + "\"thoughts\": {First check whether the current formula equals 24. "
            qs = qs + "If the current formula equals 24, output '='. "
            qs = qs + "Otherwise consider which number or operator should be appended to the current formula to make it equal 24.} \n"
        qs = qs + "\"action\": \"{number}\" or \"{operator}\" \n \}"
    elif "MiniGrid-MultiRoom" in env_name:
        # try:
        #     direction = ''.join(str(infos[0]['direction']))
        # except:
        #     direction = ''
        # try:
        #     mission = ''.join(str(infos[0]['mission']))
        # except:
        #     mission = ''
        qs = "You are an expert 2d game player. You are observing a series of connected rooms with doors that must be opened to get to next room in image. "
        qs = qs + "Your goal is to get to the green goal square."
        # qs = qs + f"\"current direction\": {direction}, \n"
        # qs = qs + f"\"mission\": {mission}, \n"
        qs = qs + "At each step you can choose one of these actions ['Turn left', 'Turn right', 'Move forward', 'Toggle']"
        qs = qs + "Your response should be a valid json file in the following format: \{\n"
        qs = qs + "\"thoughts\": \"{first read out the current status from image, then think carefully about which action to choose}\", \n"
        qs = qs + "\"action\": \"{your choosen action}\" \n}"
    elif "MiniGrid-DoorKey" in env_name:
        if sampling:
            qs = "You are an expert 2D game player in a grid-based environment. The environment has a key that to pick up in order to unlock a door represented as blue square with a minus, and then reach the green goal square. "
            qs += "You are observing the image of the current state, and your goal is to get the player to the green goal square. The player is shown by a blue arrow."
            #qs += "At each step, you can choose one of these actions: ['Turn left', 'Turn right', 'Move forward', 'Pick up', 'Toggle'].\n"
            qs += "Please evaluate each action based on the current observation and assign a score between 1 (very bad) and 5 (very good), reflecting how useful or promising each action is in reaching the goal. Use only the available information and image.\n"
            qs += "Return your answer as a valid JSON object in the following format:\n{\n"
            qs += '  "thoughts": "Describe the current scene step-by-step. First, summarize the visible area concisely. Then, specify the exact position and facing direction of the player (marked by an arrow). List nearby objects, or interactive elements with their relative locations. Do no mention what action to take yet.",\n' #Briefly describe only the current scene, position and direction of the agent shown by an arrow, and other objects by describing the current image.
            #qs = qs + "\"top_action_analysis\": \"Based on the above scene, what will be the consequences of each action."
            qs += '  "action_scores": {\n'
            qs += '    "Turn left": score_1,\n'
            qs += '    "Turn right": score_2,\n'
            qs += '    "Move forward": score_3,\n'
            qs += '    "Pick up": score_4,\n'
            qs += '    "Toggle": score_5\n'
            qs += "  }\n"
            qs += "}"
        else:
            qs = "You are an expert 2D game player in a grid-based environment. The environment has a key that to pick up in order to unlock a door represented as blue square with a minus, and then reach the green goal square. "
            #qs = "Rules: you may need to pick up a key to open a locked door, you can only interact with adjacent tiles in the direction you are facing, you can only pass through open doors."
            qs += "You are observing the image of the current state, and your goal is to get the player to a green goal tile. The player is shown by a blue triangle."
            
            qs = qs + "At each step you can choose one of these actions ['Turn left', 'Turn right', 'Move forward', 'Pick up', 'Toggle']"
            qs = qs + "Your response should be a valid json object in the following format: \{\n"
            #qs = qs + "\"action\": \"your choosen action\" "
            qs = qs + "\"thoughts\": \"Describe the current scene step-by-step. First, summarize the visible area concisely. Then, specify the exact position and facing direction of the player (marked by an arrow). List nearby objects, or interactive elements with their relative locations.\", \n"
            #qs = qs + "\n}"
            #qs = qs + "\"top_action_analysis\": \"Based on the above scene, list some plausible next actions the player might take, along with reasoning for each. Do not choose one yet."
            qs = qs + "\"action\": \"your choosen action\" \n}"
        
    return qs

# Define the function that processes the list of strings according to the specified rules
def text_projection(text_actions: List[str], env_name):
    output_indices = []
    random_mask = []
    commands = []
    if env_name == 'gym_cards/NumberLine-v0':
        action_list = ["-", "+"]
    elif env_name == 'gym_cards/Blackjack-v0':
        action_list = ["stand", "hit"]
    elif env_name == 'gym_cards/EZPoints-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    elif env_name == 'gym_cards/Points24-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
    elif "MiniGrid-MultiRoom" in env_name:
        action_list = ["Turn left", "Turn right", "Move forward", "Unused", "Unused", "Toggle"]
    elif "MiniGrid-DoorKey" in env_name:
        action_list = ["Turn left", "Turn right", "Move forward", "Pick up", "Unused",  "Toggle", "Unused"]
    else:
        raise NotImplementedError("Action list not implemented for this env!")
    action_to_id ={}
    id_to_action = {}
    for i, act in enumerate(action_list):
        if act != "Unused":
            id_to_action[i] = act
            action_to_id[act] = i
    for string in text_actions:
        if not isinstance(string, str):
            # directly output a random action if the string is not a string
            output_indices.append(random.choice(list(id_to_action.keys())))
            random_mask.append(1)
            commands.append(action_list[output_indices[-1]])
            continue
        string = string.lower()
        action_index = string.find('"action":')
        # Extract everything after "action":
        string = string[action_index:]
        contained_actions = []
        # For the 'gym_cards/Points24-v0' environment, handle '10' separately
        if 'points' in env_name.lower() and '10' in string:
            contained_actions.append('10')
            string = string.replace('10', '')  # Remove '10' to prevent it from being counted as '1'
        # Find all actions that are contained in the string
        # for action in action_list:
        #     if action in string:
        #         contained_actions.append(action)
        
        for action in action_list:
            if re.search(r'[a-zA-Z0-9]', action):
                pattern = r'\b' + re.escape(action) + r'\b'
                pattern = rf'(?i)\b{pattern}\b|(?i)"\s*{pattern}\s*"|(?i)\[\s*{pattern}\s*\]'
            else:
                pattern = re.escape(action)##pattern = r'\b' + re.escape(action) + r'\b'
            if re.search(pattern, string):
                contained_actions.append(action)
            
        # Remove duplicates by converting to a set and back to a list
        contained_actions = list(set(contained_actions))
        
        if len(contained_actions) == 1 and contained_actions[0] in action_list:
            # Only one keyword from action_list is in the string
            output_indices.append(action_list.index(contained_actions[0]))
            random_mask.append(0)
        else:
            # The string contains none or multiple keywords, randomly select an index from action_list
            output_indices.append(random.choice(list(id_to_action.keys())))
            random_mask.append(1)
        commands.append(action_list[output_indices[-1]])
    return torch.Tensor([output_indices]).long().reshape(-1, 1), torch.Tensor([random_mask]).long().reshape(-1, 1), commands

def text_projection_pr(text_actions: List[str], env_name, action_sampling = False):
    
    if not action_sampling:
        return text_projection(text_actions, env_name)
    else:
        output_indices = []
        random_mask = []
        commands = []
        
        if env_name == 'gym_cards/NumberLine-v0':
            action_list = ["-", "+"]
        elif env_name == 'gym_cards/Blackjack-v0':
            action_list = ["stand", "hit"]
        elif env_name == 'gym_cards/EZPoints-v0':
            action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                        "+", "*", "="]
        elif env_name == 'gym_cards/Points24-v0':
            action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                        "+", "-", "*", "/", "(", ")", "="]
        elif "MiniGrid-MultiRoom" in env_name:
            action_list = ["Turn left", "Turn right", "Move forward", "Unused", "Unused", "Toggle"]
        elif "MiniGrid-DoorKey" in env_name:
            action_list = ["Turn left", "Turn right", "Move forward", "Pick up", "Unused1",  "Toggle", "Unused2"]
        else:
            raise NotImplementedError("Action list not implemented for this env!")
        action_to_id ={}
        id_to_action = {}
        for i, act in enumerate(action_list):
            if act != "Unused":
                id_to_action[i] = act
                action_to_id[act] = i
        
        for string in text_actions:
            try:
                string = string.lower()
                response_json = json.loads(string)
                action_scores = response_json["action_scores"]
                
                
                scores_r = [action_scores.get(action.lower(), 0) if "Unused" not in action else -float('inf') for action in action_list]
                scores = torch.tensor(scores_r, dtype=torch.float32)
                # Softmax for sampling
                probs = torch.nn.functional.softmax(scores / 2.5, dim=0)  #TODO 1.5 is a temperature hardcoded
                
                sampled_index = torch.multinomial(probs, num_samples=1).item()
                chosen_action = action_list[sampled_index]
                output_indices.append(sampled_index)
                commands.append(chosen_action)
                random_mask.append(0)
                # print("Sampled Action:", chosen_action)
                # print("Probabilities:", probs)
            except:
                scores = [1 if "Unused" not in action else -float('inf') for action in action_list]
                probs = torch.nn.functional.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)
                sampled_index = torch.multinomial(probs, num_samples=1).item()
                chosen_action = action_list[sampled_index]
                output_indices.append(sampled_index)
                commands.append(chosen_action)
                random_mask.append(1)
                # print("JSON parse error:", e)
        
        return torch.Tensor([output_indices]).long().reshape(-1, 1), torch.Tensor([random_mask]).long().reshape(-1, 1), commands


# def generate_fake_response(outputs: List[str], commands: List[str], env_name=""):
#     new_outputs = []
#     for output, command in zip(outputs, commands):
#         try:
#             output_trimmed = output.trim("action_scores:")[0]
#             output_trimmed = output_trimmed + f" \"\{ {command} \}\" \n}"
#             new_outputs.append(output_trimmed)
#         except:
#             new_outputs.append(output)
#     return new_outputs

def generate_fake_response(outputs: List[str], commands: List[str], env_name: str =""):
    new_outputs = []
    for output, command in zip(outputs, commands):
        try:

            # Optionally remove any trailing 'action_scores' section or close the JSON cleanly
            if "action_scores" in output:
                # Trim up to the action_scores part (or you can do a better structured parse)
                output_trimmed = output.split('"action_scores":')[0].strip()
            else:
                output_trimmed = output.strip()
            #output_trimmed = output_trimmed.replace('"scene_description":', '"thoughts":')
            # Create a properly formatted JSON-style string
            fake_response = (
                f'{output_trimmed} \n "action": "{command}"\n' + "}"
            )
            new_outputs.append(fake_response)
        except Exception as e:
            print(f"[Warning] Failed to process output: {e}")
            new_outputs.append(output)
    return new_outputs
