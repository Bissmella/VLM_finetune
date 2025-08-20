from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()
print("using xformers")

import copy
import glob
import os
import time
from collections import deque
from pathlib import Path
import minigrid
import gymnasium as gym
import gym_cards
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
torch.backends.cuda.enable_mem_efficient_sdp(False)  #disabling cutlass not working on small gpus
torch.backends.cuda.enable_flash_sdp(False)
import re


from a2c_ppo_acktr import algo, utils, rl_utils

from a2c_ppo_acktr.rl_utils import get_prompt, text_projection, text_projection_pr
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue, QwenVLMValue, QwenVLMPolicy, QwenTempPredictor
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate, qwen_batch_process_multiIm
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

import math
import random
from functools import partial
from typing import List, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
import transformers

import json
from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

import warnings
warnings.filterwarnings("ignore")

def cosine_schedule(step, T_max, eta_min, base_lr):
    """
    Returns the LR multiplier at step `step`, where:
    - T_max is the total number of steps (like in CosineAnnealingLR)
    - eta_min is the final LR (absolute)
    - base_lr is the starting LR (absolute)

    Output is a multiplier relative to base_lr.
    """
    if step >= T_max:
        return eta_min / base_lr  # stays flat after schedule ends
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / T_max))
    return (eta_min + (base_lr - eta_min) * cosine_decay) / base_lr

def get_trainable_params( model, return_with_names=False):
        if return_with_names:
            return filter(lambda p: p[1].requires_grad, model.named_parameters())
        else:
            return filter(lambda p: p.requires_grad, model.parameters())
GLOBALSTEP =0
DATA_PATH = "/home/bahaduri/RL4VLM/outputs/labeled_data"
def save_image_action(traj, step, image, action=None):
    global GLOBALSTEP
    if step == 0:
        step = step + GLOBALSTEP + 1
    else:
        step = GLOBALSTEP + 1
    GLOBALSTEP = step
    trajNum= traj
    output_path = "/home/bahaduri/RL4VLM/outputs/trajs_col"
    folder_path = output_path + f"/{trajNum}"
    actions_file = folder_path + "/actions"
    actions_file = Path(actions_file)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not actions_file.exists():
        actions_file.touch()
    image.save(folder_path + f"/{step}.png")
    if action is not None:
        action = action[0]
        with open(actions_file, 'a') as f:
            f.write(action + '\n')

def save_captioned_images(img1, text1, text2, num, img2=None):
    if img2 is None:
        img2 = img1.copy() 
    text2 = str(text2)
    total_width = img1.width + img2.width
    text_space = 60  # height reserved for text area below
    max_height = max(img1.height, img2.height) + text_space

    combined = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))

    # === Draw text under each image ===
    draw = ImageDraw.Draw(combined)

    # Optional: use custom font
    # font = ImageFont.truetype("arial.ttf", size=20)
    font = ImageFont.load_default()

    # Center text below each image
    text_y = img1.height + 10
    text_width1, _ = font.getsize(text1)
    text_width2, _ = font.getsize(text2)

    text1_x = img1.width // 2 - text_width1 // 2
    text2_x = img1.width + (img2.width // 2 - text_width2 // 2)

    draw.text((text1_x, text_y), text1, fill="black", font=font)
    draw.text((text2_x, text_y), text2, fill="black", font=font)

    # === Save result ===
    combined.save(f"/home/bahaduri/RL4VLM/outputs/captioned/{num}.png")

def save_images(img1, img2, name1, name2):
    global DATA_PATH
    path1 = os.path.join(DATA_PATH, f"{name1}.png")
    path2 = os.path.join(DATA_PATH, f"{name2}.png")
    
    img1.save(path1)
    img2.save(path2)
    return f"{name1}.png", f"{name2}.png"



def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #storing trial trajs
    trajNum= 0
    # output_path = "/home/bahaduri/RL4VLM/outputs"
    # folder_path = output_path + f"/{trajNum}"
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)

    ##accelerator = accelerate.Accelerator()#gradient_accumulation_steps=args.grad_accum_steps)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ## environment interaction device is cpu
    model_device = device

    #initialization of llava
    model_path = args.model_path
    cache_dir = args.cache_dir

    processor = AutoProcessor.from_pretrained(model_path)
    print(model_path)
    if "Qwen2.5" in model_path:
        Qwen_model = Qwen2_5_VLForConditionalGeneration
    else:
        Qwen_model = Qwen2VLForConditionalGeneration
    #load_pretrained_model(model_path, model_path, model_path)
    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
        if args.q8 or args.q4:
            raise ValueError("Lora model does not support 8bit or 4bit quantization")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if args.q8:
            print("8bit quantization")
            base = Qwen_model.from_pretrained(model_path, load_in_8bit=True)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                    )
            print("4bit quantization")
            base = Qwen_model.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config)
            # if 'mistral' in model_path.lower():
            #     base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
            # else:
            #     base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            base = Qwen_model.from_pretrained(model_path, torch_dtype=torch.float16, device_map="balanced")# low_cpu_mem_usage=True)
            # if 'mistral' in model_path.lower():
            #     base =  LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
            # else:
            #     base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    base.eval()
    use_grad_ckpt = True
    if use_grad_ckpt:
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    base.config.max_length = 1024
    print("Model max context length:{}".format(base.config.max_length))
    # base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter = args.pretrain_mm_adapter)
    # image_processor = base.get_vision_tower().image_processor

    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "v_proj"],#find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config, adapter_name="policy")
        if args.temp_predictor:
            from a2c_ppo_acktr.temp_predictor import Temp_predictor
            base.add_adapter(adapter_name="adversery", peft_config=base_lora_config)
        base.set_adapter("policy")

    if "Qwen2.5" in model_path:
        hidden_dim = 2048
    else:
        hidden_dim = 1536
    value_model = QwenVLMValue(base, processor, hidden_dim, grpo = args.grpo)
    ###
    #for loading lora weights for testing purposes
    
    # lora_weights = torch.load("/home/bahaduri/RL4VLM/outputs/dk_VLM_eps_1_grpo_q25_3/model_6.checkpoint", map_location='cpu')
    # lora_weights = {k.replace("value_model.", "", 1): v for k, v in lora_weights.items() if k.startswith("value_model.")}
    
    # missing_keys, unexpected_keys = value_model.load_state_dict(lora_weights, strict=False)
    # print("**********", len(unexpected_keys))
    
    ###
    # breakpoint()
    # value_model = value_model.to(model_device)

    if "gym_cards" or "minigrid" in args.env_name.lower():
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.gamma, None, device, False, 1)
    else:
        print("Environment not supported")
        exit(1)

    # if True:
    #     temporal_predictor_model = QwenTempPredictor(processor, base)
    #     temporal_predictor = Temp_predictor(temporal_predictor_model, processor)
    obs = envs.reset()
    
    infos = None
    ## Inputing Prompt here
    use_epsilon = args.action_sampling
    qs = get_prompt(args.env_name, args.action_only_prompt, args.action_sampling, infos) #prompt for action sampling and not for optimization.
    #qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)

    INPUT_IDS = qs#tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    #INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace
    INPUT_IDS_PO = get_prompt(args.env_name, args.action_only_prompt) #original prompt to optimize for. it asks directly for action.
    projection_f = partial(text_projection_pr, env_name=args.env_name)  #text_projection

    actor_critic = QwenVLMPolicy(
                             processor=processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS_PO,
                             args=args)
    
    
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    #AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    print("here")
    #actor_critic = accelerator.prepare(actor_critic)
    print("here 2")
    
    
    
    image_tensor = obs.squeeze(0).permute(2,0,1).float()
    if image_tensor.max().item() <= 1.0:
        image_tensor = (image_tensor * 255).byte()
    to_pil = T.ToPILImage()
    image = to_pil(image_tensor)
    
    #image.save('/home/bahaduri/RL4VLM/outputs/00.png')
    #image.save(folder_path + "/00.png")
    
    print("here 5")
    #_, output_ids, action, random_mask, command, action_log_prob, action_tokens_log_prob = actor_critic.act_batch(image, INPUT_IDS)
    _, output_ids, action, random_mask, command, action_log_prob, action_tokens_log_prob = actor_critic.act(image, text = INPUT_IDS)
    
    data = []
    global DATA_PATH

    root_path=  "/home/bahaduri/RL4VLM/outputs/trajs_col"
    subfolders = [f.name for f in os.scandir(root_path) if f.is_dir()]
    subfolders = subfolders[:41]
    
    for subfolder in tqdm(subfolders):
        full_path = os.path.join(root_path, subfolder)
        action_file_path = os.path.join(full_path, "actions")
        
        with open(action_file_path, "r") as file:
            lines = file.readlines()

        # Optional: Strip newline characters
        lines = [line.strip() for line in lines]
        actions_subset = lines
        result_file = "/home/bahaduri/RL4VLM/outputs/captioned/results"
        result_file = Path(result_file)

        #getting the smallest int.png in the subfolder as image counter
        
        all_files = os.listdir(full_path)
        int_png_files = [f for f in all_files if f.endswith('.png') and f[:-4].isdigit()]
        nums = [int(f[:-4]) for f in int_png_files]
        img_counter = min(nums)
        
        for i, action in tqdm(enumerate(actions_subset)):
        # image_path = "path/to/your/image.jpg"
            #action = "Turn Left"
            """
            qs = "You are an expert 2D game player in a grid-based environment. The environment has a key that to pick up in order to unlock a door. The door is represented as blue square with a minus, and then reach the green goal square. "
            qs += "The goal is to get the player to a pink goal tile. The player is shown by triangle."
            qs += f"The first image shows the previous state and player has taken action {action}, and the second image shows the resulting state."
            qs += "Evaluate the palyer's action based on the images and assign a score between 0 (bad) and 5 (good)."
            qs = qs + "Your response should be a valid json object in the following format: \{\n"
                    #qs = qs + "\"action\": \"your choosen action\" "
            qs = qs + "\"thoughts\": \"Describe the images and how the action affect the state.\", \n"  ##Describe the current scene step-by-step and explain the visible area, position and facing direction of the player, nearby objects, or interactive elements with their relative locations.
            #qs = qs + "\n}"
            #qs = qs + "\"top_action_analysis\": \"Based on the above scene, list some plausible next actions the player might take, along with reasoning for each. Do not choose one yet."
            qs = qs + "\"score\": \"your score\" \n}"
            """
            qs = (
                "You're an expert in 2D grid-based games guiding a player by scoring his each step based. "
                "The player is shown by cyan triangle.The tip (pointy end) of the triangle is the direction the player is facing, the flat side is the back. In the game the player must pick up a key to unlock the door. The square with a minus (-) and blue or yellow color is the closed door. The player shuold reach the pink goal tile to win. "
                "At each step the player takes one of these actions ['Turn left': turns direction to left, 'Turn right': turns direction to right, 'Move forward': take one step to front, 'Pick up': picks key only if key was in front of it in first image, 'Toggle': toggle door only if door was infront of it in first image]. "
                "You are given two images of the game environment. "
                f"The first image shows the state *before* taking action: {action}. "
                "The second image shows the state *after* that action. "
                "Analyze what has changed in the second image compared to the first. Did the action have any useful effect? "
                "Rate the usefulness of the action from 0 (useless) to 10 (very useful), based only on the visual evidence in the images.\n "
                "**Scoring rubric (0 – 10):** \n"
                "- **10:** Successful interaction or completion (valid Pick up, valid Toggle that opens needed door, or reaching goal with prerequisites met). \n"
                "- **8 – 9:** Clear improvement: moved closer to the current objective by 1+ tile **or** turned to face it directly (from misaligned to aligned). \n"
                "- **5 – 7:** Minor but real progress: slight angle improvement or small distance improvement that sets up a good next step. \n"
                "- **1 – 2:** Neutral/ineffective: no distance/angle improvement; sideways movement; turn that keeps facing irrelevant space. \n"
                "- **0:** Counterproductive: increased distance, turned away from the objective, moved toward a door while a key exists elsewhere, or attempted invalid Pick up/Toggle. \n"
                "Respond strictly with a JSON object:\n"
                "{\n"
                "\"thoughts\": \"Describe ONLY the changes between before and after images.\",\n"
                "\"score\": your_score\n"
                "}"
            )
            
            """
            qs = (
                "You're an expert in 2D grid-based games guiding a player by scoring his each step based. The player is new to the game and doesn't know how to play taking most of actions randomly. "
                "The player (triangle) must pick up the key to unlock the blue door (with a minus) and reach the pink goal tile. "
                "At each step the player takes one of these actions ['Turn left': turns direction to left, 'Turn right': turns direction to right, 'Move forward': take one step to front, 'Pick up': picks key only if key was in front of it in first image, 'Toggle': toggle door only if door was infront of it in first image] "
                "You are given two images of a 2d grid-based game environment. "
                f"The first image is the state before the action '{action}'. "
                "The second image is the result after this action. "
                "Analyze what has changed in the second image compared to the first. Did the action have any useful effect? "
                "Rate the usefulness of the action from 0 (useless) to 5 (very useful), based **only on the visual evidence in the images**. "
                "Respond with a JSON object:\n"
                "{\n"
                "\"thoughts\": \"Describe ONLY what you see changed between the images.\",\n"
                "\"score\": \"your_score\"\n"
                "}"
            )
            actions = ['Turn left', 'Turn right', 'Move forward', 'Pick up', 'Toggle']
            action = random.choice(actions)

            qs = (
                "You are an expert 2D game player in a grid-based environment. The environment has a key that to pick up in order to unlock a door. The door is represented as blue square with a minus, and then reach the green goal square. "
                #qs = "Rules: you may need to pick up a key to open a locked door, you can only interact with adjacent tiles in the direction you are facing, you can only pass through open doors."
                "You are observing the image of the current state, and your goal is to get the player to a green goal tile. The player is shown by a triangle."
                
                "At each step you have chosen an action from these actions ['Turn left', 'Turn right', 'Move forward', 'Pick up', 'Toggle']"
                f"your chosen action is: {action} "
                #"Respect the preselected action. "
                "Your response should be a valid json object in the following format: \{\n"
                #qs = qs + "\"action\": \"your choosen action\" "
                #"\"thoughts\": \"Describe the current scene and state of the agent as seen in the image.\", \n"  ##Describe the current scene step-by-step and explain the visible area, position and facing direction of the player, nearby objects, or interactive elements with their relative locations.
                #qs = qs + "\n}"
                #qs = qs + "\"top_action_analysis\": \"Based on the above scene, list some plausible next actions the player might take, along with reasoning for each. Do not choose one yet."
                f'\n  "action": "{action}"\n'
                "} "
                #f'\"action\": \"{action}\" \n}'
            )
            print(qs)
            print(action)
            """
            #image = Image.open(image_path).convert("RGB")  # Convert to RGB to avoid issues with grayscale or RGBA
            transform = T.ToTensor()  # Converts [0, 255] PIL image to [0.0, 1.0] tensor
                
            # Step 3: Apply the transformation
            image_tensor = transform(image)
            images = []
            
            img1 = Image.open(root_path + f"/{subfolder}" + f"/{img_counter}.png")
            image1 = transform(img1).permute(1,2, 0).unsqueeze(0)
            images.append(image1)
            try:
                img2 = Image.open(root_path + f"/{subfolder}" + f"/{img_counter+1}.png")
                # Step 2: Define transformation to convert image to tensor
                image2 = transform(img2).permute(1,2, 0).unsqueeze(0)
                images.append(image2)
            except:
                continue
            
            #breakpoint()
            output, out = actor_critic.calc_utility(images, INPUT_IDS = qs)
            print(i, action, ": ", out)
            path1, path2 = save_images(img1, img2, img_counter, img_counter+1)

            img_counter += 1
            data_point = {
                "action": action,
                "before_img": path1,
                "after_img": path2,
                "response": out,
                "utility_score": output,
            }
            data.append(data_point)
            print( i, " Done!!")
            continue
            save_captioned_images(img1,  action, output, i, img2)
            if not result_file.exists():
                result_file.touch()
            with open(result_file, 'a') as f:
                f.write(str(i) + ":  " + out)
        #breakpoint()
        print(type(image_tensor))  # <class 'torch.Tensor'>
        print(image_tensor.shape)
    
        json_fileName = DATA_PATH + "/labels.json"
        with open(json_fileName, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data has been successfully dumped to {json_fileName}")
    

if __name__ == "__main__":
    main()

