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

from a2c_ppo_acktr import algo, utils, rl_utils

from a2c_ppo_acktr.utils import CustomWandbTracker, log_metrics
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection, text_projection_pr
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue, QwenVLMValue, QwenVLMPolicy, QwenTempPredictor
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
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
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
import transformers

from tqdm import tqdm
import wandb
import accelerate
from accelerate.state import AcceleratorState

from PIL import Image
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

def save_image_action(traj, step, image, action=None):
    trajNum= traj
    output_path = "/home/bahaduri/RL4VLM/outputs/trajs"
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
    

def main():
    args = get_args()
    

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

    
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device

    if args.use_wandb and accelerator.is_main_process: #TODO wandb
        
        print("main proc")
        run_name = args.wandb_run + "-" + args.env_name
        wandb.init(project=args.wandb_project, name=run_name, group=args.wandb_group, job_type=str(args.seed), config=args) #TODO group, job_type
        
    ## environment interaction device is cpu
    model_device = device
    seed = args.seed + accelerator.process_index
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
            base = Qwen_model.from_pretrained(model_path, torch_dtype=torch.float16)
            # if 'mistral' in model_path.lower():
            #     base =  LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
            # else:
            #     base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    
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
    #breakpoint()
    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "v_proj", "k_proj", "mlp.0", "mlp.2"],#find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config, adapter_name="policy")
        if args.utility_func or True:
            #from a2c_ppo_acktr.temp_predictor import Temp_predictor
            base.add_adapter(adapter_name="adversery", peft_config=base_lora_config)
            base.load_adapter("/home/bahaduri/RL4VLM/outputs/sft/value_lora/policy", "adversery")
            base.set_adapter("adversery")
            for n, p in base.named_parameters():
                if "adversery" in n:
                    p.requires_grad = False
        base.set_adapter("policy")
        base.load_adapter("/home/bahaduri/RL4VLM/outputs/sft/value_lora/policy", "policy")
        for n, p in base.named_parameters():
            if "policy" in n:
                p.requires_grad = True
    #breakpoint()
    if "Qwen2.5" in model_path:
        hidden_dim = 2048
    else:
        hidden_dim = 1536
    if args.dense_rewards:
        no_value = False
    elif args.utility_func:
        no_value = True
    else:
        no_value = False
    value_model = QwenVLMValue(base, processor, hidden_dim, grpo = no_value) #args.grpo) if grpo then there will be no value head
    ###
    #for loading lora weights for testing purposes
    """
    # pretrained = torch.load("/home/bahaduri/RL4VLM/outputs/sft/value_lora/policy/adapter_model.bin", map_location="cpu")

    # # sum of pretrained weights
    # pretrained_sum = sum(v.abs().sum().item() for v in pretrained.values())
    # print("Pretrained LoRA weight sum:", pretrained_sum)
    lora_weights = torch.load("/home/bahaduri/RL4VLM/outputs/dk_VLM_eps_1_notht_util/model_10.checkpoint", map_location='cpu')
    lora_weights = {k.replace("value_model.", "", 1): v for k, v in lora_weights.items() if k.startswith("value_model.")}
    print("LoRA weight sum:", sum(v.abs().sum().item() for v in lora_weights.values()))
    missing_keys, unexpected_keys = value_model.load_state_dict(lora_weights, strict=False)
    print("**********", len(unexpected_keys))
    """
    ###
    
    value_model = value_model.to(model_device)

    if "gym_cards" or "minigrid" in args.env_name.lower():
        envs = make_vec_envs(args.env_name, seed, args.num_processes,
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
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in actor_critic.value_model.named_parameters() if ("lora_A.adversery" in n or "lora_B.adversery" in n)], 'lr': 3e-4,
        },
        {
          'params': [p for n, p in actor_critic.value_model.named_parameters() if ("lora_A.adversery" not in n and "lora_B.adversery" not in n)], 'weight_decay': args.weight_decay, 'lr': args.init_lr, 'eps':args.eps,
          }
    ]
    
    optimizer = optim.Adam([p for p in actor_critic.value_model.parameters() if p.requires_grad], lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay) #optimizer_grouped_parameters)#

    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda step: cosine_schedule(step, args.lr_max_steps, args.end_lr, args.init_lr),
            #lambda step: 1.0,
        ]
    )
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    
    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)
    #actor_critic, lr_scheduler = accelerator.prepare(actor_critic, lr_scheduler)
    actor_critic.base.set_adapter("policy")
    if args.temp_predictor:
        from a2c_ppo_acktr.temp_predictor import Temp_predictor
        temporal_predictor = Temp_predictor(actor_critic.value_model, processor, optimizer, accelerator)
    else:
        temporal_predictor = None
    # actor_critic.base.set_adapter('policy')
    # actor_critic.value_model.base.set_adapter('policy')
    
    if args.grpo:
        agent = algo.GRPO(
                actor_critic,
                optimizer,
                accelerator,
                args.clip_param,
                args.ppo_epoch,
                args.mini_batch_size,
                args.value_loss_coef,
                args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                save_dir=args.save_dir,
                obs_shape=envs.observation_space.shape,
                max_new_tokens = args.max_new_tokens,
                number_steps = args.num_steps)
    else:
        agent = algo.PPO(
                actor_critic,
                optimizer,
                accelerator,
                args.clip_param,
                args.ppo_epoch,
                args.mini_batch_size,
                args.value_loss_coef,
                args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                save_dir=args.save_dir,
                grad_cumulate_step = args.grad_accum_steps,
                utility_function = args.utility_func)
    
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, args.max_new_tokens, temporal_predictor, args.act_freq_reward, scale=0.00025, grpo=False, utility_function= args.utility_func, dense_rewards = args.dense_rewards)#args.grpo)
    
    image_tensor = obs.squeeze(0).permute(2,0,1).float()
    if image_tensor.max().item() <= 1.0:
        image_tensor = (image_tensor * 255).byte()
    to_pil = T.ToPILImage()
    image = to_pil(image_tensor)
    
    #image.save('/home/bahaduri/RL4VLM/outputs/00.png')
    #image.save(folder_path + "/00.png")
    
    #_, output_ids, action, random_mask, command, action_log_prob, action_tokens_log_prob = actor_critic.act_batch(image, INPUT_IDS)
    _, output_ids, action, random_mask, command, action_log_prob, action_tokens_log_prob = actor_critic.act(image, text = INPUT_IDS)
    print("action:{}".format(action))
    print("action_log_prob:{}".format(action_log_prob))
    print("action_tokens_log_prob:{}".format(action_tokens_log_prob))

    rollouts.obs[0].copy_(obs)
    #rollouts.to(device)

    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    
    
    print(qs)
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    epsilon_start = 1.0
    epsilon_min = 0.0
    epsilon_decay = 0.995
    num_explore = int(args.explore_portion*num_updates)
    prev_infos = []
    infos = []
    
    for j in tqdm(range(num_updates)):
        n_start = False
        if use_epsilon:
            if j > 1:
                step_2 = (args.num_steps * j -1) + args.num_steps
            else:
                step_2 = 0
            epsilon = max(epsilon_min, epsilon_start - (step_2/(args.num_env_steps - 8000)) * (epsilon_start - epsilon_min))
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                INPUT_IDS = qs#tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                # INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace

                image_tensor = rollouts.obs[step].squeeze(0).permute(2,0,1).float() #TODO rollouts.obs[step] needs to be checked if expected shape
                if image_tensor.max() <= 1.0:
                    image_tensor = (image_tensor * 255).byte()
                to_pil = T.ToPILImage()
                image = to_pil(image_tensor)
                

                # Save the plot
                #image.save(f'/home/bahaduri/RL4VLM/outputs/00.png')
                #print("Active adapter", actor_critic.base.active_adapter)
                value, output_id, action, random_mask, command, action_log_prob, action_tokens_log_prob = actor_critic.act(
                        image, text = INPUT_IDS)
                #save_image_action(j, step, image, command)
            text_action = processor.decode(list(filter(lambda num: num != 151643, output_id[0].tolist()))) #151643 is the pad_token for the qwen model #TODO hardcoded
            prev_infos = copy.deepcopy(infos)
            
            obs, reward, done, infos = envs.step(action)
            #reward = reward + random_mask * (-0.5)
            # if step % 4 == 0:
            #     #if random.random() < 0.5:
            #     reward += 0.21  #TODO for testing
            #epsilon greedy
            current_action_sampling = args.action_sampling
            if use_epsilon:
                # if j > 1:
                #     step_2 = (step * j -1) + step
                # else:
                #     step_2 = 0
                # epsilon = max(epsilon_min, epsilon_start - (step_2/(args.num_env_steps - 8000)) * (epsilon_start - epsilon_min))
                if random.random() < epsilon:
                    args.action_sampling = True
                else:
                    args.action_sampling = False
            #
            
            qs = get_prompt(args.env_name, args.action_only_prompt, args.action_sampling, infos)  #this prompt is for sampling
            #qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            # masks = torch.FloatTensor([
            #     [0.0] if done_ or (step % 4 == 0) else [1.0] 
            #     for done_ in done
            # ])
            tasks = [None] * args.num_processes               # tasks is used for storing the trajectory 
                                                              # and passing it to buffer. if trajectory starts newly
            status = [None] * args.num_processes                                                   # it will be set to qs otherwise None.
            running_episode_rewards += reward.flatten()
            success = False
            fail = False
            for i, d, r, info in zip(range(args.num_processes), done, reward, infos):
                if n_start or step == 0:
                    tasks[i] = qs
                if d:# or step % 4 ==0:  ##TODO for testing
                    trajNum +=1
                    episode_rewards.append(running_episode_rewards[i].item())
                    if running_episode_rewards[i] > 0:
                        episode_success_rate.append(1)
                        status[i] = 1
                        success = True
                    else:
                        episode_success_rate.append(0)
                        fail = True
                        status[i] = 0
                    episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                    running_episode_rewards[i] = 0
                    #tasks[i] = qs
                    n_start = True
                    
                else:
                    n_start = False
                    if step == args.num_steps -1:
                        status[i] = 0

            # bad_mask is a legacy implementation of the storage.py file
            bad_masks = torch.FloatTensor( #TODO bad_transition to be checked
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert_task(tasks, command, status)  #TODO check command is a nice list
            rollouts.insert(obs, output_id, action,
                            action_log_prob, value, reward, masks, bad_masks, random_mask, torch.tensor([[current_action_sampling]]).float(), fail= fail, success = success)
            success = False
            fail = False
            #print("step: ", step)
        print("****** iteration number:{} ******".format(j))
        if use_epsilon:
            print("****Epsilon:", epsilon)
        print("prompt:{}".format(prompt))
        print("text_action:{}".format(text_action))
        #print("current observation:{}".format(prev_infos))
        print("ground truth:{}".format(infos))
        print("action log prob:{}".format(action_log_prob))
        print("action tokens log prob:{}".format(action_tokens_log_prob))
        with torch.no_grad():
            image_tensor = rollouts.obs[-1].squeeze(0).permute(2,0,1).float()
            if image_tensor.max().item() <= 1.0:
                image_tensor = (image_tensor * 255).byte()
            to_pil = T.ToPILImage()
            image = to_pil(image_tensor)
            next_value = actor_critic.get_value(
                image).detach()

        if  args.temp_predictor and j >0: #TODO add check wether temp predictor is used at all
            tmp_info = temporal_predictor.update_mm_model()
        else:
            tmp_info = {"temp_predictor.loss": 0, "temp_predictor.acc": 0}

        if args.dense_rewards:
            agent.densify_rewards(rollouts)
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits, j)
        if args.grpo:
            #if rollouts.prev_suc_step > 0 or (agent._buffer is not None and not agent._buffer.zero):
                print("success steps:", rollouts.prev_suc_step)
                value_loss, action_loss, dist_entropy, grpo_reward, grpo_total_correct = agent.update(rollouts, update_num=j)
                update = True
            #else:
            #    value_loss, action_loss, dist_entropy, grpo_reward = 0, 0, 0, 0
        if not args.grpo:
            if args.rlef:
                value_loss, action_loss, dist_entropy = agent.update_RLEF(rollouts, update_num=j)
            else:
                value_loss, action_loss, dist_entropy = agent.update(rollouts, update_num=j)
            grpo_reward = 0
            grpo_total_correct =0
        lr_scheduler.step()
        

        rollouts.after_update()
        # if args.grpo:
        #     obs = envs.reset()
        #     rollouts.obs[0].copy_(obs)
        
                #saving model and optimizer:
        """
        if j % args.save_period == 0:
            print("Saving model ...")
            model_state_dict = OrderedDict(
                {k: v for k, v in get_trainable_params(value_model, True)}
            )
            torch.save(model_state_dict, args.save_dir + f"/model_{j}.checkpoint")
            torch.save(
                optimizer.state_dict(), args.save_dir + f"/optimizer_{j}.checkpoint"
            )
        model_state_dict = OrderedDict(
                {k: v for k, v in get_trainable_params(value_model, True)}
            )
        torch.save(model_state_dict, args.save_dir + "/model_last.checkpoint")
        torch.save(
                optimizer.state_dict(), args.save_dir + "/optimizer_last.checkpoint"
            )
        """

        if len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_success_rate),
                        dist_entropy, value_loss, action_loss))
            #breakpoint()
            if args.use_wandb:
                metrics = {"iteration": j,
                        "num_timesteps": total_num_steps,
                        "FPS": int(total_num_steps / (end - start)),
                        "episode_reward.mean": torch.tensor([np.mean(episode_rewards)]).to(device),
                        "episode_reward.median": torch.tensor([np.median(episode_rewards)]).to(device),
                        "episode_reward.min": torch.tensor([np.min(episode_rewards)]).to(device),
                        "episode_reward.max": torch.tensor([np.max(episode_rewards)]).to(device),
                        "episode_success_rate.mean": torch.tensor([np.mean(episode_success_rate)]).to(device),
                        "episode_action_tokens_log_prob.mean": torch.tensor([np.mean(episode_action_tokens_log_prob)]).to(device),
                        "distribution_entropy": dist_entropy,
                        "value.loss": torch.tensor([value_loss]).to(device),
                        "action.loss": torch.tensor([action_loss]).to(device),
                        "grpo_reward.mean": grpo_reward,
                        "grpo_corrects.total": grpo_total_correct,
                        "reward.max": rollouts.rewards.max().to(device), #.item(),
                        "reward.min": rollouts.rewards.min().to(device), #.item(),
                        "reward.mean": rollouts.rewards.mean().to(device), #.item(),
                        "reward.std": rollouts.rewards.std().to(device), #.item(),
                        "reward.median": rollouts.rewards.median().to(device), #.item(),
                        "temp_rewards.max": rollouts.temp_rewards.max().to(device), #.item(),
                        "temp_rewards.min": rollouts.temp_rewards.min().to(device), #.item(),
                        "temp_rewards.mean": rollouts.temp_rewards.mean().to(device), #.item(),
                        "return.max": rollouts.returns.max().to(device), #.item(),
                        "return.min": rollouts.returns.min().to(device), #.item(),
                        "return.mean": rollouts.returns.mean().to(device), #.item(),
                        "return.std": rollouts.returns.std().to(device), #.item(),
                        "value.max": rollouts.value_preds.max().to(device), #.item(),
                        "value.min": rollouts.value_preds.min().to(device), #.item(),
                        "value.mean": rollouts.value_preds.mean().to(device), #.item(),
                        "value.std": rollouts.value_preds.std().to(device), #.item(),
                        **tmp_info
                        }
                #if accelerator.is_main_process:
                log_metrics(metrics, step=total_num_steps, accelerator=accelerator, logger_fn=wandb.log)

if __name__ == "__main__":
    main()

