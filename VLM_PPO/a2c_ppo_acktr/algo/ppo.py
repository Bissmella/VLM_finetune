import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import accelerate
from collections import OrderedDict
from .reward_funcs import *
from a2c_ppo_acktr.storage import GRPO_buffer
from a2c_ppo_acktr.llava_interface import qwen_process, qwen_batch_process, format_data_sft, qwen_process_multiImg
import re
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor
import os
from typing import Dict, Optional, Sequence, List


def reward_func(completions_in, solutions, raw_rewards, coef=0.1, groups = 4):
    actions_list = ["Turn left", "Turn right", "Move forward", "Pick up", "Unused",  "Toggle", "Unused"]
    
    solutions = [actions_list[sol] for sol in solutions for _ in range(groups)]
    completions = []
    for completion in completions_in:
        match = re.search(r"```(?:json)?\n(.*?)\n```", completion, re.DOTALL)
        if match:
            string = match.group(1)
        else:
            string = completion.strip()
        completions.append(string)
    format_r = format_reward(completions, ["Turn left", "Turn right", "Move forward", "Pick up","Toggle"])#TODO complete possible actions
    acc_r, ncorrects = accuracy_reward(completions=completions, solutions= solutions)
    acc_r = acc_r.unsqueeze(1)
    format_r = format_r.unsqueeze(1)
    raw_rewards = raw_rewards.repeat(1, groups)
    raw_rewards = raw_rewards.view(-1, 1)  #shape [n, 1] #n be number of questions
    #correct_1 = raw_rewards.max()
    #reward = torch.minimum(torch.tensor([1- coef], device=raw_rewards.device),  raw_rewards + coef) * acc_r.to(raw_rewards.device)  + coef *format_r.to(raw_rewards.device)
    reward = raw_rewards * acc_r.to(raw_rewards.device)
    
    #return the penalty and acc_r
    return 0.1 * format_r, reward, ncorrects

def densify_rewards_1d(rewards, masks):
    """
    rewards: [N] tensor of rewards (float)
    masks:   [N] tensor of 0/1 (0 = terminal step, 1 = ongoing)
    
    Output: densified rewards [N]
    """
    N = rewards.shape[0]
    densified = torch.zeros_like(rewards)

    # indices where trajectories end
    terminal_idxs = (masks == 0).nonzero(as_tuple=False).squeeze(-1)

    start = 0
    for end in terminal_idxs.tolist():
        traj_rewards = rewards[start:end+1]
        last_reward = traj_rewards[-1].item()

        if last_reward > 0:  
            # success trajectory → copy last reward backward
            densified[start:end+1] = 1
        # else:
        #     # failed trajectory → last = -1, others = 0.1
        #     densified[start:end] = 0.08
        #     densified[end] = 0.08

        start = end + 1

    return densified


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 processor,
                 actions, obs, masks):
        super(LazySupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.processor = processor
        self.actions = actions
        self.obs = obs
        self.masks = masks
        self.query = (
                "You're an expert in 2D grid-based games guiding a player by scoring his each step based. "
                "The player is shown by cyan triangle.The tip (pointy end) of the triangle is the direction the player is facing, the flat side is the back. In the game the player must pick up a key to unlock the door. The square with a minus (-) and blue or yellow color is the closed door. The player shuold reach the pink goal tile to win. "
                "At each step the player takes one of these actions ['Turn left': turns direction to left, 'Turn right': turns direction to right, 'Move forward': take one step to front, 'Pick up': picks key only if key was in front of it in first image, 'Toggle': toggle door only if door was infront of it in first image]. "
                "You are given two images of the game environment. "
                "The first image shows the state *before* taking action: {action}. "
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
                "{{\n"
                "\"thoughts\": \"Describe ONLY the changes between before and after images.\",\n"
                "\"score\": your_score\n"
                "}}"
            ) #TODO the main query where the {action} will be put in
        self.actions_list = ["Turn left", "Turn right", "Move forward", "Pick up", "Unused",  "Toggle", "Unused"]
        self.transform = T.ToTensor() 

    def __len__(self):
        return len(self.actions)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        mask = self.masks[i+1]
        obs1 = self.obs[i]
        obs2 = self.obs[i+1]
        act = self.actions[i]
        if mask == 0.0:
            obs2 = None

        action = self.actions_list[act]
        query= self.query.format(action = action)
        
        
        data_dict = {}
        data_dict["image"] = [obs1, obs2]
        data_dict["query"] = query
        formatted_data_dict = format_data_sft(data_dict)
        #batch = self.process_mm([formatted_data_dict])
        
        return formatted_data_dict
    
@dataclass
class Collate_fn_qwen(object):

    def __init__(self, processor):
        self.processor = processor


    def __call__(self, examples):
        """
        examples has a length of batch-size that is not fixed and can vary
        each consist of tuple: first is chat, second is random-mask
        """
        # Get the texts and images, and apply the chat template
        
        texts = [
            self.processor.apply_chat_template(example, tokenize=False) for example in examples
        ]  # Prepare texts for processing
        image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = self.processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors
        return batch

class PPO():
    def __init__(self,
                 actor_critic,
                 optimizer,
                 accelerator,
                 clip_param,
                 ppo_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 save_dir = "",
                 save_interval =2,
                 grad_cumulate_step=64,
                 utility_function = False):

        self.actor_critic = actor_critic

        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param

        self.ppo_epoch = ppo_epoch

        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer
        self.accelerator = accelerator

        self.save_dir = save_dir
        self.save_interval = save_interval
        self.gradient_accumulation_steps = grad_cumulate_step

        self.utility_function = utility_function

        self.value_query = (
                "You're an expert in 2D grid-based games guiding a player by scoring his each step based. "
                "The player is shown by cyan triangle.The tip (pointy end) of the triangle is the direction the player is facing, the flat side is the back. In the game the player must pick up a key to unlock the door. The square with a minus (-) and blue or yellow color is the closed door. The player shuold reach the pink goal tile to win. "
                "At each step the player takes one of these actions ['Turn left': turns direction to left, 'Turn right': turns direction to right, 'Move forward': take one step to front, 'Pick up': picks key only if key was in front of it in first image, 'Toggle': toggle door only if door was infront of it in first image]. "
                "You are given two images of the game environment. "
                "The first image shows the state *before* taking action: {action}. "
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
                "{{\n"
                "\"thoughts\": \"Describe ONLY the changes between before and after images.\",\n"
                "\"score\": your_score\n"
                "}}"
            )

    def densify_rewards(self, rollouts):
        masks = rollouts.masks[1:].view(-1)
        rewards = rollouts.rewards.view(-1)
        dense_rewards = densify_rewards_1d(rewards, masks)
        actions_list = ["Turn left", "Turn right", "Move forward", "Pick up", "Unused",  "Toggle", "Unused"]
        values = []
        self.actor_critic.value_model.base.set_adapter("adversery")
        data_generator = rollouts.value_data_generator(mini_batch_size = 6)
        for sample in data_generator:
            actions_batch, obs1_batch, obs2_batch, masks_batch = sample
            texts = []
            images = []
            for i, action in enumerate(actions_batch):
                mask = masks_batch[i]
                obs1 = obs1_batch[i]
                obs2 = obs2_batch[i]
                act = actions_list[action]
                if mask == 0.0:
                    obs2 = obs1
                query = self.value_query.format(action = act)
                texts.append(query)
                images.append([obs1, obs2])
            #print("Active adapter in ppo value", self.actor_critic.base.active_adapter)
            action_values, _ = self.actor_critic.calc_utility_batch(images, texts)
            
            values.extend(action_values)
        values = torch.tensor(values, dtype=torch.float32)
        mask_r = rollouts.rewards.view(-1) > 0
        values[mask_r] = 0
        values = torch.clamp(values, max=10)
        values[values < 3] = 0
        mask = values >= 3
        values[mask] = (values[mask] - 2) / (10 - 2) * 10
        values = values /10.0

        values = values * dense_rewards * 0.04 #alpha =0.04
        num_steps, num_procs, _ = rollouts.rewards.shape
        values = values.reshape(num_steps, num_procs, 1)
        
        rollouts.dense_rewards = rollouts.rewards + values
        self.actor_critic.value_model.base.set_adapter("policy")

    def get_values(self, rollouts):
        actions_list = ["Turn left", "Turn right", "Move forward", "Pick up", "Unused",  "Toggle", "Unused"]
        values = []
        self.actor_critic.value_model.base.set_adapter("adversery")
        data_generator = rollouts.value_data_generator(mini_batch_size = 4)
        total_bad_util =0
        for sample in data_generator:
            actions_batch, obs1_batch, obs2_batch, masks_batch = sample
            texts = []
            images = []
            for i, action in enumerate(actions_batch):
                mask = masks_batch[i]
                obs1 = obs1_batch[i]
                obs2 = obs2_batch[i]
                act = actions_list[action]
                if mask == 0.0:
                    obs2 = obs1
                query = self.value_query.format(action = act)
                texts.append(query)
                images.append([obs1, obs2])
            #print("Active adapter in ppo value", self.actor_critic.base.active_adapter)
            action_values, outputs, bad_util = self.actor_critic.calc_utility_batch(images, texts)
            total_bad_util += bad_util
            
            
            values.extend(action_values)
        values.append(0)
        values = torch.tensor(values, dtype=torch.float32)
        mask_r = rollouts.rewards.view(-1) > 0
        values[:-1][mask_r] = 10
        
        mask = rollouts.masks[1:].view(-1) == 0.0
        avg = torch.mean(values[:-1][~mask])
        mask_bad_util = values == -1
        values[:-1][mask & ~mask_r] = avg  #setting failed endings to average
        values[mask_bad_util] = avg        #setting bad utility estimates (no utility estimate) to average
        values = torch.clamp(values, max=10) 
        values = values /10.0
        
        num_steps, num_procs, _ = rollouts.value_preds.shape
        values = values.reshape(num_steps, num_procs, 1)
        rollouts.value_preds = values
        print("total bad utility: ", total_bad_util)
        self.actor_critic.value_model.base.set_adapter("policy")
        #print("Active adapter in ppo after value", self.actor_critic.base.active_adapter)
        # def hook(module, inp, out):
        #     print("LoRA fired:", module)
        # for n,m in self.actor_critic.named_modules():
        #     if 'policy' in n:
        #         m.register_forward_hook(hook)
        

    def update(self, rollouts, update_num):
        random_mask = rollouts.random_mask == 1
        rollouts.returns[random_mask] = -1
        if self.utility_function:
            self.get_values(rollouts)
            
            rollouts.value_preds[random_mask] = 1
            advantages = rollouts.value_preds[:-1] * rollouts.returns[:-1]
        else:
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        


        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_step = 0
        self.actor_critic.train()
        # adv_quant = torch.quantile(advantages, 0.6)
        # #breakpoint()
        Min_weight = torch.tensor([0.0])
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.mini_batch_size, update_num)
            print("iter..", e)
            for sample in data_generator:
                with self.accelerator.accumulate(self.actor_critic):
                    grad_step += 1
                    obs_batch, output_ids_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ, act_sampling_batch = sample
                    # Reshape to do in a single forward pass for all steps
                    
                    values, action_log_probs = self.actor_critic.evaluate_actions(
                        obs_batch, output_ids_batch)
                    # values and action_log_probs on two different devices!! because they come from two llava
                    if torch.isnan(action_log_probs).any():
                        continue
                    old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                    adv_targ = adv_targ.to(action_log_probs.device)
                    value_preds_batch = value_preds_batch.to(action_log_probs.device)
                    return_batch = return_batch.to(action_log_probs.device)


                    ratio = torch.exp(action_log_probs -
                                    old_action_log_probs_batch)
                    
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    ## ratio clip, inspired by https://github.com/huggingface/trl/blob/5a233546ee48532eaeb24b89b8d0042147574688/trl/trainer/ppo_trainer.py#L1199
                    if torch.any(ratio > 10):
                        ppo_loss = -surr2.mean()
                    else:
                        ppo_loss = -torch.min(surr1, surr2).mean()
                    bc_loss = (- action_log_probs * torch.clamp(adv_targ, min=0.0)).mean()
                    act_sampling_batch = act_sampling_batch.to(action_log_probs.device)
                    action_loss = act_sampling_batch * bc_loss + (1- act_sampling_batch) * ppo_loss
                    # print(action_loss)
                    if not self.utility_function:
                        if self.use_clipped_value_loss:
                            value_pred_clipped = value_preds_batch + \
                                (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - return_batch).pow(2)
                            value_losses_clipped = (
                                value_pred_clipped - return_batch).pow(2)
                            value_loss = 0.5 * torch.max(value_losses,
                                                        value_losses_clipped).mean()
                        else:
                            value_loss = 0.5 * (return_batch - values).pow(2).mean()

                        try:
                            assert not torch.isnan(value_loss), "value_loss is nan"
                            assert not torch.isnan(action_loss), "action_loss is nan"
                        except:
                            print("value/action loss is nan")
                            exit(1)
                        loss = value_loss * self.value_loss_coef+action_loss
                    else:
                        loss = action_loss
                        value_loss = torch.tensor([0])
                    # print("total loss: ", loss)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:

                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    # tmpmodel = self.accelerator.unwrap_model(self.actor_critic)
                    # print("loss", loss)
                    # print("LoRA grad sum:", sum(p.grad.detach().abs().sum().item() for n,p in tmpmodel.named_parameters() if 'lora' in n and p.grad is not None))
                    # print("LoRApoliy grad sum:", sum(p.grad.detach().abs().sum().item() for n,p in tmpmodel.named_parameters() if 'policy' in n and p.grad is not None))
                    # print("LoRAadversery grad sum:", sum(p.grad.detach().abs().sum().item() for n,p in tmpmodel.named_parameters() if 'adversery' in n and p.grad is not None))
                    # print("opt_params:", sum(p.numel() for g in self.optimizer.param_groups for p in g['params']), "model_params:", sum(p.numel() for p in tmpmodel.parameters()))
                    # for i, g in enumerate(self.optimizer.param_groups):
                    #     print(f"Group {i}")
                    #     for p in g['params']:
                    #         for n, q in tmpmodel.named_parameters():
                    #             if q is p:
                    #                 print("   ", n, p.shape, "requires_grad=", p.requires_grad)
                    # for i, g in enumerate(self.optimizer.param_groups):
                    #     print(f"Group {i}, num params = {len(g['params'])}")
                    #     for p in g['params'][:10]:  # just peek first 10
                    #         print("   id:", id(p), "shape:", p.shape, "requires_grad=", p.requires_grad)
                    
                    # for n, p in tmpmodel.named_parameters():
                    #     if "policy" in n:
                    #         if p.grad is not None:
                    #             print(n, "grad sum:", p.grad.abs().sum().item())
                    #         else:
                    #             print(n, "grad is None")
                    # breakpoint()
                    self.optimizer.zero_grad()

                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()

        value_loss_epoch /= grad_step
        action_loss_epoch /= grad_step
        dist_entropy_epoch /= grad_step
        
        if self.save_dir != "":
            if update_num % self.save_interval == 0:
                self.save_checkpoint(update_num)
            self.save_checkpoint("last")
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch#, temp_pred_rewards
    

    def update_RLEF(self, rollouts, update_num):
        if self.utility_function:
            self.get_values(rollouts)
            advantages = rollouts.value_preds[:-1] * rollouts.returns[:-1]
        else:
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        
        # advantages = (advantages - advantages.mean()) / (
        #     advantages.std() + 1e-5)


        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_step = 0
        self.actor_critic.train()
        Max_weight = torch.tensor([10])
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.mini_batch_size, update_num)
            for i, sample in enumerate(data_generator):
                with self.accelerator.accumulate(self.actor_critic):
                    grad_step += 1
                    obs_batch, output_ids_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample
                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs = self.actor_critic.evaluate_actions_batch(
                        obs_batch, output_ids_batch)
                    # values and action_log_probs on two different devices!! because they come from two llava
                    if torch.isnan(action_log_probs).any():
                        continue
                    old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                    adv_targ = adv_targ.to(action_log_probs.device)
                    value_preds_batch = value_preds_batch.to(values.device)
                    return_batch = return_batch.to(values.device)


                    # ratio = torch.exp(action_log_probs -
                    #                 old_action_log_probs_batch)
                    
                    ## ratio clip, inspired by https://github.com/huggingface/trl/blob/5a233546ee48532eaeb24b89b8d0042147574688/trl/trainer/ppo_trainer.py#L1199
                    #mask = (adv_targ > 0.004).float().squeeze(1)
                    weighted_advantage = torch.exp(adv_targ)
                    weighted_advantage = torch.minimum(weighted_advantage, Max_weight.to(weighted_advantage.device))
                    action_loss = (-action_log_probs * weighted_advantage).mean()
                    
                    if not self.utility_function:
                        if self.use_clipped_value_loss:
                            value_pred_clipped = value_preds_batch + \
                                (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - return_batch).pow(2)
                            value_losses_clipped = (
                                value_pred_clipped - return_batch).pow(2)
                            value_loss = 0.5 * torch.max(value_losses,
                                                        value_losses_clipped).mean()
                        else:
                            value_loss = 0.5 * (return_batch - values).pow(2).mean()

                        try:
                            assert not torch.isnan(value_loss), "value_loss is nan"
                            assert not torch.isnan(action_loss), "action_loss is nan"
                        except:
                            print("value/action loss is nan")
                            exit(1)
                        loss = value_loss * self.value_loss_coef+action_loss
                        loss = loss/ self.gradient_accumulation_steps
                    else:
                        loss = action_loss
                        value_loss = torch.tensor([0])
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:

                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    """
                    if self.accelerator.sync_gradients:

                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    """

                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()

        value_loss_epoch /= grad_step
        action_loss_epoch /= grad_step
        dist_entropy_epoch /= grad_step
        
        if self.save_dir != "":
            if update_num % self.save_interval == 0:
                self.save_checkpoint(update_num)
            self.save_checkpoint("last")
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    # def get_trainable_params(self, return_with_names=True):
    #     unwrapped_model = self.accelerator.unwrap_model(self.actor_critic)
    #     breakpoint()
    #     return filter(lambda p: p[1].requires_grad, unwrapped_model.named_parameters())
    
    def get_trainable_params(self):
    # Returns a dict of only trainable params
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.actor_critic)
        return {
            name: param.detach().cpu()
            for name, param in unwrapped_model.named_parameters()
            if param.requires_grad
        }
    
    def save_checkpoint(self,update_num):
        model_state_dict = self.get_trainable_params()
        # model_state_dict = OrderedDict(
        #     {k: v for k, v in self.get_trainable_params()}
        # )
        # self.accelerator.wait_for_everyone()
        # unwrapped_model = self.accelerator.unwrap_model(self.actor_critic)
        # breakpoint()
        # self.accelerator.save(unwrapped_model, self.save_dir + f"/model_{update_num}.checkpoint")
        #breakpoint()
        torch.save(model_state_dict, self.save_dir + f"/model_{update_num}.checkpoint")
        torch.save(self.optimizer.state_dict(), self.save_dir + f"/optimizer_{update_num}.checkpoint")
        print("model saved")




class GRPO():
    def __init__(self,
                 actor_critic,
                 optimizer,
                 accelerator,
                 clip_param,
                 ppo_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 save_dir = "",
                 save_interval =2,
                 obs_shape=None,
                 max_new_tokens=None,
                 groupSize = 2,
                 number_steps = 512):
        print("group size", groupSize)
        self.actor_critic = actor_critic
        self.obs_shape = obs_shape
        self.max_new_tokens = max_new_tokens
        self.mini_batch_size = mini_batch_size
        self.groupSize = groupSize
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param
        self.number_steps = number_steps

        self.ppo_epoch = ppo_epoch

        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer
        self.accelerator = accelerator

        self.save_dir = save_dir
        self.save_interval = save_interval
        self._buffer = None
        self.penalty_coef = 0.03


    def generate_inputs(self, rollouts, ):
        update_num = 0
        pos_steps = rollouts.returns.shape[0] - 1
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]  #this is not needed, just as placeholder for current code
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        num_steps = self.number_steps#advantages.shape[0]
        if self._buffer == None:
            self._buffer = GRPO_buffer(num_steps, self.obs_shape, self.max_new_tokens, grpo_group = self.groupSize)
            self._buffer.zero = False
        data_generator = rollouts.feed_forward_generator(
                    advantages, self.mini_batch_size, update_num)
        del advantages
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_step = 0
        max_rewards = []
        corrects = 0
        for sample in data_generator:
                # with self.accelerator.accumulate(self.actor_critic):
                #     grad_step += 1
                obs_batch, output_ids_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                output_text, output_ids, action_log_probs = self.actor_critic.act_batch(obs_batch, group=self.groupSize)
                #breakpoint()
                penalty, reward, correct_1 = reward_func(output_text, actions_batch, adv_targ, groups= self.groupSize)
                if correct_1 >0:
                    corrects += 1
                #print("value: ", value_preds_batch)
                # epsilon = 10e-5
                # #reward shape is [groupSize, 1]
                # reward = reward.view(-1, self.groupSize) #shape [n, groupSize]
                # r_mean = reward.mean(dim =1).view(-1, 1).repeat(1, self.groupSize) #[n] -> [n, 1] -> [n, groupSize]
                # r_std = reward.std(dim=1).view(-1, 1).repeat(1, self.groupSize)

                max_rewards.append(reward.max().item())
                # #advantage = (reward - reward.mean(dim=1)) / (reward.std(dim=1) + epsilon)
                # advantage = (reward - r_mean)/ (r_std + 1e-5)
                advantage = reward
                #advantage = advantage.view(-1, self.groupSize) #[n, groupSize]
                
                # advantage = (advantage - advantage.mean(dim=1)) / (
                #                 advantage.std(dim=1) + 1e-5)
                #TODO RuntimeError: output with shape [4, 1] doesn't match the broadcast shape [4, 4] in insert self.advantages[self.step].copy_(adv)
                
                self._buffer.insert(obs_batch, output_ids, action_log_probs, return_batch.squeeze(0), actions_batch, advantage, penalty, value_preds_batch.squeeze(0), correct_1)
                del output_ids
                del action_log_probs
                del reward, advantage, obs_batch

        """
        old_data_generator = self._buffer.old_feed_forward_generator()
        if old_data_generator is not None:
            for nsample in old_data_generator:
                # with self.accelerator.accumulate(self.actor_critic):
                #     grad_step += 1
                obs_batch, output_ids_batch, actions_batch, \
                    return_batch, adv_batch, old_action_log_probs_batch, indices = nsample
                output_text, output_ids, action_log_probs = self.actor_critic.act_batch(obs_batch, group=self.groupSize)
                actions_batch = actions_batch.to(torch.int)
                reward = reward_func(output_text, actions_batch, return_batch, groups= self.groupSize)
                epsilon = 10e-5
                #reward shape is [groupSize, 1]
                reward = reward.view(-1, self.groupSize) #shape [n, groupSize]
                r_mean = reward.mean(dim =1).view(-1, 1).repeat(1, self.groupSize) #[n] -> [n, 1] -> [n, groupSize]
                r_std = reward.std(dim=1).view(-1, 1).repeat(1, self.groupSize)
                max_rewards.append(reward.max().item())
                #advantage = (reward - reward.mean(dim=1)) / (reward.std(dim=1) + epsilon)
                advantage = (reward - r_mean)/ (r_std + 1e-5)
                #advantage = advantage.view(-1, self.groupSize) #[n, groupSize]
                
                # advantage = (advantage - advantage.mean(dim=1)) / (
                #                 advantage.std(dim=1) + 1e-5)
                self._buffer.insert_old(obs_batch, output_ids, action_log_probs, return_batch, actions_batch, advantage.permute(1,0), indices)
                del output_ids
                del action_log_probs
                del reward, r_mean, r_std, advantage, obs_batch
        """
        reward_mean = torch.tensor(max_rewards).mean()
        return reward_mean, corrects
        #return action_loss_epoch, dist_entropy_epoch


    def update(self, rollouts, update_num):
        #advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        
        #call generate_inputs first
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_step = 0
        self.actor_critic.train()

        grpo_reward_mean, corrects_total = self.generate_inputs(rollouts)
        
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_step = 0
        
        for e in range(self.ppo_epoch):  #-1 because one epoch has been done in generate_inputs
            data_generator = self._buffer.feed_forward_generator(
                    self.mini_batch_size, update_num)
            
            #no data_generator but groups,  grouped by actions

            for sample in data_generator:
                with self.accelerator.accumulate(self.actor_critic):
                #with self.accelerator.autocast():
                    grad_step += 1
                    obs_batch, output_ids_batch, actions_batch, \
                    return_batch, adv_batch, old_action_log_probs_batch, penalty_batch, value_preds_batch, num_correct= sample
                    

                    output_ids_batch = output_ids_batch.view(-1, output_ids_batch.shape[-1])
                    values, action_log_probs = self.actor_critic.evaluate_actions_batch(
                        obs_batch, output_ids_batch)
                    # values and action_log_probs on two different devices!! because they come from two llava
                    values = values.mean(0, keepdim=True) #values are from same state so taking average of it
                    #breakpoint()
                    if torch.isnan(action_log_probs).any():
                        continue
                    
                    old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                    adv_targ = adv_batch.to(action_log_probs.device).view(-1)
                    value_preds_batch = value_preds_batch.to(values.device)
                    return_batch = return_batch.to(values.device)

                    # print("adv max: ", adv_targ.max())
                    # print("adv min: ", adv_targ.min())
                    # print("log_probs min:", action_log_probs.min(), "max:", action_log_probs.max())
                    # print("old act log prob min: ", old_action_log_probs_batch.min(), "max: ", old_action_log_probs_batch.max())
                    ratio = torch.exp(action_log_probs -
                                    old_action_log_probs_batch)

                    surr1 = ratio * adv_targ.squeeze()
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ.squeeze()
                    ## ratio clip, inspired by https://github.com/huggingface/trl/blob/5a233546ee48532eaeb24b89b8d0042147574688/trl/trainer/ppo_trainer.py#L1199
                    if torch.any(ratio > 10):
                        action_loss = -surr2.sum()
                    else:
                        action_loss = -torch.min(surr1, surr2).sum()
                    
                    num_correct= num_correct.to(action_loss.device).squeeze()
                    if num_correct > 0:
                        action_loss = action_loss / num_correct
                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()

                    try:
                        #assert not torch.isnan(value_loss), "value_loss is nan"
                        assert not torch.isnan(action_loss), "action_loss is nan"
                    except:
                        print("value/action loss is nan")
                        exit(1)
                    penalty_batch = penalty_batch.to(action_log_probs.device)
                    penalty_loss = self.penalty_coef * (action_log_probs * penalty_batch).sum()
                    action_loss = action_loss + penalty_loss
                    loss = value_loss * self.value_loss_coef + action_loss
                    # if loss > 10:
                    #     breakpoint()
                    # if loss < -10:
                    #     breakpoint()
                    # print(loss)
                    # if self.beta != 0.0:
                    #TODO  adding KL divergence loss and entropy

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:

                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()

        value_loss_epoch /= grad_step
        #value_loss_epoch =  0.0  # no value loss in GRPO
        action_loss_epoch /= grad_step  # no need as already taken care of in division by G and num_actions
        dist_entropy_epoch /= grad_step  # no need as already taken care of in division by G and num_actions
        
        if self.save_dir != "":
            if update_num % self.save_interval == 0:
                self.save_checkpoint(update_num)
            self.save_checkpoint("last")
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, grpo_reward_mean.item(), corrects_total#, temp_pred_rewards
    

    def update_RLEF(self, rollouts, update_num):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)


        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_step = 0
        self.actor_critic.train()
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.mini_batch_size, update_num)
            for sample in data_generator:
                with self.accelerator.accumulate(self.actor_critic):
                    grad_step += 1
                    print("step")
                    obs_batch, output_ids_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample
                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs = self.actor_critic.evaluate_actions(
                        obs_batch, output_ids_batch)
                    # values and action_log_probs on two different devices!! because they come from two llava
                    if torch.isnan(action_log_probs).any():
                        continue
                    old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                    adv_targ = adv_targ.to(action_log_probs.device)
                    value_preds_batch = value_preds_batch.to(values.device)
                    return_batch = return_batch.to(values.device)


                    # ratio = torch.exp(action_log_probs -
                    #                 old_action_log_probs_batch)
                    
                    ## ratio clip, inspired by https://github.com/huggingface/trl/blob/5a233546ee48532eaeb24b89b8d0042147574688/trl/trainer/ppo_trainer.py#L1199
                    weighted_advantage = torch.exp(adv_targ/0.05)
                    action_loss = (-action_log_probs * weighted_advantage).mean()
                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()

                    try:
                        assert not torch.isnan(value_loss), "value_loss is nan"
                        assert not torch.isnan(action_loss), "action_loss is nan"
                    except:
                        print("value/action loss is nan")
                        exit(1)
                    loss = value_loss * self.value_loss_coef+action_loss
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:

                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()

        value_loss_epoch /= grad_step
        action_loss_epoch /= grad_step
        dist_entropy_epoch /= grad_step
        if self.accelerator.is_main_process:
            if self.save_dir != "":
                if update_num % self.save_interval == 0:
                    self.save_checkpoint(update_num)
                self.save_checkpoint("last")
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    # def get_trainable_params(self, return_with_names=True):
    #     unwrapped_model = self.accelerator.unwrap_model(self.actor_critic)
    #     breakpoint()
    #     return filter(lambda p: p[1].requires_grad, unwrapped_model.named_parameters())
    
    def get_trainable_params(self):
    # Returns a dict of only trainable params
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.actor_critic)
        return {
            name: param.detach().cpu()
            for name, param in unwrapped_model.named_parameters()
            if param.requires_grad
        }
    
    def save_checkpoint(self,update_num):
        model_state_dict = self.get_trainable_params()
        # model_state_dict = OrderedDict(
        #     {k: v for k, v in self.get_trainable_params()}
        # )
        # self.accelerator.wait_for_everyone()
        # unwrapped_model = self.accelerator.unwrap_model(self.actor_critic)
        # breakpoint()
        # self.accelerator.save(unwrapped_model, self.save_dir + f"/model_{update_num}.checkpoint")
        #breakpoint()
        torch.save(model_state_dict, self.save_dir + f"/model_{update_num}.checkpoint")
        torch.save(self.optimizer.state_dict(), self.save_dir + f"/optimizer_{update_num}.checkpoint")
        print("model saved")


