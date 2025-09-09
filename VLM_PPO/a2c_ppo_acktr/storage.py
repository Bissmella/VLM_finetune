import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

from torch.utils.data import Sampler

class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, max_new_tokens, temporal_predictor=None, act_freq_reward=False, scale = 0.006, grpo=False, utility_function=False, dense_rewards=False):
        self.act_freq_reward = act_freq_reward
        self.temporal_predictor = temporal_predictor
        if self.temporal_predictor:  # is not None
            self.temp_pred_reward = True
        else:
            self.temp_pred_reward = False
        self.grpo = grpo
        self.utility_function = utility_function
        self.dense_rewards_flag = dense_rewards
        self.task_texts = [[None for _ in range(num_processes)] for _ in range(num_steps)]
        self.status = [[None for _ in range(num_processes)] for _ in range(num_steps)]
        self.action_texts = [[None for _ in range(num_processes)] for _ in range(num_steps)]
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.random_mask = torch.zeros(num_steps + 1, num_processes, action_shape)
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        #hard-code to cases of max_new_tokens being smaller than 32
        self.output_ids = torch.zeros(
            num_steps, num_processes, 2*max_new_tokens).long()
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.act_policy = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0
        self.prev_suc_step = 0
        self.temporal_predictor = temporal_predictor
        self.int_reward_scale = scale

    def to(self, device):
        self.obs = self.obs.to(device)
        self.output_ids = self.output_ids.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.random_mask = self.random_mask.to(device)

    def insert_task(self, tasks, command, status):
        self.task_texts[self.step] = tasks
        self.action_texts[self.step] = command
        self.status[self.step] = status
    def insert(self, obs, output_ids, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, rand_mask, action_sampling, fail=False, success=False, ):
        self.obs[self.step + 1].copy_(obs)
        self.output_ids[self.step].copy_(output_ids)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.random_mask[self.step].copy_(rand_mask)  #action was selected randomly and does not come from policy
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.act_policy[self.step].copy_(action_sampling)
        
        self.step = (self.step + 1) % self.num_steps
        if self.grpo:
            if fail and not success:
                self.step =self.prev_suc_step
            elif success and not fail:
                self.prev_suc_step =self.step

    def after_update(self):
        if self.grpo:
            self.prev_suc_step = 0
            return
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.random_mask[0].copy_(self.random_mask[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True,
                        num_update = 0):
        if self.grpo or self.utility_function:
            returns = self.returns
            #returns = 'ab'
            returns[-1] =0
            rewards = self.rewards.clone()
            if rewards [-1] == 0:
                rewards[-1] = 0.08
            mask = self.masks[1:] == 0.0
            mask_r = self.rewards == 0
            rewards[mask & mask_r] = 0.08
            for step in reversed(range(self.rewards.size(0))):
                #try:
                self.returns[step] = rewards[step] + gamma * returns[step+1] * self.masks[step+ 1]
                # except:
                #     breakpoint()
            if self.act_freq_reward:
                self.freq_rewards = self.compute_freq_reward(scale=self.int_reward_scale)
            self.temp_rewards = torch.zeros((1))
            if self.temp_pred_reward:
                temp_rewards = self.get_temp_rewards(self.temporal_predictor)
                temp_rewards = temp_rewards * self.int_reward_scale
                
                if num_update > 0:
                    self.temp_rewards = temp_rewards
            return
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                if self.dense_rewards_flag:
                    rewards = self.dense_rewards
                else:
                    rewards = self.rewards
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]
        print("shape of returns: *****", self.returns.shape)
        if self.act_freq_reward:
            self.freq_rewards = self.compute_freq_reward(scale=self.int_reward_scale)
        self.temp_rewards = torch.zeros((1))
        if self.temp_pred_reward:
            temp_rewards = self.get_temp_rewards(self.temporal_predictor)
            temp_rewards = temp_rewards * self.int_reward_scale
            
            if num_update > 0:
                self.temp_rewards = temp_rewards

    def feed_forward_generator(self,
                               advantages,
                               mini_batch_size=None,
                               update_num=0
                               ):
        if self.grpo:
            batch_size = self.prev_suc_step
        else:
            num_steps, num_processes = self.rewards.size()[0:2]
            batch_size = num_processes * num_steps
        
        if self.temp_pred_reward:
            temp_rewards = self.temp_rewards#self.get_temp_rewards(self.temporal_predictor)
        if self.act_freq_reward:
            freq_rewards = self.freq_rewards
            print("freq shape: ", freq_rewards.shape)
        if self.temp_pred_reward or self.act_freq_reward:
            if update_num > 0:
                print("tmp rewards: ", self.temp_rewards.shape)
                freq_rewards = self.freq_rewards.to(self.temp_rewards.device)
                self.temp_rewards += freq_rewards
            else:
                self.temp_rewards = self.freq_rewards
        
            temp_rewards = self.temp_rewards[:, None, None].to(advantages.device)
            advantages += temp_rewards
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        # obs =  self.obs[:-1].view(-1, *self.obs.size()[2:])
        # actions = self.actions.view(-1, self.actions.size(-1))
        # value_preds = self.value_preds[:-1].view(-1, 1)
        # returns = self.returns[:-1].view(-1, 1)
        # masks = self.masks[:-1].view(-1, 1)
        # action_log_probs = self.action_log_probs.view(-1, 1)
        # if success_samples is not None:
        #     obs = torch.cat((obs, success_samples[0]), dim=0)
        #     actions = torch.cat((actions, SuccessStorage[1]), dim=1)
        #     value_preds = torch.cat((value_preds))
        #breakpoint()
        #print(self.returns)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            output_ids_batch = self.output_ids.view(-1,
                                              self.output_ids.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            act_sampling_batch = self.act_policy.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, output_ids_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, act_sampling_batch
    
    def value_data_generator(self, mini_batch_size =1,):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        sampler = BatchSampler(
            SubsetSampler(range(batch_size)),
            mini_batch_size,
            drop_last=False)
        for indices in sampler:
            obs1_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            obs2_batch = self.obs[1:].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            
            masks_batch = self.masks[1:].view(-1, 1)[indices]

            yield actions_batch, obs1_batch, obs2_batch, masks_batch

    def save_action_log_probs(self, indices, log_probs):
        flat = self.action_log_probs.reshape(-1, 1)
        flat[indices] = log_probs.detach()
        self.action_log_probs.copy_(flat.reshape(self.action_log_probs.shape))

    def copy_success_trajs(self, successRollout):
        # copies successful trajectories from current storage to successRollout
        #successRollout is a deque of dictionaires containing np arrays as values
        status_flattened = self.flatten_envs(self.status)
        status_flattened = [s for s in status_flattened if s is not None]
        traj_starts, traj_ends, traj_lens = self.get_trajectory_bounds_flat()
        for s in enumerate(status_flattened):
            if s == 1:
                start = traj_starts[s]
                end = traj_ends[s]
                successRollout.append({
                    "obs": self.obs[:-1].view(-1, *self.obs.size()[2:])[start:end],
                    "actions": self.actions.view(-1, self.actions.size(-1))[start:end],
                    "output_ids": self.output_ids.view(-1, self.output_ids.size(-1))[start:end],
                    #value_preds should be computed at each sampling time
                    "returns": self.returns[:-1].view(-1, 1)[start:end],
                    "masks": self.masks[:-1].view(-1, 1)[start:end],
                })


    def get_temp_rewards(self, temp_predictor):
        tasks, actions, traj_lens, status = self.extract_trajectories()
        traj_starts, _, _ = self.get_trajectory_bounds_flat()
        traj_starts_indices = traj_starts
        obs_flattened = self.obs[:-1].view(-1, *self.obs.size()[2:])
        
        start_obs = obs_flattened[traj_starts_indices]
        temp_rewards = []
        
        random_masks_flattened = self.random_mask.permute(1, 0, 2).contiguous()
        random_masks_flattened = random_masks_flattened.view(-1)
        random_masks = []

        bad_masks_flattened = self.bad_masks[1:, :, :]
        bad_masks_flattened = bad_masks_flattened.permute(1, 0, 2).contiguous()
        bad_masks_flattened = bad_masks_flattened.view(-1)
        bad_masks = []

        
        
        counter =0
        for i, seg_len in enumerate(traj_lens):
            random_masks.append(random_masks_flattened[counter: counter + seg_len])
            bad_masks.append(bad_masks_flattened[counter+seg_len -1])
            counter += seg_len
        
        temp_rewards = temp_predictor.compute_novelty_batch(tasks, actions, start_obs, traj_lens, random_masks, bad_masks, status)
        
        return temp_rewards

    def extract_trajectories(self):
        actions_flattened = self.flatten_envs(self.action_texts)
        tasks_flattened = self.flatten_envs(self.task_texts)
        status_flattened = self.flatten_envs(self.status)
        
        status_flattened = [s for s in status_flattened if s is not None]
        
        tasks_flattened = [t for t in tasks_flattened if t is not None]
        _, _, trajectory_lengths = self.get_trajectory_bounds_flat()
        trajectory_lengths = trajectory_lengths.tolist()
        
        return tasks_flattened, actions_flattened, trajectory_lengths, status_flattened


    
    def flatten_envs(self, elements):
        num_envs = len(elements[0])
        elements_flatten = []
        for i in range(num_envs):
            tmp_element = [ele[i] for ele in elements]
            elements_flatten.extend(tmp_element)
        return elements_flatten
    
    def compute_freq_reward(self, scale=0.0096):
        # Update count
        tasks_flattened, actions_flattened, trajectory_lengths, status = self.extract_trajectories()
        #actions_flattened = self.flatten_envs(self.action_texts)
        #trajectory_start, trajectory_end, trajectory_lengths = self.get_trajectory_bounds_flat()
        #trajectory_lengths = trajectory_lengths.tolist()
        random_masks_flattened = self.random_mask.permute(1, 0, 2).contiguous()
        random_masks_flattened = random_masks_flattened.view(-1)

        counter = 0
        freq_reward = []
        
        
        for i, traj_len in enumerate(trajectory_lengths):
            actions_counter = {}
            for j, action in enumerate(actions_flattened[counter: counter + traj_len]):
                if action not in actions_counter:
                    actions_counter[action] = 0
                actions_counter[action] += 1
                freq_reward.append((1- random_masks_flattened[counter + j]) * (1 - status[i]) * scale * 1 / np.sqrt(actions_counter[action]))
                #TODO  #TODO consider random_mask consideration as well..
        # Return inverse square root reward
        return torch.Tensor(freq_reward)
    
    def get_traj_start_end(self):
        masks_flattened = self.masks.view(-1)
        masks_flattened = self.masks.permute(1, 0, 2)
        masks_flattened = masks_flattened.view(-1)
        done_indices = (masks_flattened == 0).nonzero(as_tuple=False).squeeze(1)
        trajectory_end = done_indices  # since ends at t
        T = masks_flattened.shape[0] - 1
        if trajectory_end.numel() == 0 or trajectory_end[-1].item() < T:
            trajectory_end = torch.cat([trajectory_end, torch.tensor([T], device=trajectory_end.device)])
        trajectory_start = torch.cat([torch.tensor([0], device=done_indices.device), trajectory_end[:-1]])
        return trajectory_start, trajectory_end
    
    def get_traj_start_end2(self):
        #TODO   do start and end in multidim, can't be done easily in flattened way. or in a foor loop seperately for each process
        masks_flattened = self.masks[:-1]
        masks_flattened = masks_flattened.permute(1, 0, 2) #step, num_processes
        masks_flattened[0] = 0   #first step is start for all processes
        masks_flattened = masks_flattened.view(-1)
        done_indices = (masks_flattened == 0).nonzero(as_tuple=False).squeeze(1)
        trajectory_start = done_indices
        #trajectory_start = torch.cat([[torch.tensor([0], device =done_indices.device), trajectory_start]])
        trjaectory_end = done_indices -1
        T = masks_flattened.shape[0]
        trajectory_end = done_indices  # since ends at t
        T = masks_flattened.shape[0] - 1
        if trajectory_end.numel() == 0 or trajectory_end[-1].item() < T:
            trajectory_end = torch.cat([trajectory_end, torch.tensor([T], device=trajectory_end.device)])
        trajectory_start = torch.cat([torch.tensor([0], device=done_indices.device), trajectory_end[:-1]])
        return trajectory_start, trajectory_end
    
    def get_trajectory_bounds_flat(self):
        masks = self.masks[:-1]
        masks = masks.permute(1, 0, 2)
        masks = masks.squeeze(-1)  # shape: [num_procs, steps]
        num_procs, steps = masks.shape

        is_reset = (masks == 0)
        is_reset[:, 0] = True  # force first timestep as start

        # Find (proc_id, timestep) for starts and ends
        starts = is_reset.nonzero(as_tuple=False)

        padded_reset = torch.concatenate(
            [is_reset[:, 1:], torch.ones((num_procs, 1), dtype=bool, device=masks.device)],
            dim=1
        )
        ends = padded_reset.nonzero(as_tuple=False)
        flat_starts = starts[:, 0] * steps + starts[:, 1]
        flat_ends = ends[:, 0] * steps + ends[:, 1]
        lengths = flat_ends - flat_starts
        lengths = lengths + 1
        
        return flat_starts, flat_ends, lengths




class GRPO_buffer(object):

    def __init__(self, num_steps, obs_shape, max_new_tokens, action_shape=1, temporal_predictor=None, act_freq_reward=False, scale = 0.006, grpo_group=4):
        self.grpo_group = grpo_group
        self.obs = torch.zeros(num_steps + 1, *obs_shape)
        #hard-code to cases of max_new_tokens being smaller than 32
        self.num_steps = num_steps
        self.output_ids = torch.zeros(num_steps, grpo_group,  2*max_new_tokens).long()
        self.rewards = torch.zeros(num_steps, 1)
        self.values = torch.zeros(num_steps, 1)
        self.ncorrects = torch.zeros(num_steps, 1)
        self.advantages = torch.zeros(num_steps, grpo_group, 1)
        self.penalties = torch.zeros(num_steps, grpo_group, 1)
        self.action_log_probs = torch.zeros(num_steps, grpo_group, 1)
        self.actions = torch.zeros(num_steps, action_shape)
        self.step =0
        self.zero = True
        self.empty = True
        self.new_indices = []

    def insert(self, obs, output_ids, action_log_probs, reward, action, adv, penalty, value, ncorrect):
        self.obs[self.step].copy_(obs.squeeze(0))
        self.output_ids[self.step].copy_(output_ids)
        self.action_log_probs[self.step].copy_(action_log_probs.unsqueeze(1))
        self.rewards[self.step].copy_(reward)
        self.values[self.step].copy_(value)
        self.ncorrects[self.step].copy_(torch.tensor([ncorrect]))
        self.actions[self.step].copy_(action.view(-1))
        self.advantages[self.step].copy_(adv)
        self.penalties[self.step].copy_(penalty)
        if self.step == self.num_steps:
            self.empty = False
        self.new_indices.append(self.step)
        self.step = (self.step + 1) % self.num_steps

    def insert_old(self, obs, output_ids, action_log_probs, reward, action, adv, indices):
        self.obs[indices].copy_(obs.squeeze(0))
        self.output_ids[indices].copy_(output_ids)
        self.action_log_probs[indices].copy_(action_log_probs.unsqueeze(1))
        self.rewards[indices].copy_(reward)
        self.actions[indices].copy_(action.view(-1))
        self.advantages[indices].copy_(adv)
        


    def to(self, device):
        self.obs = self.obs.to(device)
        self.output_ids = self.output_ids.to(device)
        self.rewards = self.rewards.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
    
    def comput_advantages(self):
        rewards = self.rewards[:self.step + 1]
        advantages = (rewards - rewards.mean())/rewards.std()  #TODO define the dim
        self.advantages = advantages
        return advantages
    
    def feed_forward_generator(self,
                               mini_batch_size=1,
                               update_num=0
                               ):
        # if self.empty:
        #     batch_size = self.step
        # else:
        batch_size = self.num_steps
        
        print("grpo_buf size: ", batch_size)
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),  #SubsetSampler
            mini_batch_size,
            drop_last=True)
        
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[1:])[indices] #TODO check the -1 in here
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            output_ids_batch = self.output_ids.view(-1,self.grpo_group,
                                              self.output_ids.size(-1))[indices]
            rewards_batch = self.rewards.view(-1, 1)[indices]
            value_batch = self.values.view(-1, 1)[indices]
            penalty_batch = self.penalties.view(-1,self.grpo_group, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, self.grpo_group,
                                                                    1)[indices]
            adv_batch = self.advantages.view(-1, self.grpo_group, 1)[indices]
            ncorrect_batch = self.ncorrects.view(-1, 1)[indices]
            yield obs_batch, output_ids_batch, actions_batch, \
                rewards_batch, adv_batch, old_action_log_probs_batch, penalty_batch, value_batch, ncorrect_batch

    def old_feed_forward_generator(self,
                               mini_batch_size=1,
                               update_num=0
                               ):
        if self.empty:
            batch_size = self.step
        else:
            batch_size = self.num_steps
        
        #all_indices = range(batch_size)
        old_indices = list(set(range(batch_size)) - set(self.new_indices))

        self.new_indices = []
        if len(old_indices) == 0:
            return None
        
        
        #print("grpo_buf size: ", batch_size)
        sampler = BatchSampler(
            SubsetRandomSampler(old_indices),
            mini_batch_size,
            drop_last=True)
        
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[1:])[indices] #TODO check the -1 in here
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            output_ids_batch = self.output_ids.view(-1,self.grpo_group,
                                              self.output_ids.size(-1))[indices]
            rewards_batch = self.rewards.view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, self.grpo_group,
                                                                    1)[indices]
            adv_batch = self.advantages.view(-1, self.grpo_group, 1)[indices]
            yield obs_batch, output_ids_batch, actions_batch, \
                rewards_batch, adv_batch, old_action_log_probs_batch, indices

"""
class SuccessStorage(object):

    # storage for success trajectories:
    # "obs": self.obs[:-1].view(-1, *self.obs.size()[2:])[start:end],
    # "actions": self.actions.view(-1, self.actions.size(-1))[start:end],
    # "output_ids": self.output_ids.view(-1, self.output_ids.size(-1))[start:end],
    # #value_preds should be computed at each sampling time
    # "returns": self.returns[:-1].view(-1, 1)[start:end],
    # "masks"

    def __init__(self, max_size=100):
        self.max_size = max_size
        self.buffer = deque(maxlen= max_size)


    def sample(self, batch_size, value_function, device='cpu'):

        # Randomly samples full trajectories until the approximate number of steps reaches batch_size.
        # Returns stacked tensors.

        import random
        trajs = random.sample(list(self.buffer), min(len(self.buffer), batch_size))
        

        obs = torch.cat([traj['obs'] for traj in self.buffer], dim=0).to(device)
        actions = torch.cat([traj['actions'] for traj in self.buffer], dim=0).to(device)
        output_ids = torch.cat([traj['output_ids'] for traj in self.buffer], dim=0).to(device)
        value_preds = torch.cat([traj['value_preds'] for traj in self.buffer], dim=0).to(device)
        returns = torch.cat([traj['returns'] for traj in self.buffer], dim=0).to(device)
        trajs = random.sample(range(obs.shape[0]), min(obs.shape[0], batch_size))

        obs_sampled = obs[trajs]
        actions_sampled = actions[trajs]
        output_ids_sampled = output_ids[trajs]
        value_preds_sampled = value_preds[trajs]
        returns_sampled = returns[trajs]
        advantages = returns_sampled - value_preds_sampled
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        return obs_sampled, actions_sampled, output_ids_sampled, value_preds_sampled, returns_sampled, advantages
    
    def __len__(self):
        return len(self.buffer)
    
"""
