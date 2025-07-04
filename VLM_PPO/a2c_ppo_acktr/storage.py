import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, max_new_tokens, temporal_predictor=None):
        self.act_freq_reward = True
        self.temp_pred_reward = True
        self.task_texts = [[None for _ in range(num_processes)] for _ in range(num_steps)]
        self.action_texts = [[None for _ in range(num_processes)] for _ in range(num_steps)]
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.random_mask = torch.zeros(num_steps, num_processes, action_shape)
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        #hard-code to cases of max_new_tokens being smaller than 32
        self.output_ids = torch.zeros(
            num_steps, num_processes, 2*max_new_tokens).long()
        self.rewards = torch.zeros(num_steps, num_processes, 1)
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
        self.temporal_predictor = temporal_predictor

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

    def insert_task(self, tasks, command):
        self.task_texts[self.step] = tasks
        self.action_texts[self.step] = command
    def insert(self, obs, output_ids, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, rand_mask):
        self.obs[self.step + 1].copy_(obs)
        self.output_ids[self.step].copy_(output_ids)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.random_mask[self.step].copy_(rand_mask)  #action was selected randomly and does not come from policy
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
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
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
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

    def feed_forward_generator(self,
                               advantages,
                               mini_batch_size=None,
                               ):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if self.temp_pred_reward:
            temp_rewards = self.get_temp_rewards(self.temporal_predictor)
        if self.act_freq_reward:
            freq_rewards = self.compute_freq_reward()
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
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
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, output_ids_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
            
    def get_temp_rewards(self, temp_predictor):
        tasks, actions, traj_lens = self.extract_trajectories()
        traj_starts, _ = self.get_traj_start_end()
        traj_starts_indices = traj_starts + 1
        obs_flattened = self.obs[:-1].view(-1, *self.obs.size()[2:])
        start_obs = obs_flattened[traj_starts_indices]
        temp_rewards = []
        
        random_masks_flattened = self.random_mask.permute(1, 0, 2).contiguous()
        random_masks_flattened = random_masks_flattened.view(-1)
        random_masks = []
        counter =0
        for i, seg_len in enumerate(traj_lens):
            random_masks.append(random_masks_flattened[counter: counter + seg_len])
            counter += seg_len
        temp_rewards = temp_predictor.compute_novelty_batch(tasks, actions, start_obs, traj_lens, random_masks)
        counter =0
        for task, obs, traj_len in zip(tasks, start_obs,  traj_lens):
            temp_reward = temp_predictor.compute_novelty(task, actions[counter: counter + traj_len], obs).tolist()
            temp_reward.append(0) #TODO check if adding 0 is correct, normally the advantage for each trajectory is len of n+1; if n then it should be removed

            temp_rewards.extend(temp_reward)
            counter += traj_len
        return torch.Tensor(temp_rewards)

    def extract_trajectories(self):
        actions_flattened = self.flatten_envs(self.action_texts)
        tasks_flattened = self.flatten_envs(self.task_texts)
        
        tasks_flattened = [t for t in tasks_flattened if t is not None]
        masks_flattened = self.masks.view(-1)
        done_indices = (masks_flattened == 0).nonzero(as_tuple=False).squeeze(1)
        trajectory_end = done_indices  # since ends at t
        T = masks_flattened.shape[0] - 1
        if trajectory_end.numel() == 0 or trajectory_end[-1].item() < T:
            trajectory_end = torch.cat([trajectory_end, torch.tensor([T], device=trajectory_end.device)])
        trajectory_start = torch.cat([torch.tensor([0], device=done_indices.device), trajectory_end[:-1]])
        trajectory_lengths = (trajectory_end - trajectory_start).tolist()
        
        return tasks_flattened, actions_flattened, trajectory_lengths


    
    def flatten_envs(self, elements):
        num_envs = len(elements[0])
        elements_flatten = []
        for i in range(num_envs):
            tmp_element = [ele[i] for ele in elements]
            elements_flatten.extend(tmp_element)
        return elements_flatten
    
    def compute_freq_reward(self, scale=0.0096):
        # Update count
        actions_flattened = self.flatten_envs(self.action_texts)
        masks_flattened = self.masks.view(-1)
        done_indices = (masks_flattened == 0).nonzero(as_tuple=False).squeeze(1)
        trajectory_end = done_indices  # since ends at t
        T = masks_flattened.shape[0] - 1
        if trajectory_end.numel() == 0 or trajectory_end[-1].item() < T:
            trajectory_end = torch.cat([trajectory_end, torch.tensor([T], device=trajectory_end.device)])
        trajectory_start = torch.cat([torch.tensor([0], device=done_indices.device), trajectory_end[:-1]])
        trajectory_lengths = (trajectory_end - trajectory_start).tolist()
        counter = 0
        freq_reward = []
        for traj_len in trajectory_lengths:
            actions_counter = {}
            for action in actions_flattened[counter: counter + traj_len]:
                if action not in actions_counter:
                    actions_counter[action] = 0
                actions_counter[action] += 1
                freq_reward.append(scale * 1 / np.sqrt(actions_counter[action]))
        # Return inverse square root reward
        return torch.Tensor(freq_reward)
    
    def get_traj_start_end(self):
        masks_flattened = self.masks.view(-1)
        done_indices = (masks_flattened == 0).nonzero(as_tuple=False).squeeze(1)
        trajectory_end = done_indices  # since ends at t
        T = masks_flattened.shape[0] - 1
        if trajectory_end.numel() == 0 or trajectory_end[-1].item() < T:
            trajectory_end = torch.cat([trajectory_end, torch.tensor([T], device=trajectory_end.device)])
        trajectory_start = torch.cat([torch.tensor([0], device=done_indices.device), trajectory_end[:-1]])
        return trajectory_start, trajectory_end


