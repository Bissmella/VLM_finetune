import glob
import os

import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize
from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional

import wandb

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)




class CustomWandbTracker(GeneralTracker):
    name = "wandb"
    requires_logging_directory = False

    @on_main_process
    def __init__(self, args):
        self.args = args
        self.run = None

    @on_main_process
    def init(self):
        run_name = self.args.wandb_run + "-" + self.args.env_name
        self.run_name = run_name
        self.run = wandb.init(project=self.args.wandb_project, name=run_name, group=self.args.wandb_group, job_type=str(self.args.seed), config=self.args)

    @property
    def tracker(self):
        return self.run.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        wandb.config(values)

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        wandb.log(values, step=step)


import torch
import numpy as np
from typing import Dict, Any
from accelerate import Accelerator

def log_metrics(metrics: Dict[str, Any], step: int, accelerator: Accelerator, logger_fn):
    """
    Logs metrics in a safe, process-aware way:
    - Gathers torch tensors across processes (if needed)
    - Converts all values to Python scalars
    - Logs only from the main process
    - Accepts mixed types: scalar, numpy, torch.Tensor
    """
    logged_metrics = {}

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach()
            # Aggregate across processes if tensor is not scalar
            if value.numel() > 1:
                value = accelerator.gather(value.flatten()).float().mean()
            else:
                value = accelerator.gather(value).mean()
            value = value.item()

        elif isinstance(value, np.ndarray):
            value = value.mean().item() if value.size > 1 else float(value.item())

        elif isinstance(value, (list, tuple)):
            # Try to treat as numeric values
            try:
                value = float(np.mean(value))
            except:
                continue  # skip non-numeric

        elif isinstance(value, (int, float)):
            pass  # no conversion needed

        else:
            continue  # skip non-numeric or unrecognized types

        logged_metrics[key] = value
    print("accelerator main proc: ", accelerator.is_main_process)
    if accelerator.is_main_process:
        logger_fn(logged_metrics,)