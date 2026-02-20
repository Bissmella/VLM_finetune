# Fine-Tuning Vision-Language Models for Agentic Tasks

This repository provides a complete pipeline for fine-tuning Vision-Language Models (VLMs) using **Reinforcement Learning (PPO)** and **Supervised Fine-Tuning (SFT)** on agentic tasks. The system trains VLM agents to interact with visual environments by observing rendered frames and producing text-based actions.

## Supported Models

- **Qwen2-VL-2B-Instruct** / **Qwen2.5-VL-3B-Instruct** (primary)
- **LLaVA-Mistral-7B** (legacy)

## Supported Environments

- **[Gym-Cards](gym-cards/)** — Card game environments (NumberLine, Blackjack, EZPoints, Points24)
- **[MiniGrid](https://github.com/Farama-Foundation/Minigrid)** — Grid-world navigation tasks (DoorKey, Empty, etc.)
- **[ALFWorld](VLM_PPO_ALF/)** — Text+vision household tasks (via AI2-THOR)

## Pipeline

```

 │  Data Collection  │────▶│  Data Preprocessing │────▶│    SFT     │────▶│  RL (PPO)     │────▶│  Eval    │
    (Qwen 32B)     

```

1. **Data Collection** — Run a large VLM (Qwen2 32B) in the environment to collect labeled trajectories
2. **Preprocessing** — Convert trajectories into (image, conversation) pairs for SFT
3. **SFT** — Supervised fine-tuning with LoRA on the collected data
4. **RL (PPO)** — Further fine-tune with PPO using environment rewards
5. **Evaluation** — Test the trained agent and generate analysis artifacts

## Repository Structure

```
VLM_finetune/
├── VLM_PPO/                  # Main codebase (RL, SFT, data, evaluation)
│   ├── main.py               #   PPO training — LLaVA on gym-cards
│   ├── main_qwen.py          #   PPO training — Qwen on gym-cards
│   ├── main_minigrid.py      #   PPO training — Qwen on MiniGrid
│   ├── train_sft.py          #   Supervised fine-tuning
│   ├── vlm_traj_label.py     #   Trajectory data collection & labeling
│   ├── vlm_traj_preprocess.py  # Data preprocessing for SFT
│   ├── eval_minigrid.py      #   Evaluation on MiniGrid
│   ├── a2c_ppo_acktr/        #   RL algorithm, policy models, env wrappers
│   ├── SFT/                  #   Custom HuggingFace trainer
│   ├── scripts/              #   Shell scripts for launching experiments
│   └── ...
├── VLM_PPO_ALF/              # ALFWorld variant (AI2-THOR environments)
├── LLaVA/                    # Forked LLaVA repo (patched for RL training)
├── gym-cards/                # Custom gym environments for card games
└── docs/notes/               # Developer notes and scratch files
```

> See [VLM_PPO/README.md](VLM_PPO/README.md) for detailed documentation on all files, scripts, and configs.

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (A100/H100 recommended for large models)
- Conda

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd VLM_finetune

# Create conda environment
conda create -n vlm python=3.10 -y
conda activate vlm

# Install dependencies
cd VLM_PPO
pip install -e .
pip install -r requirements.txt

# Install LLaVA (required for LLaVA-based training)
pip install -e ../LLaVA

# Install gym-cards environment
pip install -e ../gym-cards

# Install additional dependencies
pip install transformers accelerate deepspeed peft bitsandbytes
pip install qwen-vl-utils  # For Qwen models
```

### ALFWorld Setup

See [VLM_PPO_ALF/README.md](VLM_PPO_ALF/README.md) for ALFWorld-specific installation.

## Quick Start

### Train with PPO on MiniGrid

```bash
cd VLM_PPO/scripts
bash run_minigrid_qwen.sh 1 "my_run" "/path/to/output" 29488
```

### Train with SFT

```bash
cd VLM_PPO/scripts
bash run_minigrid_sft.sh /path/to/output
```

### Evaluate

```bash
cd VLM_PPO/scripts
bash eval_minigrid_qwen.sh 1 "eval_run" "/path/to/output" 29488
```



## License

See [LICENSE.txt](LICENSE.txt).
