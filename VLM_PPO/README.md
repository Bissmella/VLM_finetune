# VLM_PPO — Vision-Language Model Fine-Tuning with RL & SFT

This is the main codebase for fine-tuning Vision-Language Models (VLMs) on agentic tasks using **Proximal Policy Optimization (PPO)** and **Supervised Fine-Tuning (SFT)**. The pipeline supports both LLaVA and Qwen2-VL / Qwen2.5-VL model families.

## Pipeline Overview

```
 ┌──────────────────┐     ┌─────────────────────┐     ┌─────────────┐     ┌──────────────┐
 │  1. Data Collect  │────▶│  2. Data Preprocess  │────▶│   3. SFT     │────▶│  4. RL (PPO)  │
 │  vlm_traj_label   │     │  vlm_traj_preprocess │     │  train_sft   │     │  main_*.py    │
 └──────────────────┘     └─────────────────────┘     └─────────────┘     └──────────────┘
                                                                                │
                                                                           ┌────▼──────┐
                                                                           │ 5. Eval   │
                                                                           │ eval / val │
                                                                           └───────────┘
```

## Supported Environments

| Environment | Entry Point | Scripts |
|---|---|---|
| **Gym-Cards** (NumberLine, Blackjack, EZPoints, Points24) | `main.py` (LLaVA), `main_qwen.py` (Qwen) | `run_nl.sh`, `run_bj.sh`, `run_ezp.sh`, `run_p24.sh` |
| **MiniGrid** (DoorKey-6x6, Empty-Random, etc.) | `main_minigrid.py` (Qwen) | `run_minigrid_qwen.sh` |

## File Reference

### Training Scripts

| File | Description |
|---|---|
| `main.py` | PPO RL training — **LLaVA-Mistral-7B** on gym-cards |
| `main_qwen.py` | PPO RL training — **Qwen2-VL-2B** on gym-cards |
| `main_minigrid.py` | PPO RL training — **Qwen2.5-VL-3B** on MiniGrid |
| `train_sft.py` | Supervised fine-tuning on collected trajectory data |

### Data Collection & Preprocessing

| File | Description |
|---|---|
| `vlm_traj_label.py` | Collect trajectories in MiniGrid and label with Qwen2 (32B) |
| `vlm_score_traj_label.py` | Variant: collect trajectories with quality scoring |
| `vlm_traj_preprocess.py` | Preprocess labeled trajectories into SFT-ready format |

### Evaluation

| File | Description |
|---|---|
| `eval_minigrid.py` | Evaluate a trained agent on MiniGrid (metrics + logging) |
| `test_vlm_val.py` | Validation with captioned trajectory image generation |

### Core Library (`a2c_ppo_acktr/`)

| File | Description |
|---|---|
| `algo/` | PPO algorithm implementation |
| `model.py` | Policy and value network architectures for VLM-based agents |
| `storage.py` | Rollout buffer for PPO trajectory collection |
| `rl_utils.py` | RL utility functions (GAE, reward processing, etc.) |
| `envs.py` | Environment wrappers for gym-cards and MiniGrid |
| `arguments.py` | Command-line argument definitions |
| `distributions.py` | Action distribution classes |
| `utils.py` | General utilities |
| `llava_interface/` | Interface layer for LLaVA model inference |
| `temp_predictor.py` | Temperature predictor auxiliary head |

### SFT Support (`SFT/`)

| File | Description |
|---|---|
| `trainer.py` | `VLMTrainer` — custom HuggingFace Trainer with LoRA checkpoint saving and modality-grouped batching |

### Other

| File | Description |
|---|---|
| `patch.py` | xFormers attention patch for LLaMA (reduces VRAM) |
| `setup.py` | Package setup for `a2c_ppo_acktr` |
| `requirements.txt` | Python dependencies |

## Launch Scripts (`scripts/`)

### Production Scripts (Parameterized)

| Script | Purpose |
|---|---|
| `run_minigrid_qwen.sh` | MiniGrid PPO training (accepts seed, wandb, save dir, flags) |
| `eval_minigrid_qwen.sh` | MiniGrid evaluation (same parameterization) |
| `run_minigrid_sft.sh` | SFT training (accepts save dir) |
| `batch__vlm.sh` | SLURM batch submission for MiniGrid training |

### Quick-Run Scripts (Hardcoded, for quick experiments)

| Script | Environment | Model |
|---|---|---|
| `run_nl.sh` | NumberLine | LLaVA-Mistral-7B |
| `run_nl_qwen.sh` | NumberLine | Qwen2-VL-2B |
| `run_bj.sh` | Blackjack | LLaVA-Mistral-7B |
| `run_ezp.sh` | EZPoints | LLaVA-Mistral-7B |
| `run_p24.sh` | Points24 | LLaVA-Mistral-7B |

### Test/Debug Scripts

| Script | Purpose |
|---|---|
| `run_minigrid_qwen_test.sh` | Quick PPO test with small batch |
| `run_minigrid_sft_test.sh` | Quick SFT test with small batch |
| `run_mgDoorKey_qwen_test.sh` | DoorKey PPO test (with resume) |
| `run_mgDoorKey_valTest.sh` | Data preprocessing test |

### DeepSpeed / Accelerate Configs

| Config | Description |
|---|---|
| `config_zero2.yaml` | Accelerate config for ZeRO Stage 2 |
| `config_zero2_ds.yaml` | Accelerate config with DeepSpeed integration |
| `config_zero3.yaml` | Accelerate config for ZeRO Stage 3 |
| `ds_config.json` | DeepSpeed config for SFT training |

## Quick Start

### 1. Collect Trajectory Data (MiniGrid)

```bash
cd scripts
bash run_mgDoorKey_valTest.sh  # uses vlm_traj_preprocess.py
```

### 2. Supervised Fine-Tuning

```bash
cd scripts
bash run_minigrid_sft.sh /path/to/output/dir
```

### 3. PPO RL Training

```bash
cd scripts
bash run_minigrid_qwen.sh 1 "my_run" "/path/to/output" 29488
```

### 4. Evaluation

```bash
cd scripts
bash eval_minigrid_qwen.sh 1 "eval_run" "/path/to/output" 29488
```
