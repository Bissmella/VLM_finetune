#!/bin/bash
# ============================================================================
# PPO RL Training — Gym-Cards Environments
#
# Unified launch script for PPO training on all gym-cards environments.
# Supports: NumberLine-v0, Blackjack-v0, EZPoints-v0, Points24-v0
#
# Usage:
#   bash run_gymcards.sh <ENV_NAME> [GPU_IDS] [PORT]
#
# Examples:
#   bash run_gymcards.sh gym_cards/NumberLine-v0
#   bash run_gymcards.sh gym_cards/Blackjack-v0 "0,1" 29380
#   bash run_gymcards.sh gym_cards/EZPoints-v0
#   bash run_gymcards.sh gym_cards/Points24-v0
# ============================================================================

ENV_NAME=${1:?"Usage: bash run_gymcards.sh <ENV_NAME> [GPU_IDS] [PORT]"}
GPU_IDS=${2:-"0"}
PORT=${3:-29488}

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_PROCESSES=${#GPU_ARRAY[@]}

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --config_file config_zero2.yaml \
    --main_process_port $PORT \
    ../train_ppo_gymcards.py \
    --env-name $ENV_NAME \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 100 \
    --mini-batch-size 25 \
    --num-processes 1 \
    --use-lora \
    --model-path "Qwen/Qwen2-VL-2B-Instruct" \
    --conv-mode qwen \
    --use-wandb \
    --wandb-project "VLM_RL" \
    --wandb-run "$ENV_NAME"
