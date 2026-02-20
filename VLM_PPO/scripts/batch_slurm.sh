#!/bin/bash
# ============================================================================
# SLURM Batch Script — MiniGrid PPO RL Training
#
# Submits a Qwen2.5-VL PPO training job on MiniGrid-DoorKey-6x6 via SLURM.
# Activates the conda env and calls run_minigrid_qwen.sh with experiment args.
#
# Usage:  sbatch batch__vlm.sh
# ============================================================================
#SBATCH --partition=electronic
#SBATCH --job-name=vlm_mingrid_1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=48:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err
nvidia-smi -L
source /usr/local/miniconda/etc/profile.d/conda.sh
cd /home/bahaduri/RL4VLM/VLM_PPO/scripts


#"aerosmith,top,zz"
conda activate vrenv-alf
SEED=1
WANDBRUN="vlm_1"
OUTPUT_DIR="/home/bahaduri/RL4VLM/outputs/vlm_1"  #$2
PORT=12352
bash run_minigrid_qwen.sh "$SEED" "$WANDBRUN" "$OUTPUT_DIR" "$PORT"