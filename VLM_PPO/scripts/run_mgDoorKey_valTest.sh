DEVICES="0"
NUM_PROCESS=1
SEED=2            # $1                #
WANDB_RUN="run1"      #$2     #
SAVE_DIR="/home/bahaduri/RL4VLM/outputs/tmp"      #"./"   #$3            #
PORT=29488           #          #29488  $4           #
TEMP_PREDICTOR_FLAG=""                     #"--temp-predictor"             ##"--temp-predictor"
ACT_FREQ_REWARD_FLAG=""                      #"--act-freq-reward"     #""
USE_WANDB_FLAG=""              #"--use-wandb"            #
GROUP="vlm"

#Qwen/Qwen2.5-VL-3B-Instruct
#Qwen/Qwen2-VL-2B-Instruct
if [ ! -d "$SAVE_DIR" ]; then
    mkdir -p "$SAVE_DIR"
    echo "Created directory: $SAVE_DIR"
fi
#test_vlm_val.py

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$DEVICES CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes=$NUM_PROCESS --config_file config_zero2.yaml --main_process_port $PORT ../vlm_traj_preprocess.py \
    --env-name MiniGrid-DoorKey-6x6-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 100 \
    --num-env-steps 15000 \
    --num-steps 512 \
    --grad-accum-steps 128 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.3 \
    --use-gae \
    --seed $SEED \
    --temperature 1.0 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --use-lora \
    --train-vision all \
    --save-dir "$SAVE_DIR" \
    --action-sampling \
    $ACT_FREQ_REWARD_FLAG \
    $TEMP_PREDICTOR_FLAG \
    $USE_WANDB_FLAG \
    --save-interval 2 \
    --wandb-project "minigrid" \
    --wandb-run "$WANDB_RUN" \
    --wandb-group "$GROUP"
    #    --action-sampling  \
    # --use-wandb \
    # --q4
    #"Qwen/Qwen2-VL-2B-Instruct"
    #"liuhaotian/llava-v1.6-mistral-7b"
    #num-steps  512
