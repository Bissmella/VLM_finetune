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

if [ ! -d "$SAVE_DIR" ]; then
    mkdir -p "$SAVE_DIR"
    echo "Created directory: $SAVE_DIR"
fi

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --num_processes=$NUM_PROCESS --config_file config_zero2.yaml --main_process_port $PORT ../main_minigrid.py \
    --env-name MiniGrid-MultiRoom-N2-S4-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 512 \
    --grad-accum-steps 128 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed $SEED \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path "Qwen/Qwen2-VL-2B-Instruct" \
    --use-lora \
    --train-vision all \
    --save-dir "$SAVE_DIR" \
    $ACT_FREQ_REWARD_FLAG \
    $TEMP_PREDICTOR_FLAG \
    $USE_WANDB_FLAG \
    --save-interval 2 \
    --wandb-project "minigrid" \
    --wandb-run "$WANDB_RUN" \
    --wandb-group "$GROUP"
    # --use-wandb \
    # --q4
    #"Qwen/Qwen2-VL-2B-Instruct"
    #"liuhaotian/llava-v1.6-mistral-7b"
    #num-steps  512
