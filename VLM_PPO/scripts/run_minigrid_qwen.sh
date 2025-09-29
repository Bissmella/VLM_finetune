DEVICES="0"
NUM_PROCESS=1
SEED=$1                #1            # 
WANDB_RUN=$2     #"run1"      #
SAVE_DIR=$3            #"/home/bahaduri/RL4VLM/outputs/vlm_1"      #"./"   #
PORT=$4           #29488           #          #29488
TEMP_PREDICTOR_FLAG=$5                          #"--temp-predictor"             ##"--temp-predictor"
ACT_FREQ_REWARD_FLAG=$6                              #"--act-freq-reward"     #""
GROUP=$7                                               #"vlm-tmp-cur"
ALGO_FLAG=$8
ACT_SAMPLE_FLAG=$9
RLEF_FLAG=${10}
RESUME_FLAG=${11}
WANDB_ID=${12}
START_UPDATE=${13}
USE_WANDB_FLAG="--use-wandb"   #--use-wandb"            #""              #  #MiniGrid-Empty-Random-6x6-v0   #MiniGrid-DoorKey-6x6-v0


if [ ! -d "$SAVE_DIR" ]; then
    mkdir -p "$SAVE_DIR"
    echo "Created directory: $SAVE_DIR"
fi

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --num_processes=$NUM_PROCESS --config_file config_zero2.yaml --main_process_port $PORT ../main_minigrid.py \
    --env-name MiniGrid-DoorKey-6x6-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 16 \
    --num-env-steps 80000 \
    --num-steps 512 \
    --grad-accum-steps 64 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.3 \
    --use-gae \
    --seed $SEED \
    --temperature 1.0 \
    --ppo-epoch 3 \
    --mini-batch-size 1 \
    --model-path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --use-lora \
    --train-vision all \
    --save-dir "$SAVE_DIR" \
    $ACT_SAMPLE_FLAG  \
    $ALGO_FLAG \
    $RLEF_FLAG \
    $ACT_FREQ_REWARD_FLAG \
    $TEMP_PREDICTOR_FLAG \
    $USE_WANDB_FLAG \
    --save-interval 5 \
    --wandb-project "minigrid" \
    --wandb-run "$WANDB_RUN" \
    --wandb-group "$GROUP" \
    $RESUME_FLAG \
    --wandb-id "$WANDB_ID" \
    --start-update $START_UPDATE
    # --use-wandb \
    # --q4
    #"Qwen/Qwen2-VL-2B-Instruct"
    #"liuhaotian/llava-v1.6-mistral-7b"
    #num-steps  512
