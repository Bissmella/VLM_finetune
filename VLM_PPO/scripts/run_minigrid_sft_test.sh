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

deepspeed ../train_sft.py \
    --deepspeed ds_config.json \
    --lora_enable True \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --data_path /home/bahaduri/RL4VLM/outputs/score_trajs/labels.json \
    --image_folder /home/bahaduri/RL4VLM/outputs/labeled_data \
    --bf16 False \
    --fp16 True \
    --output_dir $SAVE_DIR \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --report_to wandb \
    --run_name "sft-train"
    #--version $PROMPT_VERSION \
    #--lazy_preprocess True \
    # --use-wandb \
    # --q4
    #"Qwen/Qwen2-VL-2B-Instruct"
    #"liuhaotian/llava-v1.6-mistral-7b"
    #num-steps  512
