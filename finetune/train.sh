#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export model_path="Pipeline_Path" # train base our model
export data_root="../dataset/"
export checkpointing_steps=250
export pipeline_steps=2500
export validation_steps=250
export resolution="49x480x640"
export meta_data="data.jsonl" 

# Model Configuration
MODEL_ARGS=(
    --model_path $model_path
    --model_name "cogvideox1.5-t2v"  # ["cogvideox-t2v"]
    --model_type "t2v"
    --training_type "sft"

    # for mmc
    --use_decouple_modal true
    --use_modal_emb_condgen true

    # for dataset
    --dataset_type "cal" # save while running

    # --check_cache "true" # ! prepare data cache before training

    # for save pipeline
    --pipeline_steps $pipeline_steps 
    # --save_pipeline true # save pipeline
)

# Output Configuration
OUTPUT_ARGS=( 
    --output_dir "./train/"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root $data_root
    --meta_data $meta_data
    --task_keys "rgb" "depth" "canny" "segment" 
    --train_resolution $resolution  # (frames x height x width), frames should be 8N+1 and height, width should be multiples of 16
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10000 # number of training epochs
    --seed 42 # random seed
    --learning_rate 2e-5 # learning rate

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps $checkpointing_steps # save checkpoint every x steps
    --checkpointing_limit 5 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
    --save_checkpoint  true 
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir $data_root
    --validation_steps $validation_steps  # should be multiple of checkpointing_steps
    --validation_prompts "validation_prompts.txt"
    --gen_fps 16
)

# Combine all arguments and launch training
accelerate launch --config_file accelerate_config-2.yaml --main_process_port 29714 train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
