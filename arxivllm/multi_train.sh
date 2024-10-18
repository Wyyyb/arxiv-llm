#!/bin/bash

export GPUS_PER_NODE=8
export NNODES=4
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-60000}}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

torchrun --nproc_per_node $GPUS_PER_NODE \
 --master_addr $MASTER_ADDR \
 --node_rank $NODE_RANK \
 --master_port $MASTER_PORT \
 --nnodes $NNODES \
 train.py \
 --deepspeed ds_zero3_config.json \
 --output_dir arxivllm \
 --model_name_or_path Qwen/Qwen2.5-1.5B \
 --save_steps 500 \
 --dataset_name json \
 --dataset_path /data/yubowang/cite_llm/data/train_data_1016_0.jsonl \
 --bf16 \
 --normalize \
 --temperature 0.01 \
 --per_device_train_batch_size 1 \
 --gradient_checkpointing \
 --learning_rate 1e-5 \
 --query_max_len 32768 \
 --passage_max_len 32768 \
 --num_train_epochs 1 \
 --logging_steps 1 \
 --overwrite_output_dir \
 --gradient_accumulation_steps 4