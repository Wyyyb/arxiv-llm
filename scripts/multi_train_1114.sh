#!/bin/bash

export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-60000}}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

cd /gpfs/public/research/xy/yubowang/arxiv-llm/arxivllm
source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

output_dir="../model_output/v1114"
mkdir -p "$output_dir"

torchrun --nproc_per_node $GPUS_PER_NODE \
 --master_addr $MASTER_ADDR \
 --node_rank $NODE_RANK \
 --master_port $MASTER_PORT \
 --nnodes $NNODES \
 train.py \
 --deepspeed ds_zero3_config.json \
 --output_dir ${output_dir} \
 --model_name_or_path /gpfs/public/research/xy/yubowang/models/Qwen2.5-7B \
 --save_steps 500 \
 --dataset_name json \
 --dataset_path ../local/training_data/train_data_1103.jsonl \
 --bf16 \
 --normalize \
 --temperature 0.01 \
 --per_device_train_batch_size 1 \
 --gradient_checkpointing \
 --learning_rate 1e-5 \
 --query_max_len 16384 \
 --passage_max_len 16384 \
 --num_train_epochs 5 \
 --logging_steps 1 \
 --overwrite_output_dir \
 --gradient_accumulation_steps 16\
 --use_flash_attn_2 true

