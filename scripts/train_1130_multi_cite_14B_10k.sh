#!/bin/bash

# 创建日志文件，使用时间戳作为文件名
log_dir="../logs"
mkdir -p "$log_dir"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/training_${timestamp}.log"
cd /gpfs/public/research/xy/yubowang/arxiv-llm/arxivllm

# 将所有标准输出和错误输出重定向到日志文件，同时在终端显示
{
export GPUS_PER_NODE=8
export NNODES=4
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-60000}}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

output_dir="../model_output/v1130_multi_cite_14B_10k"
mkdir -p "$output_dir"

# 记录开始时间和基本信息
echo "Training started at $(date)"
echo "Output directory: $output_dir"
echo "Log file: $log_file"

torchrun --nproc_per_node $GPUS_PER_NODE \
 --master_addr $MASTER_ADDR \
 --node_rank $NODE_RANK \
 --master_port $MASTER_PORT \
 --nnodes $NNODES \
 train.py \
 --deepspeed ds_zero3_config.json \
 --output_dir ${output_dir} \
 --model_name_or_path /gpfs/public/research/xy/yubowang/models/Qwen2.5-14B \
 --save_steps 1000 \
 --dataset_name json \
 --dataset_path ../local_1128_14B_10k/train_data_1128_14B_10k.jsonl \
 --bf16 \
 --normalize \
 --temperature 0.01 \
 --per_device_train_batch_size 1 \
 --gradient_checkpointing \
 --learning_rate 1e-5 \
 --query_max_len 10200 \
 --passage_max_len 10200 \
 --num_train_epochs 3 \
 --logging_steps 1 \
 --overwrite_output_dir \
 --gradient_accumulation_steps 16

echo "Training finished at $(date)"
} 2>&1 | tee -a "$log_file"