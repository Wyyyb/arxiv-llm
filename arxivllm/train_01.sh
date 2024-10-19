cd /gpfs/public/research/xy/yubowang/arxiv-llm/arxivllm
source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60000 train.py \
  --deepspeed ds_zero3_config.json \
  --output_dir arxivllm \
  --model_name_or_path /gpfs/public/research/xy/yubowang/models/Qwen2.5-7B \
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