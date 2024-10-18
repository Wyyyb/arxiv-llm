#conda init
#conda activate arxiv-llm

deepspeed --include localhost:4 --master_port 60000 train.py \
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
  --query_max_len 8192 \
  --passage_max_len 8192 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 1