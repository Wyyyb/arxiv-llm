source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

cd ../arxivllm
EMBEDDING_OUTPUT_DIR="../embedded_corpus/"
dataset_path="../corpus_data/meta_data_1020_sample.jsonl"
model_path="/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/test_1020/checkpoint-152"
for s in 0 1
do
gpuid=$s
CUDA_VISIBLE_DEVICES=$gpuid python -m encode \
  --output_dir=temp \
  --model_name_or_path ${model_path} \
  --bf16 \
  --pooling eos \
  --normalize \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 1024 \
  --dataset_name json \
  --dataset_path ${dataset_path} \
  --dataset_number_of_shards 2 \
  --dataset_shard_index ${s} \
  --encode_output_path ${EMBEDDING_OUTPUT_DIR}/corpus.${s}.pkl
done