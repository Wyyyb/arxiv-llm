source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

cd /gpfs/public/research/xy/yubowang/arxiv-llm/arxivllm

EMBEDDING_OUTPUT_DIR="../embedded_corpus/1103/"
mkdir -p ${EMBEDDING_OUTPUT_DIR}
# dataset_path="../corpus_data/meta_data_1020_sample.jsonl"
dataset_path="../corpus_data/meta_data_1103.jsonl"
model_path="/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1103/checkpoint-1000/"

cp ../tokenizer_files/*.json ${model_path}

for s in 1
do
echo ${s}
gpuid=$s
CUDA_VISIBLE_DEVICES=$gpuid python -m encode \
  --output_dir temp \
  --model_name_or_path ${model_path} \
  --bf16 \
  --pooling eos \
  --normalize \
  --per_device_eval_batch_size 16 \
  --query_max_len 32 \
  --passage_max_len 1024 \
  --dataset_name json \
  --dataset_path ${dataset_path} \
  --dataset_number_of_shards 1 \
  --dataset_shard_index ${s} \
  --encode_output_path ${EMBEDDING_OUTPUT_DIR}/corpus.${s}.pkl
done