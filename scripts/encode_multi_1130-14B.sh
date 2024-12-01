source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

cd /gpfs/public/research/xy/yubowang/arxiv-llm/arxivllm

EMBEDDING_OUTPUT_DIR="../embedded_corpus/1130_14B_shards/"
mkdir -p ${EMBEDDING_OUTPUT_DIR}
# dataset_path="../corpus_data/meta_data_1020_sample.jsonl"
dataset_path="../corpus_data/corpus_data_arxiv_1129.jsonl"
model_path="/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1128_multi_cite_14B_10k/checkpoint-1000/"
# model_path="/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1103/checkpoint-2815/"

cp ../tokenizer_files/*.json ${model_path}

for s in {0..3}  # 使用8张卡，从0到7
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
    --dataset_number_of_shards 8 \
    --dataset_shard_index ${s} \
    --encode_output_path ${EMBEDDING_OUTPUT_DIR}/corpus.${s}.pkl &  # 添加 & 实现并行
done
wait  # 等待所有进程完成

