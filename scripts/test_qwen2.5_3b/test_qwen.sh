source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

cd /gpfs/public/research/xy/yubowang/arxiv-llm/scripts/test_qwen2.5_3b

export CUDA_VISIBLE_DEVICES=1
python test_qwen2.5_3b_extract_title.py
