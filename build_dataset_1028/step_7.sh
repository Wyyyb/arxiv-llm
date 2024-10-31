source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

cd /gpfs/public/research/xy/yubowang/arxiv-llm/build_dataset_1028


export CUDA_VISIBLE_DEVICES=0

python step_7_format_and_split.py