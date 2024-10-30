source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

cd /gpfs/public/research/xy/yubowang/arxiv-llm
rm -rf qwen_extract_title_data
unzip qwen_extract_title_data.zip

cd /gpfs/public/research/xy/yubowang/arxiv-llm/build_dataset_1028


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python step_4_extract_titles_by_qwen.py
