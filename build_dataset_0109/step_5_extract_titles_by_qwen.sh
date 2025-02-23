source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

export CUDA_VISIBLE_DEVICES=0,1,2,3

python step_5_extract_titles_by_qwen.py
