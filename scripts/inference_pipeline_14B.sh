source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

export CUDA_VISIBLE_DEVICES=1

cd ../arxivllm

python inference_pipeline_1130_14B_test.py



