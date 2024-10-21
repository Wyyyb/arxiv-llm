source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

export CUDA_VISIBLE_DEVICES=0

cd ../arxivllm

python inference_pipeline.py



