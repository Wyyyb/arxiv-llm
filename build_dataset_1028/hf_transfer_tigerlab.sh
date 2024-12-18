mkdir -p ../scholar_copilot_data
cd ../scholar_copilot_data

cp ../embedded_corpus/scholar-hnsw-1207/index ./
cp ../embedded_corpus/scholar-hnsw-1207/lookup_indices.npy ./
cp ../corpus_data/corpus_data_arxiv_1129.jsonl ./
cp ../local_bibtex_info/bibtex_info_1202.jsonl ./

#huggingface-cli upload TIGER-Lab/ScholarCopilot-Data-v1208 . --repo-type dataset
huggingface-cli upload TIGER-Lab/ScholarCopilot-Data-v1 . --repo-type dataset
#
#cd /gpfs/public/research/xy/yubowang/arxiv-llm/embedded_corpus/scholar-hnsw-1207/
#huggingface-cli upload TIGER-Lab/scholar-hnsw-1207 . --repo-type dataset
#
#
#cd /gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-2000
#huggingface-cli upload TIGER-Lab/ScholarCopilot-v1208 . --repo-type model
#
#
#
#
#mkdir -p ../embedded_corpus
#mkdir -p ../embedded_corpus/scholar-hnsw-1207
#huggingface-cli download TIGER-Lab/scholar-hnsw-1207 --local-dir ../embedded_corpus/scholar-hnsw-1207 --repo-type dataset
#
#mkdir -p ../model_output
#mkdir -p ../model_output/v1208
#

huggingface-cli upload TIGER-Lab/htmls . --repo-type dataset --private
