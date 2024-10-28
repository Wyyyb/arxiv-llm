from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch
import faiss
from transformers import StoppingCriteria, StoppingCriteriaList
import pickle
import json
import numpy as np
import h5py
import json
from tqdm import tqdm
import os
import glob


def single_complete_introduction(model, tokenizer, device, input_text):
    # 编码输入文本
    inputs = tokenizer(input_text, return_tensors="pt").to(device)  # 将输入移动到 GPU
    if len(inputs.input_ids[0]) > 15000:
        return input_text, None
    stop_token_ids = tokenizer.convert_tokens_to_ids(['<|cite_start|>', '<|paper_end|>'])
    print("stop_token_ids", stop_token_ids)
    eos_token_id = stop_token_ids[0]
    # 生成续写的introduction
    max_new_tokens = 2500  # 您可以根据需要调整这个值
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            eos_token_id=eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    print("inputs.input_ids", len(inputs.input_ids[0]))
    print("output_hidden_states", len(output.hidden_states))
    print("output_hidden_states", len(output.hidden_states[-1]))
    print("output_hidden_states", len(output.hidden_states[-1][-1]))
    print("output.sequences[0]", len(output.sequences[0]), output.sequences[0])
    generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=False)

    new_input = tokenizer(generated_text, return_tensors="pt").to(device)
    with torch.no_grad():
        new_output = model(
            new_input.input_ids,
            attention_mask=new_input.attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
    cite_rep = new_output.hidden_states[-1][:, -1, :]

    # 提取生成的新内容
    # new_content = generated_text[len(input_text):]
    new_content = generated_text
    if "<|paper_end|>" in new_content:
        return generated_text, None

    print("---------------new_content:\n", new_content)
    # return new_content, output.hidden_states[-1][-1][-1]
    return new_content, cite_rep


def complete_intro(model_path, embedded_corpus_path, title, abstract, partial_intro):
    encoded_corpus, lookup_indices = load_corpus_base(embedded_corpus_path)
    meta_data = load_meta_data()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path, device)
    # input_text = f"Title: {title}\n\nAbstract: {abstract}\n\nIntroduction: <|paper_start|>{partial_intro}"
    input_text = f"Title: {title}\n\nAbstract: {abstract}\n\nIntroduction: {partial_intro}"
    # input_text = f"<|paper_start|>{partial_intro}"
    input_text, cite_start_hidden_state = single_complete_introduction(model, tokenizer, device, input_text)
    while cite_start_hidden_state is not None:
        retrieved_k_results = retrieve_reference(encoded_corpus, lookup_indices, cite_start_hidden_state, top_k=5)
        reference = llm_rerank(retrieved_k_results, meta_data)
        input_text = input_text + reference
        input_text, cite_start_hidden_state = single_complete_introduction(model, tokenizer, device, input_text)
    print("generated result", input_text)
    return input_text


def load_model(model_path, device):
    # 加载模型配置
    config = AutoConfig.from_pretrained(model_path)

    # 直接加载模型和权重
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
    model.to(device)  # 将模型移动到 GPU

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 添加特殊token
    special_tokens = ['<|paper_start|>', '<|paper_end|>', '<|cite_start|>', '<|cite_end|>', '<|reference_start|>',
                      '<|reference_end|>']
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def llm_rerank(retrieved_k_results, meta_data):
    recall_results = []
    titles = []
    for each in retrieved_k_results:
        index, distance = each
        if index not in meta_data:
            print("index not found in meta_data", index)
            continue
        recall_results.append(meta_data[index]["abstract"])
        titles.append(meta_data[index]["title"])
    # 假设llm就选第一个
    res = recall_results[1]
    title = titles[1]
    res = "(Reference: " + title + ": " + res
    reference = res.replace("<|reference_start|>", "").replace("<|reference_end|>", "<|cite_end|>")
    print("llm_rerank results", reference)
    return reference


def load_meta_data():
    meta_data_path = "../corpus_data/meta_data_1022.jsonl"
    meta_data = {}
    with open(meta_data_path, "r") as fi:
        for line in tqdm(fi.readlines()):
            curr = json.loads(line)
            if curr["docs_id"] not in meta_data:
                meta_data[curr["docs_id"]] = curr
    return meta_data


def load_corpus_base_bk():
    # 尝试加载 HDF5 格式
    try:
        with h5py.File("../embedded_corpus/corpus.0.h5", 'r') as f:
            encoded = f['encoded'][:]
            lookup_indices = f['lookup_indices'][:]
        return encoded, lookup_indices
    except Exception as e:
        print(f"Error loading HDF5: {e}")

    # 如果所有格式都加载失败
    print("Failed to load data from all formats")
    return None, None


def load_corpus_base(corpus_dir="../embedded_corpus/multi_1027/"):
    encoded_list = []
    lookup_indices_list = []

    # 获取目录下所有的.h5文件
    h5_files = sorted(glob.glob(os.path.join(corpus_dir, "*.h5")))

    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {corpus_dir}")

    print(f"Found {len(h5_files)} files to load")

    # 依次读取每个文件
    for file_path in h5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                encoded_list.append(f['encoded'][:])
                lookup_indices_list.append(f['lookup_indices'][:])
            print(f"Successfully loaded {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    # 合并所有数据
    if encoded_list and lookup_indices_list:
        encoded = np.concatenate(encoded_list, axis=0)
        lookup_indices = np.concatenate(lookup_indices_list, axis=0)
        print(f"Combined shape - encoded: {encoded.shape}, lookup_indices: {lookup_indices.shape}")
        return encoded, lookup_indices
    else:
        raise ValueError("No data was successfully loaded")


def retrieve_reference(encoded_corpus, lookup_indices, cite_start_hidden_state, top_k=5):
    print("Retrieving reference")
    # 确保输入是numpy数组
    if isinstance(encoded_corpus, torch.Tensor):
        encoded_corpus = encoded_corpus.cpu().numpy()
    if isinstance(cite_start_hidden_state, torch.Tensor):
        cite_start_hidden_state = cite_start_hidden_state.cpu().numpy()

    # 确保维度正确
    if cite_start_hidden_state.ndim == 1:
        cite_start_hidden_state = cite_start_hidden_state.reshape(1, -1)

    # 获取向量维度
    d = encoded_corpus.shape[1]
    index = faiss.IndexFlatIP(d)

    # 将语料库向量添加到索引中
    faiss.normalize_L2(encoded_corpus)
    index.add(encoded_corpus)

    # 执行搜索
    faiss.normalize_L2(cite_start_hidden_state)
    distances, indices = index.search(cite_start_hidden_state, top_k)
    # 获取对应的 lookup_indices
    retrieved_indices = [str(lookup_indices[i], 'ascii') for i in indices[0]]
    print("retrieved_indices", retrieved_indices)
    print("distances[0]", distances[0])
    # 返回结果
    return list(zip(retrieved_indices, distances[0]))


def test():
    title = "Graph-tree Fusion Model with Bidirectional Information Propagation for Long Document Classification"
    abstract = "Long document classification presents challenges in capturing both local and global dependencies due to their extensive content and complex structure. Existing methods often struggle with token limits and fail to adequately model hierarchical relationships within documents. To address these constraints, we propose a novel model leveraging a graph-tree structure. Our approach integrates syntax trees for sentence encodings and document graphs for document encodings, which capture fine-grained syntactic relationships and broader document contexts, respectively. We use Tree Transformers to generate sentence encodings, while a graph attention network models inter- and intra-sentence dependencies. During training, we implement bidirectional information propagation from word-to-sentence-to-document and vice versa, which enriches the contextual representation. Our proposed method enables a comprehensive understanding of content at all hierarchical levels and effectively handles arbitrarily long contexts without token limit constraints. Experimental results demonstrate the effectiveness of our approach in all types of long document classification tasks."
    partial_intro = "Long document understanding has garnered increasing attention in the field of natural language processing (NLP) due to its wide range of applications across various domains, including legal document analysis, scientific literature categorization, and clinical text mining."
    # title = "Diffusion-based Extreme Image Compression with Compressed Feature Initialization"
    # abstract = "Diffusion-based extreme image compression methods have achieved impressive performance at extremely low bitrates. However, constrained by the iterative denoising process that starts from pure noise, these methods are limited in both fidelity and efficiency. To address these two issues, we present Relay Residual Diffusion Extreme Image Compression (RDEIC), which leverages compressed feature initialization and residual diffusion. Specifically, we first use the compressed latent features of the image with added noise, instead of pure noise, as the starting point to eliminate the unnecessary initial stages of the denoising process. Second, we design a novel relay residual diffusion that reconstructs the raw image by iteratively removing the added noise and the residual between the compressed and target latent features. Notably, our relay residual diffusion network seamlessly integrates pre-trained stable diffusion to leverage its robust generative capability for high-quality reconstruction. Third, we propose a fixed-step fine-tuning strategy to eliminate the discrepancy between the training and inference phases, further improving the reconstruction quality. Extensive experiments demonstrate that the proposed RDEIC achieves state-of-the-art visual quality and outperforms existing diffusion-based extreme image compression methods in both fidelity and efficiency. "
    # partial_intro = "Extreme image compression is becoming increasingly important with the growing demand for efficient storage and transmission of images where storage capacity or bandwidth is limited, such as in satellite communications and mobile devices. Conventional compression standards lik"
    # # embedded_corpus_path = "../embedded_corpus/multi_1027/"
    model_path = "/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/test_1020/checkpoint-140/"
    embedded_corpus_path = "../embedded_corpus/1022/"
    # model_path = "/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/unweighted_1027/checkpoint-600/"
    result = complete_intro(model_path, embedded_corpus_path, title, abstract, partial_intro)


test()

