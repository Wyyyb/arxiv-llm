from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import faiss
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

    max_new_tokens = 4096
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
        end_index = new_content.index("<|paper_end|>")
        return generated_text[:end_index + len("<|paper_end|>")], None

    print("---------------new_content:\n", new_content)
    # return new_content, output.hidden_states[-1][-1][-1]
    return new_content, cite_rep


def complete_intro(model_path, embedded_corpus_path, title, abstract, partial_intro):
    encoded_corpus, lookup_indices = load_corpus_base(embedded_corpus_path)
    meta_data = load_meta_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path, device)
    input_text = f"<|paper_start|> Title: {title}\nAbstract: {abstract}\nIntroduction\n{partial_intro}"
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
    res = recall_results[0]
    title = titles[0]
    res = "(Reference:" + res
    reference = res.replace("<|reference_start|>", "").replace("<|reference_end|>", "<|cite_end|>")
    print("llm_rerank results", reference)
    return reference


def load_meta_data():
    meta_data_path = "../corpus_data/corpus_data_arxiv_1129.jsonl"
    meta_data = {}
    with open(meta_data_path, "r") as fi:
        for line in tqdm(fi.readlines()):
            curr = json.loads(line)
            if curr["corpus_id"] not in meta_data:
                meta_data[curr["corpus_id"]] = curr
    return meta_data


def load_corpus_base(corpus_dir="../embedded_corpus/1128_shards/"):
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
    # title = "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark"
    # abstract = "In the age of large-scale language models, benchmarks like the Massive Multitask Language Understanding (MMLU) have been pivotal in pushing the boundaries of what AI can achieve in language comprehension and reasoning across diverse domains. However, as models continue to improve, their performance on these benchmarks has begun to plateau, making it increasingly difficult to discern differences in model capabilities. This paper introduces MMLU-Pro, an enhanced dataset designed to extend the mostly knowledge-driven MMLU benchmark by integrating more challenging, reasoning-focused questions and expanding the choice set from four to ten options. Additionally, MMLU-Pro eliminates the trivial and noisy questions in MMLU. Our experimental results show that MMLU-Pro not only raises the challenge, causing a significant drop in accuracy by 16% to 33% compared to MMLU but also demonstrates greater stability under varying prompts. With 24 different prompt styles tested, the sensitivity of model scores to prompt variations decreased from 4-5% in MMLU to just 2% in MMLU-Pro. Additionally, we found that models utilizing Chain of Thought (CoT) reasoning achieved better performance on MMLU-Pro compared to direct answering, which is in stark contrast to the findings on the original MMLU, indicating that MMLU-Pro includes more complex reasoning questions. Our assessments confirm that MMLU-Pro is a more discriminative benchmark to better track progress in the field."
    # partial_intro = "In recent years, advancements in large language models (LLMs) have significantly transformed the field of natural language processing (NLP)."

    # title = "MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks"
    # abstract = "We present MEGA-Bench, an evaluation suite that scales multimodal evaluation to over 500 real-world tasks, to address the highly heterogeneous daily use cases of end users. Our objective is to optimize for a set of high-quality data samples that cover a highly diverse and rich set of multimodal tasks, while enabling cost-effective and accurate model evaluation. In particular, we collected 505 realistic tasks encompassing over 8,000 samples from 16 expert annotators to extensively cover the multimodal task space. Instead of unifying these problems into standard multi-choice questions (like MMMU, MMBench, and MMT-Bench), we embrace a wide range of output formats like numbers, phrases, code, \LaTeX, coordinates, JSON, free-form, etc. To accommodate these formats, we developed over 40 metrics to evaluate these tasks. Unlike existing benchmarks, MEGA-Bench offers a fine-grained capability report across multiple dimensions (e.g., application, input type, output format, skill), allowing users to interact with and visualize model capabilities in depth. We evaluate a wide variety of frontier vision-language models on MEGA-Bench to understand their capabilities across these dimensions."
    # partial_intro = ""

    # title = "VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks"
    # abstract = "Embedding models have been crucial in enabling various downstream tasks such as semantic similarity, information retrieval, and clustering. Recently, there has been a surge of interest in developing universal text embedding models that can generalize across tasks (e.g., MTEB). However, progress in learning universal multimodal embedding models has been relatively slow despite their importance. In this work, we aim to explore the potential for building universal embeddings capable of handling a wide range of downstream tasks. Our contributions are twofold: (1) MMEB (Massive Multimodal Embedding Benchmark), which covers 4 meta-tasks (i.e. classification, visual question answering, multimodal retrieval, and visual grounding) and 36 datasets, including 20 training and 16 evaluation datasets, and (2) VLM2Vec (Vision-Language Model -> Vector), a contrastive training framework that converts any state-of-the-art vision-language model into an embedding model via training on MMEB. Unlike previous models such as CLIP and BLIP, VLM2Vec can process any combination of images and text to generate a fixed-dimensional vector based on task instructions. We build a series of VLM2Vec models on Phi-3.5-V and evaluate them on MMEB's evaluation split. Our results show that VLM2Vec achieves an absolute average improvement of 10% to 20% over existing multimodal embedding models on both in-distribution and out-of-distribution datasets in MMEB."
    # partial_intro = "Embeddings, or distributed representations, encode inputs (whether text or images) as fixed-dimensional vectors, enabling a range of downstream tasks."

    title = "Harnessing Webpage UIs for Text-Rich Visual Understanding"
    abstract = "Text-rich visual understanding-the ability to process environments where dense textual content is integrated with visuals-is crucial for multimodal large language models (MLLMs) to interact effectively with structured environments. To enhance this capability, we propose synthesizing general multimodal instructions from webpage UIs using text-based large language models (LLMs). Despite lacking direct visual input, text-based LLMs are able to process structured text representations from webpage accessibility trees. These instructions are then paired with UI screenshots to train multimodal models. We introduce MultiUI, a dataset containing 7.3 million samples from 1 million websites, covering diverse multimodal tasks and UI layouts. Models trained on MultiUI not only excel in web UI tasks-achieving up to a 48% improvement on VisualWebBench and a 19.1% boost in element accuracy on a web agent dataset Mind2Web-but also generalize surprisingly well to non-web UI tasks and even to non-UI domains, such as document understanding, OCR, and chart interpretation. These results highlight the broad applicability of web UI data for advancing text-rich visual understanding across various scenarios."
    partial_intro = "Text-rich visual understanding, the ability to interpret environments where textual content is densely intertwined with visual elements, is a crucial cognitive skill in humans. For multimodal large language models (MLLMs) "

    embedded_corpus_path = "../embedded_corpus/1129_shards/"
    # model_path = "/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1103/checkpoint-1000/"
    model_path = "/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-3000/"
    result = complete_intro(model_path, embedded_corpus_path, title, abstract, partial_intro)
    os.makedirs("../local_1130", exist_ok=True)
    os.makedirs("../local_1130/test_results_1130", exist_ok=True)
    with open("../local_1130/test_results_1130/harness-ckpt-3000_v1.txt", "w") as fo:
        fo.write(result)


test()

