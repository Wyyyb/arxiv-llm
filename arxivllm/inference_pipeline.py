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


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop if isinstance(stop, list) else [stop] for stop in stops]
        self.encounters = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class HiddenStateCapture(LogitsProcessor):
    def __init__(self, target_token_id):
        self.target_token_id = target_token_id
        self.hidden_state = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, cur_len) -> torch.FloatTensor:
        if input_ids[0][-1] == self.target_token_id and self.hidden_state is None:
            self.hidden_state = input_ids.hidden_states[-1][:, -1, :]
        return scores


def single_complete_introduction(input_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型配置
    model_path = "/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/test_1020/checkpoint-140/"
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

    # 编码输入文本
    inputs = tokenizer(input_text, return_tensors="pt").to(device)  # 将输入移动到 GPU
    if len(inputs.input_ids[0]) > 15000:
        return input_text, None
    stop_token_ids = tokenizer.convert_tokens_to_ids(['<|cite_start|>', '<|paper_end|>'])
    stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stops=stop_token_ids)])
    hidden_state_capture = HiddenStateCapture(stop_token_ids[0])

    # 生成续写的introduction
    max_new_tokens = 2500  # 您可以根据需要调整这个值
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=1,
            temperature=0.1,
            stopping_criteria=stopping_criteria,
            logits_processor=LogitsProcessorList([hidden_state_capture]),
            output_hidden_states=True,
            return_dict_in_generate=True
        )

    generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=False)

    # 提取生成的新内容
    # new_content = generated_text[len(input_text):]
    new_content = generated_text
    if new_content.endswith("<|paper_end|>"):
        return generated_text, None

    # 获取 <|cite_start|> token 的隐藏状态
    cite_start_hidden_state = hidden_state_capture.hidden_state
    print("new_content", new_content)
    print("cite_start_hidden_state", cite_start_hidden_state.shape)
    return new_content, cite_start_hidden_state


def complete_intro(title, abstract, partial_intro):
    encoded_corpus, lookup_indices = load_corpus_base()
    meta_data = load_meta_data()
    input_text = f"Title: {title}\n\nAbstract: {abstract}\n\nIntroduction: <|paper_start|>{partial_intro}"
    input_text, cite_start_hidden_state = single_complete_introduction(input_text)
    while cite_start_hidden_state:
        retrieved_k_results = retrieve_reference(encoded_corpus, lookup_indices, cite_start_hidden_state, top_k=5)
        reference = llm_rerank(retrieved_k_results, meta_data)
        input_text = input_text + reference
        input_text, cite_start_hidden_state = single_complete_introduction(input_text)
    print("generated result", input_text)
    return input_text


def llm_rerank(retrieved_k_results, meta_data):
    recall_results = []
    for each in retrieved_k_results:
        index, distance = each
        if index not in meta_data:
            print("index not found in meta_data", index)
            continue
        recall_results.append(meta_data[index])
    # 假设llm就选第一个
    res = recall_results[0]
    reference = res.replace("<|reference_start|>", "(Reference: ").replace("<|reference_end|>", "<|cite_end|>")
    print("llm_rerank results", reference)
    return reference


def load_meta_data():
    meta_data_path = "../corpus_data/meta_data_1020.jsonl"
    meta_data = {}
    with open(meta_data_path, "r") as fi:
        for line in fi.readlines():
            curr = json.loads(line)
            if curr["docs_id"] not in meta_data:
                meta_data[curr["docs_id"]] = curr
    return meta_data


def load_corpus_base_bk():
    corpus_base_path = "../embedded_corpus/corpus.0.pkl"
    with open(corpus_base_path, "rb") as fi:
        data = pickle.load(fi)
    encoded, lookup_indices = data
    return encoded, lookup_indices


def load_corpus_base():
    # 尝试加载 HDF5 格式
    try:
        with h5py.File("../embedded_corpus/corpus.0.h5", 'r') as f:
            encoded = f['encoded'][:]
            lookup_indices = f['lookup_indices'][:]
        return encoded, lookup_indices
    except Exception as e:
        print(f"Error loading HDF5: {e}")

    # 如果 HDF5 加载失败，尝试加载 NPZ 格式
    try:
        data = np.load("../embedded_corpus/corpus.0.npz")
        encoded = data['encoded']
        lookup_indices = data['lookup_indices']
        return encoded, lookup_indices
    except Exception as e:
        print(f"Error loading NPZ: {e}")

    # 如果 NPZ 加载失败，尝试加载 JSON 格式
    try:
        with open("../embedded_corpus/corpus.0.json", 'r') as f:
            data = json.load(f)
        encoded = np.array(data['encoded'])
        lookup_indices = np.array(data['lookup_indices'])
        return encoded, lookup_indices
    except Exception as e:
        print(f"Error loading JSON: {e}")

    # 如果所有格式都加载失败
    print("Failed to load data from all formats")
    return None, None


def retrieve_reference(encoded_corpus, lookup_indices, cite_start_hidden_state, top_k=5):
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

    # 创建 GPU 索引
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, d)

    # 将语料库向量添加到索引中
    index.add(encoded_corpus)

    # 执行搜索
    distances, indices = index.search(cite_start_hidden_state, top_k)

    # 获取对应的 lookup_indices
    retrieved_indices = [lookup_indices[i] for i in indices[0]]

    # 返回结果
    return list(zip(retrieved_indices, distances[0]))


def test():
    title = "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark"
    abstract = "In the age of large-scale language models, benchmarks like the Massive Multitask Language Understanding (MMLU) have been pivotal in pushing the boundaries of what AI can achieve in language comprehension and reasoning across diverse domains. However, as models continue to improve, their performance on these benchmarks has begun to plateau, making it increasingly difficult to discern differences in model capabilities. This paper introduces MMLU-Pro, an enhanced dataset designed to extend the mostly knowledge-driven MMLU benchmark by integrating more challenging, reasoning-focused questions and expanding the choice set from four to ten options. Additionally, MMLU-Pro eliminates the trivial and noisy questions in MMLU. Our experimental results show that MMLU-Pro not only raises the challenge, causing a significant drop in accuracy by 16% to 33% compared to MMLU but also demonstrates greater stability under varying prompts. With 24 different prompt styles tested, the sensitivity of model scores to prompt variations decreased from 4-5% in MMLU to just 2% in MMLU-Pro. Additionally, we found that models utilizing Chain of Thought (CoT) reasoning achieved better performance on MMLU-Pro compared to direct answering, which is in stark contrast to the findings on the original MMLU, indicating that MMLU-Pro includes more complex reasoning questions. Our assessments confirm that MMLU-Pro is a more discriminative benchmark to better track progress in the field."
    partial_intro = "In recent years, advancements in large language models (LLMs) have significantly transformed the field of natural language processing (NLP)."
    result = complete_intro(title, abstract, partial_intro)


test()

