from vllm import LLM, SamplingParams
import time
import json
import os
from tqdm import tqdm


def extract_titles_from_bibitems(llm, sampling_params, text_list):
    # 构建prompts
    prompts = []
    for text in text_list:
        prompt = create_prompt_for_bibitem(text)
        prompts.append(prompt)
    # 批量生成
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print("costing time:", time.time() - start)

    # 提取titles
    titles = []
    for output in outputs:
        # 清理生成的文本
        title = output.outputs[0].text
        # 如果生成的文本包含多行，只取第一行
        title = title.split('\n')[0]
        titles.append(title)

    return titles


def load_model():
    # 初始化模型 (可以根据需要选择不同的模型)
    llm = LLM(model="/gpfs/public/research/xy/yubowang/models/Qwen2.5-3B",
              gpu_memory_utilization=0.8,
              tensor_parallel_size=8)

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=0.1,  # 降低随机性，使输出更确定
        top_p=0.95,
        max_tokens=120,  # 根据预期title长度调整
        stop=["\n"]  # 在换行处停止生成
    )
    return llm, sampling_params


def create_prompt_for_bibitem(bibitem):
    # 创建2-shot示例
    prompt = """从下列bibitem中提取出文章的title，要求干净准确且不遗漏。

Example 1:
Input: \\bibitem{he2015-deep_residual_learning}
K.~He, X.~Zhang, S.~Ren, and J.~Sun.
\\newblock Deep residual learning for image recognition.
\\newblock {\\em arXiv}, 1512.03385, 2015.
Title: Deep residual learning for image recognition

Example 2:
Input: \\bibitem{chakrabarti}
S. Chakrabarti, M. van den Berg and B. Dom,
\\emph{Focused crawling: A new approach to topic-specific web resource discovery},
Computer Networks 31 (1999) 1623--1640.
Title: Focused crawling: A new approach to topic-specific web resource discovery

Input: """ + bibitem + "\nTitle:"

    return prompt


def read_bibitems_from_file(file_path):
    bibitems = []
    with open(file_path, 'r') as fi:
        ori_data = json.load(fi)
        for each in ori_data:
            prompt = create_prompt_for_bibitem(each[2])
            bibitems.append(prompt)
    return bibitems, ori_data


def inference_batch(llm, sampling_params, batch_file_path):
    output_path = batch_file_path.replace("failed_items", "qwen_res")
    if os.path.exists(output_path):
        print("already exists, skip it", output_path)
        return
    bibitems, ori_data = read_bibitems_from_file(batch_file_path)
    title_res = extract_titles_from_bibitems(llm, sampling_params, bibitems)
    if len(title_res) != len(bibitems):
        print("wrong length of title and bibitems", batch_file_path)
        return None
    res = []
    for i in range(len(title_res)):
        res.append([ori_data[i][0], ori_data[i][1], ori_data[i][2], title_res[i]])
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))


def run_on_01(failed_items_dir):
    # 初始化模型 (可以根据需要选择不同的模型)
    llm = LLM(model="/gpfs/public/research/xy/yubowang/models/Qwen2.5-3B",
              gpu_memory_utilization=0.8,
              tensor_parallel_size=8)

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=120,
        stop=["\n"]
    )
    for file in tqdm(os.listdir(failed_items_dir)):
        if not file.endswith(".json"):
            continue
        print("processing", file)
        batch_file_path = os.path.join(failed_items_dir, file)
        try:
            inference_batch(llm, sampling_params, batch_file_path)
        except Exception as e:
            print("error processing", e)


# 使用示例
if __name__ == "__main__":
    run_on_01("../qwen_extract_title_data")



