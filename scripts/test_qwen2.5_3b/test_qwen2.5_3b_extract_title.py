from vllm import LLM, SamplingParams
import time


def extract_titles_from_bibitems(text_list):
    # 初始化模型 (可以根据需要选择不同的模型)
    llm = LLM(model="/gpfs/public/research/xy/yubowang/models/Qwen2.5-3B")

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=0.1,  # 降低随机性，使输出更确定
        top_p=0.95,
        max_tokens=100,  # 根据预期title长度调整
        stop_sequences=["\n"]  # 在换行处停止生成
    )

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
        title = output.text.strip()
        # 如果生成的文本包含多行，只取第一行
        title = title.split('\n')[0]
        titles.append(title)

    return titles


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
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 用两个换行符分割文本
    items = content.split('\n\n\n')

    # 清理每个item
    bibitems = []
    for item in items:
        # 移除首尾空白字符，但保留item内的换行
        cleaned_item = item.strip()
        if cleaned_item:  # 只添加非空的item
            bibitems.append(cleaned_item)

    return bibitems


# 使用示例
if __name__ == "__main__":
    test_samples = read_bibitems_from_file("bibitem_patterns_collect.bbl")

    titles = extract_titles_from_bibitems(test_samples)

    # 打印结果
    for i, (text, title) in enumerate(zip(test_samples, titles)):
        print(f"\nInput {i + 1}:")
        print(f"BibItem: {text}")
        print(f"Extracted Title: {title}")

