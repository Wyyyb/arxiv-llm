from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate_introduction(title, abstract, partial_intro):
    # 加载模型和分词器
    model_path = "/gpfs/public/research/xy/yubowang/arxiv-llm/arxivllm/model_output/checkpoint-140/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 添加特殊token
    special_tokens = ['<|paper_start|>', '<|paper_end|>', '<|cite_start|>', '<|cite_end|>', '<|reference_start|>',
                      '<|reference_end|>']
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 构建输入文本
    input_text = f"Title: {title}\n\nAbstract: {abstract}\n\nIntroduction: <|paper_start|>{partial_intro}"

    # 编码输入文本
    inputs = tokenizer(input_text, return_tensors="pt")

    # 生成续写的introduction
    max_new_tokens = 500  # 您可以根据需要调整这个值
    output = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # 解码输出
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

    # 提取生成的新内容
    new_content = generated_text[len(input_text):]

    return new_content


# 使用示例
title = "Large Language Models for Forecasting and Anomaly Detection: A Systematic Literature Review"
abstract = "This systematic literature review comprehensively examines the application of Large Language Models (LLMs) in forecasting and anomaly detection, highlighting the current state of research, inherent challenges, and prospective future directions. LLMs have demonstrated significant potential in parsing and analyzing extensive datasets to identify patterns, predict future events, and detect anomalous behavior across various domains. However, this review identifies several critical challenges that impede their broader adoption and effectiveness, including the reliance on vast historical datasets, issues with generalizability across different contexts, the phenomenon of model hallucinations, limitations within the models' knowledge boundaries, and the substantial computational resources required. Through detailed analysis, this review discusses potential solutions and strategies to overcome these obstacles, such as integrating multimodal data, advancements in learning methodologies, and emphasizing model explainability and computational efficiency. Moreover, this review outlines critical trends that are likely to shape the evolution of LLMs in these fields, including the push toward real-time processing, the importance of sustainable modeling practices, and the value of interdisciplinary collaboration. Conclusively, this review underscores the transformative impact LLMs could have on forecasting and anomaly detection while emphasizing the need for continuous innovation, ethical considerations, and practical solutions to realize their full potential."
partial_intro = "Language represents a rigorously structured communicative system characterized by its grammar and vocabulary. It serves as the principal medium through which humans articulate and convey meaning."

generated_intro = generate_introduction(title, abstract, partial_intro)
print(generated_intro)

