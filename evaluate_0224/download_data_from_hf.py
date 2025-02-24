from datasets import load_dataset
import json


def download_dataset(output_path='multi_cite_eval.json'):
    try:
        # 使用datasets加载数据集
        # dataset = load_dataset("ubowang/cite-llm-multi-cite-train", split="train")
        dataset = load_dataset("ubowang/cite-llm-multi-cite-eval", split="train")
        data = []
        for item in dataset:
            data.append(item)
        # 将数据集保存为jsonl文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, indent=2))

        print(f"数据已成功下载并保存到: {output_path}")
        print(f"数据集大小: {len(dataset)} 条记录")

    except Exception as e:
        print(f"处理数据时出错: {e}")


if __name__ == "__main__":
    download_dataset()




