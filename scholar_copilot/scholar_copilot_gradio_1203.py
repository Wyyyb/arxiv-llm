import gradio as gr
import time
import pyperclip  # 用于复制文本到剪贴板
import json
from datetime import datetime
import tempfile
from scholar_copilot_model import *
import torch

citations_data = []

# 添加一个全局变量来追踪已插入的引用
inserted_citations = set()


# 事件处理
def search_and_show_citations():
    global citations_data
    citations_data = mock_search_citation("")
    choices = [cit["citation_key"] + ": " + cit["title"] for cit in citations_data]
    return {
        citation_box: gr.Group(visible=True),
        citation_checkboxes: gr.CheckboxGroup(choices=choices, value=[])
    }


def update_bibtex(selected_citations):
    """根据选中的引用更新BibTeX信息"""
    global citations_data

    # 获取选中引用的索引
    selected_indices = []
    # print("selected_citations", selected_citations)
    # print("citations_data", citations_data)
    for selected in selected_citations:
        for i, cit in enumerate(citations_data):
            if selected.startswith(cit["citation_key"]):
                selected_indices.append(i)
                break

    # 生成BibTeX文本
    bibtex_entries = [citations_data[i]["bibtex"] for i in selected_indices]
    bibtex_text = "\n\n".join(bibtex_entries)

    return bibtex_text


# 模拟函数保持不变
def mock_autocomplete_model(input_text, num_sentences=3):
    output_text, citation_info_list = autocomplete_model(model, tokenizer, device, encoded_corpus,
                                                         lookup_indices, meta_data, citation_map_data,
                                                         input_text, num_sentences=3)
    return output_text, citation_info_list


# 修改mock_search_citation函数，增加bibtex信息
def mock_search_citation(text):
    """模拟搜索相关引用"""
    time.sleep(1)
    return [
        {
            "title": "Deep Learning: A Comprehensive Survey",
            "authors": "Smith et al.",
            "year": "2023",
            "citation_key": "smith2023deep",
            "citation": "[1] Smith et al. Deep Learning: A Comprehensive Survey. Nature, 2023",
            "bibtex": """@article{smith2023deep,
    title={Deep Learning: A Comprehensive Survey},
    author={Smith, John and Doe, Jane},
    journal={Nature},
    year={2023},
    volume={123},
    pages={45--67}
}"""
        },
        {
            "title": "Machine Learning Applications",
            "authors": "Johnson et al.",
            "year": "2022",
            "citation_key": "johnson2022machine",
            "citation": "[2] Johnson et al. Machine Learning Applications. Science, 2022",
            "bibtex": """@article{johnson2022machine,
    title={Machine Learning Applications},
    author={Johnson, Robert and Williams, Mary},
    journal={Science},
    year={2022},
    volume={456},
    pages={89--112}
}"""
        },
        {
            "title": "AI in Modern World",
            "authors": "Williams et al.",
            "year": "2024",
            "citation_key": "williams2024ai",
            "citation": "[3] Williams et al. AI in Modern World. AI Journal, 2024",
            "bibtex": """@article{williams2024ai,
    title={AI in Modern World},
    author={Williams, Peter and Brown, Sarah},
    journal={AI Journal},
    year={2024},
    volume={789},
    pages={113--135}
}"""
        },
        {
            "title": "Neural Networks Overview",
            "authors": "Brown et al.",
            "year": "2023",
            "citation_key": "brown2023neural",
            "citation": "[4] Brown et al. Neural Networks Overview. IEEE, 2023",
            "bibtex": """@article{brown2023neural,
    title={Neural Networks Overview},
    author={Brown, Michael and Davis, Emily},
    journal={IEEE Transactions},
    year={2023},
    volume={234},
    pages={136--158}
}"""
        },
        {
            "title": "Future of AI",
            "authors": "Davis et al.",
            "year": "2023",
            "citation_key": "davis2023future",
            "citation": "[5] Davis et al. Future of AI. AI Review, 2023",
            "bibtex": """@article{davis2023future,
    title={Future of AI},
    author={Davis, Thomas and Wilson, Emma},
    journal={AI Review},
    year={2023},
    volume={567},
    pages={159--181}
}"""
        }
    ]


# 核心功能函数
def complete_next_sentences(text):
    """生成接下来的几个句子"""
    global citations_data
    completion, citation_data_list = mock_autocomplete_model(text)
    citations_data += citation_data_list
    return [text, completion]


def generate_remaining(input_text):
    """生成剩余的完整论文内容"""
    global citations_data
    completion, citation_data_list = autocomplete_model(model, tokenizer, device, encoded_corpus,
                                                        lookup_indices, meta_data, citation_map_data,
                                                        input_text, num_sentences=-1)
    citations_data += citation_data_list
    return [input_text, completion]


def update_text(original, completion, accept):
    """更新文本内容"""
    if accept:
        return completion
    return original


# 修改insert_selected_citations函数来追踪插入的引用
def insert_selected_citations(text, selected_citations):
    """插入选中的引用并追踪"""
    global inserted_citations, citations_data

    if not selected_citations:
        return text

    # 更新已插入的引用集合
    for selected in selected_citations:
        citation_key = selected.split(": ")[0]
        for citation in citations_data:
            if citation["citation_key"] == citation_key:
                inserted_citations.add(json.dumps(citation))  # 使用JSON字符串作为集合元素

    selected_citations = [each.split(": ")[0] for each in selected_citations]
    citations = ", ".join(selected_citations)
    new_text = text + " \\cite{" + citations + "}"
    return new_text


# 添加下载历史引用的函数
def download_citation_history():
    """生成包含所有历史引用的BibTeX文件"""
    global inserted_citations

    if not inserted_citations:
        return None  # 如果没有引用历史，返回None

    citations = [json.loads(cit) for cit in inserted_citations]
    bibtex_entries = [cit["bibtex"] for cit in citations]
    content = "\n\n".join(bibtex_entries)

    # 添加时间戳注释
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"% Citation history generated at {timestamp}\n% Total citations: {len(citations)}\n\n"

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".bib") as temp_file:
        temp_file.write(header + content)
        temp_file_path = temp_file.name

    return temp_file_path


model_path = "/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-2000/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_model(model_path, device)
embedded_corpus_path = "../embedded_corpus/1129_shards/"
encoded_corpus, lookup_indices = load_corpus_base(embedded_corpus_path)
meta_data = load_meta_data()
citation_map_data_path = "../local_bibtex_info/bibtex_info_1202.jsonl"
citation_map_data = load_citation_map_data(citation_map_data_path)

with gr.Blocks() as app:
    gr.Markdown("# Scholar Copilot - Your Academic Writing Assistant")

    with gr.Row():
        # 左侧编辑区
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=30,
                label="Write your paper here",
                placeholder="Start writing your academic paper..."
            )

            with gr.Row():
                complete_btn = gr.Button("Complete me (3 sentences)")
                generate_btn = gr.Button("Generate to the end")
                citation_btn = gr.Button("Insert citation")

        # 右侧预览区
        with gr.Column(scale=2):
            preview = gr.Textbox(
                lines=30,
                label="Preview of completion",
                interactive=False
            )

            with gr.Row():
                accept_btn = gr.Button("Accept completion")
                reject_btn = gr.Button("Reject completion")

    # 引用建议区
    with gr.Row():
        citation_box = gr.Group(visible=True)
        with citation_box:
            gr.Markdown("### Citation Suggestions")
            citation_checkboxes = gr.CheckboxGroup(
                choices=[],
                label="Select citations to insert",
                interactive=True
            )
            insert_citation_btn = gr.Button("Insert selected citations")

    with gr.Row():
        # copy_btn = gr.Button("Copy BibTeX")
        download_history_btn = gr.Button("Download Citation History")
        copy_status = gr.Textbox(
            value="",
            label="",
            interactive=False,
            show_label=False
        )
        # with gr.Column():
        #     gr.Markdown("### BibTeX Information")
        #     bibtex_info = gr.TextArea(
        #         label="BibTeX entries",
        #         interactive=False,
        #         lines=10,
        #         container=False
        #     )
        #     with gr.Row():
        #         copy_btn = gr.Button("Copy BibTeX")
        #         download_history_btn = gr.Button("Download Citation History")
        #         copy_status = gr.Textbox(
        #             value="",
        #             label="",
        #             interactive=False,
        #             show_label=False
        #         )

    # 原有事件处理
    complete_btn.click(
        fn=complete_next_sentences,
        inputs=[text_input],
        outputs=[text_input, preview]
    )

    generate_btn.click(
        fn=generate_remaining,
        inputs=[text_input],
        outputs=[text_input, preview]
    )

    accept_btn.click(
        fn=update_text,
        inputs=[text_input, preview, gr.Checkbox(value=True, visible=False)],
        outputs=[text_input]
    )

    reject_btn.click(
        fn=update_text,
        inputs=[text_input, preview, gr.Checkbox(value=False, visible=False)],
        outputs=[text_input]
    )

    citation_btn.click(
        fn=search_and_show_citations,
        inputs=[],
        outputs=[citation_box, citation_checkboxes]
    )

    insert_citation_btn.click(
        fn=insert_selected_citations,
        inputs=[text_input, citation_checkboxes],
        outputs=[text_input]
    )

    # 选中引用时更新BibTeX
    citation_checkboxes.change(
        fn=update_bibtex,
        inputs=[citation_checkboxes],
        outputs=[bibtex_info]
    )

    download_history_btn.click(
        fn=download_citation_history,
        inputs=[],
        outputs=[gr.File()]
    )

    # 修改复制功能
    def copy_to_clipboard(text):
        """返回要复制的文本和成功消息"""
        try:
            pyperclip.copy(text)
            return text, "✓ Copied!"
        except Exception as e:
            print("Exception in copy_to_clipboard", e)
            return text, "Failed to copy"


    def clear_status():
        time.sleep(0.5)  # 延迟2秒
        return ""


    # 复制BibTeX按钮事件
    # copy_btn.click(
    #     fn=copy_to_clipboard,
    #     inputs=[bibtex_info],
    #     outputs=[bibtex_info, copy_status]
    # ).then(
    #     fn=clear_status,
    #     inputs=None,
    #     outputs=copy_status
    # )

app.launch(share=True)
# app.launch()

