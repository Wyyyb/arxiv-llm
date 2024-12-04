import gradio as gr
import time
import pyperclip  # 用于复制文本到剪贴板


# 模拟函数保持不变
def mock_autocomplete_model(text, num_sentences=3):
    time.sleep(1)
    return text + " [This is a generated sentence 1. This is a generated sentence 2. This is a generated sentence 3.]"


def generate_full_paper(text):
    time.sleep(2)
    return text + " [This is the complete remaining part of the paper, including multiple paragraphs and sections...]"


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
    completion = mock_autocomplete_model(text)
    return [text, completion]


def generate_remaining(text):
    """生成剩余的完整论文内容"""
    completion = text + " [This is the complete remaining part of the paper, " \
                        "including multiple paragraphs and sections...]"
    return [text, completion]


def update_text(original, completion, accept):
    """更新文本内容"""
    if accept:
        return completion
    return original


def insert_selected_citations(text, selected_citations):
    """插入选中的引用"""
    if not selected_citations:
        return text
    selected_citations = [each.split(": ")[0] for each in selected_citations]
    citations = ", ".join(selected_citations)
    new_text = text + " \\cite{" + citations + "}"
    return new_text


citations_data = []


with gr.Blocks() as app:
    gr.Markdown("# Scholar Copilot - Your Academic Writing Assistant")

    with gr.Row():
        # 左侧编辑区
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=15,
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
                lines=15,
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

    # BibTeX区域
    with gr.Row():
        with gr.Column():
            gr.Markdown("### BibTeX Information")
            bibtex_info = gr.TextArea(
                label="BibTeX entries",
                interactive=False,
                lines=10,
                container=False
            )
            with gr.Row():
                copy_btn = gr.Button("Copy BibTeX")
                copy_status = gr.Textbox(
                    value="",
                    label="",
                    interactive=False,
                    show_label=False
                )

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
    copy_btn.click(
        fn=copy_to_clipboard,
        inputs=[bibtex_info],
        outputs=[bibtex_info, copy_status]
    ).then(
        fn=clear_status,
        inputs=None,
        outputs=copy_status
    )

# app.launch(share=True)
app.launch()

