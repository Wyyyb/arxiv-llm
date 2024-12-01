import gradio as gr
import time


# 模拟函数保持不变
def mock_autocomplete_model(text, num_sentences=3):
    time.sleep(1)
    return text + " [This is a generated sentence 1. This is a generated sentence 2. This is a generated sentence 3.]"


def generate_full_paper(text):
    time.sleep(2)
    return text + " [This is the complete remaining part of the paper, including multiple paragraphs and sections...]"


# 新增模拟引用搜索函数
def mock_search_citation(text):
    """模拟搜索相关引用"""
    time.sleep(1)
    return [
        {"title": "Deep Learning: A Comprehensive Survey", "authors": "Smith et al.", "year": "2023",
         "citation": "[1] Smith et al. Deep Learning: A Comprehensive Survey. Nature, 2023"},
        {"title": "Machine Learning Applications", "authors": "Johnson et al.", "year": "2022",
         "citation": "[2] Johnson et al. Machine Learning Applications. Science, 2022"},
        {"title": "AI in Modern World", "authors": "Williams et al.", "year": "2024",
         "citation": "[3] Williams et al. AI in Modern World. AI Journal, 2024"},
        {"title": "Neural Networks Overview", "authors": "Brown et al.", "year": "2023",
         "citation": "[4] Brown et al. Neural Networks Overview. IEEE, 2023"},
        {"title": "Future of AI", "authors": "Davis et al.", "year": "2023",
         "citation": "[5] Davis et al. Future of AI. AI Review, 2023"}
    ]


def complete_next_sentences(text):
    completion = mock_autocomplete_model(text)
    return [text, completion]


def generate_remaining(text):
    completion = generate_full_paper(text)
    return [text, completion]


def update_text(original, completion, accept):
    if accept:
        return completion
    return original


def insert_selected_citations(text, selected_citations):
    """
    text: 当前文本
    selected_citations: CheckboxGroup返回的选中引用列表
    """
    if not selected_citations:  # 如果没有选中任何引用
        return text

    # 直接使用选中的引用列表
    new_text = text + "\n\nCitations:\n" + "\n".join(selected_citations)
    return new_text


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

    # 新增：引用建议区
    with gr.Row():
        citation_box = gr.Group(visible=True)  # 默认隐藏
        with citation_box:
            gr.Markdown("### Citation Suggestions")
            citation_checkboxes = gr.CheckboxGroup(
                choices=[],
                label="Select citations to insert",
                interactive=True
            )
            insert_citation_btn = gr.Button("Insert selected citations")

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

    def search_and_show_citations():
        citations = mock_search_citation("")
        choices = [cit["citation"] for cit in citations]
        return {
            citation_box: gr.Group(visible=True),
            citation_checkboxes: gr.CheckboxGroup(choices=choices, value=[])  # 添加 value=[] 来清除选择
        }


    citation_btn.click(
        fn=search_and_show_citations,
        inputs=[],
        outputs=[citation_box, citation_checkboxes]
    )

    # 处理选中引用的插入
    insert_citation_btn.click(
        fn=insert_selected_citations,
        inputs=[text_input, citation_checkboxes],
        outputs=[text_input]
    )

app.launch(share=True)


