import gradio as gr
from datetime import datetime
import tempfile
from scholar_copilot_model_1206 import *
import torch
import faiss
import time


def generate_citation(input_text):
    global index
    new_input_text = input_text + " <|cite_start|>"
    new_input = tokenizer(new_input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        new_output = model(
            new_input.input_ids,
            attention_mask=new_input.attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
    cite_rep = new_output.hidden_states[-1][:, -1, :]
    retrieved_k_results = retrieve_reference(index, encoded_corpus, lookup_indices, cite_rep, top_k=10)
    searched_citations = []
    for each in retrieved_k_results:
        index, distance = each
        print("index", index)
        if index not in meta_data:
            print("index not found in meta_data", index)
            continue
        paper_id = meta_data[index]["paper_id"]
        print("paper_id", paper_id)
        citation_info = citation_map_data[paper_id]
        print("citation_info", citation_info)
        searched_citations.append(citation_info)
    return searched_citations


def split_yield_list(input_text, prefix_length):
    # print("split_yield_list input_text", input_text)
    # print("split_yield_list prefix_length", prefix_length)
    prefix_text = input_text[:prefix_length]
    text = input_text[prefix_length:]
    # print("split_yield_list text", text)
    text_list = text.split(" ")
    return prefix_text, text_list


def stream_complete_3_sentence(text, progress=gr.Progress()):
    global citations_data
    sentence_num = 0
    enough = False
    current_text = text
    current_text = preprocess_input_text(current_text)
    input_text_length = len(current_text)
    curr_prefix_length = len(current_text) - len("<|paper_start|> ")
    current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
    reference_id_list = []
    display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
    citations_data += citation_data_list
    curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
    for each in yield_list:
        if "." in each and (each.endswith(".") or ".\n" in each):
            sentence_num += 1
            print("sentence_num: ", sentence_num, "each", each)
        curr_yield_text += " " + each
        yield curr_yield_text
        if sentence_num == 3:
            enough = True
            display_text = curr_yield_text
            break
        time.sleep(0.1)
    curr_prefix_length = len(current_text) - len("<|paper_start|> ")
    while cite_start_hidden_state is not None and not enough:
        # enough_sentences, res_text = cut_after_third_sentence(current_text[input_text_length:], 3)
        # if enough_sentences:
        #     current_text = res_text
        #     break
        retrieved_k_results = retrieve_reference(index, encoded_corpus, lookup_indices, cite_start_hidden_state, top_k=1)
        reference, curr_index = llm_rerank(retrieved_k_results, meta_data)
        reference_id_list.append(curr_index)
        current_text = current_text + reference
        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
        display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
        citations_data += citation_data_list
        curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
        for each in yield_list:
            if "." in each and (each.endswith(".") or ".\n" in each):
                sentence_num += 1
                print("sentence_num: ", sentence_num, "each", each)
            curr_yield_text += " " + each
            yield curr_yield_text
            if sentence_num == 3:
                enough = True
                display_text = curr_yield_text
                break
            time.sleep(0.1)
        curr_prefix_length = len(current_text) - len("<|paper_start|> ")
    # print("11 current_text", current_text)
    # display_text, _ = replace_citations(current_text, reference_id_list, citation_map_data)
    # print("22 display_text", display_text)
    display_text, citation_data_list = post_process_output_text(display_text, reference_id_list, citation_map_data)
    # print("33 display_text", display_text)
    citations_data += citation_data_list
    print("global citations_data", citations_data)
    yield display_text
    time.sleep(0.1)


def stream_generate(text, progress=gr.Progress()):
    global citations_data
    current_text = text
    current_text = preprocess_input_text(current_text)
    input_text_length = len(current_text)
    curr_prefix_length = len(current_text) - len("<|paper_start|> ")
    current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
    reference_id_list = []
    display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
    citations_data += citation_data_list
    curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
    for each in yield_list:
        curr_yield_text += " " + each
        yield curr_yield_text
        time.sleep(0.1)
    curr_prefix_length = len(current_text) - len("<|paper_start|> ")
    while cite_start_hidden_state is not None:
        # enough_sentences, res_text = cut_after_third_sentence(current_text[input_text_length:], 3)
        # if enough_sentences:
        #     current_text = res_text
        #     break
        retrieved_k_results = retrieve_reference(index, encoded_corpus, lookup_indices, cite_start_hidden_state,
                                                 top_k=1)
        reference, curr_index = llm_rerank(retrieved_k_results, meta_data)
        reference_id_list.append(curr_index)
        current_text = current_text + reference
        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
        display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
        citations_data += citation_data_list
        curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
        sentence_num = 0
        enough = False
        for each in yield_list:
            if "." in each and each.endswith("."):
                sentence_num += 1
            curr_yield_text += " " + each
            yield curr_yield_text
            # if sentence_num == 3:
            #     enough = True
            #     break
            time.sleep(0.1)
        if enough:
            break
        curr_prefix_length = len(current_text) - len("<|paper_start|> ")
    # print("11 current_text", current_text)
    display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
    citations_data += citation_data_list
    # print("22 display_text", display_text)
    display_text, citation_data_list = post_process_output_text(display_text, reference_id_list, citation_map_data)
    # print("33 display_text", display_text)
    citations_data += citation_data_list
    yield display_text
    time.sleep(0.1)


def search_and_show_citations(input_text):
    global citations_data
    curr_citations_data = generate_citation(input_text)
    citations_data += curr_citations_data
    choices = [cit["citation_key"] + ": " + cit["title"] for cit in curr_citations_data]
    return {
        citation_box: gr.Group(visible=True),
        citation_checkboxes: gr.CheckboxGroup(choices=choices, value=[])
    }


def insert_selected_citations(text, selected_citations):
    """插入选中的引用并追踪"""
    global citations_data

    if not selected_citations:
        return text

    selected_citations = [each.split(": ")[0] for each in selected_citations]
    citations = ", ".join(selected_citations)
    new_text = text + " \\cite{" + citations + "}"
    return new_text


def download_citation_history():
    """生成包含所有历史引用的BibTeX文件"""
    global citations_data
    print("citations_data", citations_data)
    if not citations_data:
        return None  # 如果没有引用历史，返回None

    bibtex_entries = []
    for cit in citations_data:
        if cit["bibtex"] not in bibtex_entries:
            bibtex_entries.append(cit["bibtex"])
    content = "\n\n".join(bibtex_entries)

    # 添加时间戳注释
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"% Citation history generated at {timestamp}\n% Total citations: {len(bibtex_entries)}\n\n"

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".bib") as temp_file:
        temp_file.write(header + content)
        temp_file_path = temp_file.name

    return temp_file_path


def clear_cache():
    global citations_data
    citations_data = []
    return "", []


with gr.Blocks(theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate",
)) as app:
    # Logo和标题区
    with gr.Row(elem_classes="header-container"):
        with gr.Column(scale=1):
            gr.Image(value="https://cdn-avatars.huggingface.co/v1/production/uploads/6313a86154e6e5d9f0f94e04/Noi3Qq3RYz8Jdq6BaFteq.png",  # 这里使用emoji作为临时logo，您可以替换为实际logo图片URL
                     width=100,
                     height=100,
                     show_label=False,
                     container=False,
                     interactive=False,
                     elem_classes="logo-image")
        with gr.Column(scale=4):
            gr.Markdown(
                """
                # Scholar Copilot - Your Academic Writing Assistant
                ### Elevate Your Academic Writing with AI-Powered Assistance
                """
            )

    # 介绍文本
    gr.Markdown(
        """
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <h4>Welcome to Scholar Copilot! 📚</h4>
        <p>Scholar Copilot improves the academic writing process by seamlessly integrating automatic text completion and intelligent citation suggestions into a cohesive, human-in-the-loop AI-driven pipeline. Designed to enhance productivity and creativity, it provides researchers with high-quality text generation and precise citation recommendations powered by iterative and context-aware Retrieval-Augmented Generation (RAG).</p>
        <p>The current version of Scholar Copilot leverages a state-of-the-art 7-billion-parameter language model (LLM) trained on the complete Arxiv full paper corpus. This unified model for retrieval and generation is adept at making context-sensitive decisions about when to cite, what to cite, and how to generate coherent content based on reference papers.</p>
        <p>The demo supports three core features tailored to the academic workflow:</p>
        <ul>
            <li>📝 Next-3-Sentence Suggestions: Facilitates writing by predicting the next sentences with automatic retrieval and citation of relevant reference papers.</li>
            <li>🔍 Citation Suggestions on Demand: Provides precise, contextually appropriate paper citations whenever needed.</li>
            <li>✨ Full Section Auto-Completion: Assists in brainstorming and drafting comprehensive paper content and structure.</li>
            <li> Enhance your writing quality</li>
        </ul>
        <p>Start writing in the editor below, and let Scholar Copilot assist you in creating outstanding academic papers.</p>
        </div>
        """
    )

    with gr.Row():
        # 主编辑区
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=30,
                label="Write your paper here",
                placeholder="Start writing your academic paper...",
                container=True,
                elem_classes="main-textarea"
            )

            with gr.Row():
                complete_btn = gr.Button(
                    "Complete me (3 sentences)",
                    elem_classes="custom-button",
                    variant="primary"
                )
                generate_btn = gr.Button(
                    "Generate to the end",
                    elem_classes="custom-button",
                    variant="primary"
                )
                citation_btn = gr.Button(
                    "Insert citation",
                    elem_classes="custom-button",
                    variant="secondary"
                )
                clear_btn = gr.Button(
                    "Clear All",
                    elem_classes="custom-button",
                    variant="stop"
                )

    # 引用建议区
    with gr.Row():
        citation_box = gr.Group(visible=True)
        with citation_box:
            gr.Markdown("### 📑 Citation Suggestions")
            citation_checkboxes = gr.CheckboxGroup(
                choices=[],
                label="Select citations to insert",
                interactive=True,
                elem_classes="citation-checkboxes"
            )
            insert_citation_btn = gr.Button(
                "Insert selected citations",
                elem_classes="custom-button",
                variant="secondary"
            )

    with gr.Row():
        download_history_btn = gr.Button(
            "📥 Download Citation History",
            elem_classes="custom-button",
            variant="secondary"
        )
        copy_status = gr.Textbox(
            value="",
            label="",
            interactive=False,
            show_label=False
        )

    # CSS样式部分更新
    gr.HTML(
        """
        <style>
        /* 设置全局字体 */
        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Helvetica Neue', 'Microsoft YaHei', sans-serif;
        }

        .header-container {
            background: linear-gradient(90deg, #1a237e, #283593);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: white;
            display: flex;
            align-items: center;
        }

        /* 标题样式 */
        .header-container h1 {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            font-weight: 600;
            font-size: 28px;
            margin-bottom: 8px;
        }

        .header-container h3 {
            font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            font-weight: 400;
            font-size: 16px;
            opacity: 0.9;
        }

        /* 介绍文本样式 */
        .introduction h4 {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            font-weight: 500;
            font-size: 20px;
            margin-bottom: 16px;
        }

        .introduction p, .introduction li {
            font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            font-size: 15px;
            line-height: 1.6;
            color: #2c3e50;
        }

        .logo-image {
            width: 100px !important;
            height: 100px !important;
            object-fit: contain;
            background-color: transparent;
        }

        .logo-image > div {
            border: none !important;
        }

        .logo-image img {
            width: 100% !important;
            height: 100% !important;
            object-fit: contain;
            pointer-events: none;
        }

        .logo-image .controls {
            display: none !important;
        }

        .logo-image img:hover {
            transform: none !important;
        }

        .main-textarea {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #ffffff;
            font-family: 'SF Mono', Menlo, Monaco, Consolas, 'Courier New', monospace;
            font-size: 14px;
        }

        .custom-button {
            font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-weight: 500;
            font-size: 14px;
            padding: 8px 16px;
            margin: 5px;
            border-radius: 6px;
            transition: all 0.3s ease;
        }

        .custom-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .citation-checkboxes {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 14px;
        }

        /* Citation标题样式 */
        .citation-box h3 {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-weight: 500;
            font-size: 18px;
            margin-bottom: 12px;
        }

        :root {
            --primary-color: #1a237e;
            --secondary-color: #283593;
            --background-color: #f5f7fa;
        }

        body {
            background-color: var(--background-color);
        }
        </style>
        """
    )

    # 事件处理保持不变
    complete_btn.click(
        fn=stream_complete_3_sentence,
        inputs=[text_input],
        outputs=[text_input],
        queue=True
    )

    generate_btn.click(
        fn=stream_generate,
        inputs=[text_input],
        outputs=[text_input],
        queue=True
    )

    citation_btn.click(
        fn=search_and_show_citations,
        inputs=[text_input],
        outputs=[citation_box, citation_checkboxes]
    )

    insert_citation_btn.click(
        fn=insert_selected_citations,
        inputs=[text_input, citation_checkboxes],
        outputs=[text_input]
    )

    download_history_btn.click(
        fn=download_citation_history,
        inputs=[],
        outputs=[gr.File()]
    )

    clear_btn.click(
        fn=clear_cache,
        inputs=[],
        outputs=[text_input, citation_checkboxes]
    )

if __name__ == "__main__":
    model_path = "/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-2000/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path, device)
    embedded_corpus_path = "../embedded_corpus/1129_shards/"
    encoded_corpus, lookup_indices = load_corpus_base(embedded_corpus_path)
    meta_data = load_meta_data()
    citation_map_data_path = "../local_bibtex_info/bibtex_info_1202.jsonl"
    citation_map_data = load_citation_map_data(citation_map_data_path)
    d = encoded_corpus.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(encoded_corpus)
    index.add(encoded_corpus)
    print("index building finished")
    citations_data = []

    app.queue()  # 启用整个应用的队列功能
    app.launch(share=True)





