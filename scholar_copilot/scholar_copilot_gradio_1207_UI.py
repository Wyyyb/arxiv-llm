from datetime import datetime
import tempfile
from scholar_copilot_model_1206 import *
import torch
import faiss
import time
temp_dir = "./gradio_tmp"
os.makedirs(temp_dir, exist_ok=True)
# 设置环境变量
os.environ['GRADIO_TEMP_DIR'] = temp_dir
tempfile.tempdir = temp_dir
import gradio as gr


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
    retrieved_k_results = retrieve_reference(index, lookup_indices, cite_rep, top_k=10)
    searched_citations = []
    for each in retrieved_k_results:
        curr_index, distance = each
        print("index", curr_index)
        if curr_index not in meta_data:
            print("index not found in meta_data", curr_index)
            continue
        paper_id = meta_data[curr_index]["paper_id"]
        print("paper_id", paper_id)
        citation_info = citation_map_data[paper_id]
        print("generate_citation citation_info", citation_info)
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
        retrieved_k_results = retrieve_reference(index, lookup_indices, cite_start_hidden_state, top_k=1)
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
        retrieved_k_results = retrieve_reference(index, lookup_indices, cite_start_hidden_state,
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


# def search_and_show_citations(input_text):
#     global citations_data
#     curr_citations_data = generate_citation(input_text)
#     citations_data += curr_citations_data
#     choices = []
#     for cit in curr_citations_data:
#         paper_id = cit["id"]
#         item = cit["citation_key"] + ": " + cit["title"] + f" (https://arxiv.org/abs/{paper_id})"
#         choices.append(item)
#     return {
#         citation_box: gr.Group(visible=True),
#         citation_checkboxes: gr.CheckboxGroup(choices=choices, value=[])
#     }


def search_and_show_citations(input_text):
    global citations_data, curr_search_candidates
    curr_citations_data = generate_citation(input_text)
    curr_search_candidates = curr_citations_data
    choices = []
    for cit in curr_citations_data:
        paper_id = cit["id"]
        # 使用HTML格式创建带超链接的文本
        item = f'{cit["citation_key"]}: {cit["title"]} (https://arxiv.org/abs/{paper_id})'
        choices.append(item)
    return {
        citation_box: gr.Group(visible=True),
        citation_checkboxes: gr.CheckboxGroup(
            choices=choices,
            value=[],
        )
    }


def insert_selected_citations(text, selected_citations):
    """插入选中的引用并追踪"""
    global citations_data, curr_search_candidates

    if not selected_citations:
        return text

    selected_citations = [each.split(": ")[0] for each in selected_citations]
    citations = ", ".join(selected_citations)
    new_text = text + " \\cite{" + citations + "}"
    for each_candidate in curr_search_candidates:
        if each_candidate["citation_key"] in selected_citations:
            citations_data.append(each_candidate)
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
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file.write(header + content)
        temp_file_path = temp_file.name

    return temp_file_path


def clear_cache():
    global citations_data
    citations_data = []
    return "", []


with gr.Blocks(css="""
    :root {
        --color-1: #B2A59B;
        --color-2: #DED0B6; 
        --color-3: #FFFFFF;
        --color-4: #607274;
    }

    .container {
        max-width: 1200px;
        margin: auto;
        padding: 20px;
        background-color: var(--color-3);
    }
    .header {
        text-align: center;
        margin-bottom: 40px;
        background: linear-gradient(135deg, var(--color-1), var(--color-2));
        padding: 30px;
        border-radius: 15px;
        color: var(--color-4);
    }
    .logos {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        margin: 20px 0;
    }
    .logo {
        width: 80px;
        height: 80px;
        background: white;
        border-radius: 50%;
        padding: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .logo:hover {
        transform: scale(1.05);
    }
    .intro-section {
        background: #FAEED1;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-list {
        background: var(--color-2);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .main-editor {
        background: #FAEED1;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .button-row {
        display: flex;
        gap: 10px;
        margin-top: 15px;
        flex-wrap: wrap;
    }
    .button-row button {
        flex: 1;
        min-width: 200px;
        background: var(--color-1);
        border: none;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .button-row button:hover {
        background: var(--color-4);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .citation-section {
        background: #FAEED1;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .citation-section button {
        background: var(--color-2);
        border: none;
        color: var(--color-4);
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .citation-section button:hover {
        background: var(--color-1);
        color: white;
    }
    .textbox textarea {
        border: 2px solid var(--color-2);
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        transition: border-color 0.3s ease;
    }
    .textbox textarea:focus {
        border-color: var(--color-1);
        outline: none;
    }
    .checkbox-group {
        background: var(--color-3);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
""") as app:
    with gr.Column(elem_classes="container"):
        # Header section with logos
        with gr.Column(elem_classes="header"):
            gr.Markdown("""
                <div style='text-align: center;'>
                    <h1 style='font-size: 2.5em; margin-bottom: 10px;'>Scholar Copilot</h1>
                    <h3 style='font-weight: normal;'>Your Academic Writing Assistant</h3>
                </div>
            """)

            # Logos section
            with gr.Row(elem_classes="logos"):
                # App logo (using ASCII art as placeholder)
                gr.Markdown("""
                    <div class='logo'>
                        <pre style='font-size: 24px; color: #2193b0;'>
                        SC
                        </pre>
                    </div>
                """)
                # Organization logo (using ASCII art as placeholder)
                gr.Markdown("""
                    <div class='logo'>
                        <pre style='font-size: 24px; color: #2193b0;'>
                        ORG
                        </pre>
                    </div>
                """)

        # Introduction section
        with gr.Column(elem_classes="intro-section"):
            gr.Markdown("""
                Scholar Copilot improves the academic writing process by seamlessly integrating automatic text completion and intelligent citation suggestions into a cohesive, human-in-the-loop AI-driven pipeline. Designed to enhance productivity and creativity, it provides researchers with high-quality text generation and precise citation recommendations powered by iterative and context-aware Retrieval-Augmented Generation (RAG).

                The current version of Scholar Copilot leverages a state-of-the-art 7-billion-parameter language model (LLM) trained on the complete Arxiv full paper corpus. This unified model for retrieval and generation is adept at making context-sensitive decisions about when to cite, what to cite, and how to generate coherent content based on reference papers.
            """)

            with gr.Column(elem_classes="feature-list"):
                gr.Markdown("""
                    ### 🚀 Core Features:

                    * 📝 **Next-3-Sentence Suggestions**: Facilitates writing by predicting the next sentences with automatic retrieval and citation of relevant reference papers.
                    * 📚 **Citation Suggestions on Demand**: Provides precise, contextually appropriate paper citations whenever needed.
                    * ✨ **Full Section Auto-Completion**: Assists in brainstorming and drafting comprehensive paper content and structure.
                """)

        # Main editor section
        with gr.Column(elem_classes="main-editor"):
            text_input = gr.Textbox(
                lines=30,
                label="Write your paper here",
                placeholder="Start writing your academic paper...",
            )

            with gr.Row(elem_classes="button-row"):
                complete_btn = gr.Button("🔄 Complete 3 sentences", size="lg")
                generate_btn = gr.Button("✨ Generate to the end", size="lg")
                citation_btn = gr.Button("📚 Insert citation", size="lg")
                clear_btn = gr.Button("🗑️ Clear All", size="lg")

        # Citation section
        with gr.Column(elem_classes="citation-section"):
            citation_box = gr.Group(visible=True)
            with citation_box:
                gr.Markdown("### 📚 Citation Suggestions")
                citation_checkboxes = gr.CheckboxGroup(
                    choices=[],
                    label="Select citations to insert",
                    interactive=True
                )
                insert_citation_btn = gr.Button("📎 Insert selected citations", size="lg")

        with gr.Row():
            download_history_btn = gr.Button("📥 Download Citation History", size="lg")
            copy_status = gr.Textbox(
                value="",
                label="",
                interactive=False,
                show_label=False
            )

        # Event handlers
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
    # model_path = "/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-2000/"
    # model_path = "/data/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-2000/"
    # model_path = "../model_output/v1127_multi_cite/checkpoint-2000/"
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # model, tokenizer = load_model(model_path, device)
    # embedded_corpus_path = "../embedded_corpus/1129_shards/"
    # # encoded_corpus, lookup_indices = load_corpus_base(embedded_corpus_path)
    # meta_data = load_meta_data()
    # citation_map_data_path = "../local_bibtex_info/bibtex_info_1202.jsonl"
    # citation_map_data = load_citation_map_data(citation_map_data_path)
    # # index_dir = "/data/xueguang/scholar-hnsw-single"
    # index_dir = "../embedded_corpus/scholar-hnsw-1207/"
    # index, lookup_indices = load_faiss_index(index_dir)
    # print("index building finished")
    # citations_data = []
    # curr_search_candidates = []
    model_path = "../model_output/v1127_multi_cite/checkpoint-2000/"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model, tokenizer = None, None
    embedded_corpus_path = "../embedded_corpus/1129_shards/"
    # encoded_corpus, lookup_indices = load_corpus_base(embedded_corpus_path)
    meta_data = None
    citation_map_data_path = "../local_bibtex_info/bibtex_info_1202.jsonl"
    citation_map_data = None
    # index_dir = "/data/xueguang/scholar-hnsw-single"
    index_dir = "../embedded_corpus/scholar-hnsw-1207/"
    index, lookup_indices = None, None
    citations_data = []
    curr_search_candidates = []

    app.queue()  # 启用整个应用的队列功能
    app.launch(share=True)





