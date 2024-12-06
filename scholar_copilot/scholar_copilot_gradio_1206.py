import gradio as gr
from datetime import datetime
import tempfile
from scholar_copilot_model_1206 import *
import torch
import faiss


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
    print("split_yield_list input_text", input_text)
    print("split_yield_list prefix_length", prefix_length)
    prefix_text = input_text[:prefix_length]
    text_list = input_text[prefix_length:].split(" ")
    return prefix_text, text_list


def stream_complete_3_sentence(text, progress=gr.Progress()):
    current_text = text
    current_text = preprocess_input_text(current_text)
    input_text_length = len(current_text)
    curr_prefix_length = len(current_text)
    current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
    reference_id_list = []
    curr_yield_text, yield_list = split_yield_list(current_text, curr_prefix_length)
    for each in yield_list:
        curr_yield_text += " " + each
        yield curr_yield_text
    curr_prefix_length = len(current_text)
    while cite_start_hidden_state is not None:
        enough_sentences, res_text = cut_after_third_sentence(current_text[input_text_length:], 3)
        if enough_sentences:
            current_text = res_text
            break
        retrieved_k_results = retrieve_reference(index, encoded_corpus, lookup_indices, cite_start_hidden_state, top_k=1)
        reference, curr_index = llm_rerank(retrieved_k_results, meta_data)
        reference_id_list.append(curr_index)
        current_text = current_text + reference
        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
        display_text = replace_citations(current_text, reference_id_list, citation_map_data)
        curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
        for each in yield_list:
            curr_yield_text += " " + each
            yield curr_yield_text
    display_text = replace_citations(current_text, reference_id_list, citation_map_data)
    display_text = post_process_output_text(display_text, reference_id_list, citation_map_data)
    yield display_text


def stream_generate(text, progress=gr.Progress()):
    current_text = text
    current_text = preprocess_input_text(current_text)
    curr_prefix_length = len(current_text)
    current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
    reference_id_list = []
    curr_yield_text, yield_list = split_yield_list(current_text, curr_prefix_length)
    for each in yield_list:
        curr_yield_text += " " + each
        yield curr_yield_text
    curr_prefix_length = len(current_text)
    while cite_start_hidden_state is not None:
        retrieved_k_results = retrieve_reference(index, encoded_corpus, lookup_indices, cite_start_hidden_state, top_k=1)
        reference, curr_index = llm_rerank(retrieved_k_results, meta_data)
        reference_id_list.append(curr_index)
        current_text = current_text + reference
        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
        display_text = replace_citations(current_text, reference_id_list, citation_map_data)
        curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
        for each in yield_list:
            curr_yield_text += " " + each
            yield curr_yield_text
    display_text = replace_citations(current_text, reference_id_list, citation_map_data)
    display_text = post_process_output_text(display_text, reference_id_list, citation_map_data)
    yield display_text


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


with gr.Blocks() as app:
    gr.Markdown("# Scholar Copilot - Your Academic Writing Assistant")

    with gr.Row():
        # 主编辑区
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=30,
                label="Write your paper here",
                placeholder="Writing your academic paper..."
            )

            with gr.Row():
                complete_btn = gr.Button("Complete 3 sentences")
                generate_btn = gr.Button("Complete to the end")
                citation_btn = gr.Button("Insert citation")

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
        download_history_btn = gr.Button("Download Citation History")
        copy_status = gr.Textbox(
            value="",
            label="",
            interactive=False,
            show_label=False
        )

    # 修改后的事件处理
    complete_btn.click(
        fn=stream_complete_3_sentence,
        inputs=[text_input],
        outputs=[text_input],
        queue=True  # 启用队列以支持流式输出
    )

    generate_btn.click(
        fn=stream_generate,
        inputs=[text_input],
        outputs=[text_input],
        queue=True  # 启用队列以支持流式输出
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





