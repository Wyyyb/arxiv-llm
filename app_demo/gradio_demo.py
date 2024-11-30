import gradio as gr
import time


def get_completion(prompt, max_new_tokens=32):
    """获取代码补全"""
    # inputs = tokenizer(prompt, return_tensors="pt")
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=max_new_tokens,
    #         temperature=0.2,
    #         do_sample=True,
    #         pad_token_id=tokenizer.eos_token_id
    #     )
    # completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    completion = "it's a good day"
    return completion


def process_stream(text):
    """处理文本流,返回补全建议"""
    # 防止过于频繁的API调用
    time.sleep(0.1)

    # 获取最后一个不完整的行
    lines = text.split('\n')
    current_line = lines[-1] if lines else ""

    # 如果当前行太短,不进行补全
    if len(current_line.strip()) < 3:
        return text

    # 获取补全建议
    try:
        completion = get_completion(text)
        return text + completion
    except:
        return text


# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# Real-time Code Completion Demo")

    editor = gr.Code(
        label="Code Editor",
        language="python",
        interactive=True,
        lines=10
    )

    # 每次文本变化时触发补全
    editor.change(
        fn=process_stream,
        inputs=editor,
        outputs=editor,
        show_progress=False
    )

demo.queue().launch()
