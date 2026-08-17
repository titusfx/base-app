import gradio as gr
from .logic_basic_example import greet

# UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI

# Create the Gradio interface
# interface = gr.Interface(fn=greet, inputs="text", outputs="text").api(name="/greet")
# btn = gr.Button("Execute")
# btn.click(fn=greet, inputs=inp, outputs=out)

gr.Markdown("Start typing below and then click **Run** to see the output.")
with gr.Row():
    inp = [gr.Textbox(placeholder="What is your name?")]
    out = [gr.Textbox()]
btn = gr.Button("Run")
btn.click(fn=greet, inputs=inp, outputs=out)


