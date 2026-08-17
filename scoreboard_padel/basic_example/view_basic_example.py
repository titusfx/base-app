import gradio as gr
from .logic_basic_example import find_scoreboard

# UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI

# Create the Gradio interface
# interface = gr.Interface(fn=greet, inputs="text", outputs="text").api(name="/greet")
# btn = gr.Button("Execute")
# btn.click(fn=greet, inputs=inp, outputs=out)

gr.Markdown("Start typing below and then click **Run** to see the output.")
with gr.Blocks():
    # Second row: Video input
    video_input = gr.Video(label="Video Input")
    output_text = gr.Textbox(label="Detected Scores")
    output_images = gr.Gallery(label="Detected Frames with Bounding Boxes")
    output = [
        output_text,
        # output_images,
    ]
    # Button to trigger the process
    btn = gr.Button("Run")
    btn.click(fn=find_scoreboard, inputs=video_input, outputs=output)
