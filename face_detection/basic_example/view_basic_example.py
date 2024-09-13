import gradio as gr
from .logic_basic_example import extract_faces

# UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI  UI

# Create the Gradio interface
# interface = gr.Interface(fn=greet, inputs="text", outputs="text").api(name="/greet")
# btn = gr.Button("Execute")
# btn.click(fn=greet, inputs=inp, outputs=out)


num_images = 10
gr.Markdown("# Face Extractor from Video")
with gr.Row():
    # First row: Video upload
    video_input = gr.Video(label="Upload Video")
with gr.Row():
    # Second row: Display images (as placeholders)
    images_output = [gr.Image(label=f"Image {i+1}") for i in range(num_images)]

# Third row: Button to trigger face extraction
extract_button = gr.Button("Extract Faces")

# Setting up interaction
extract_button.click(fn=extract_faces, inputs=video_input, outputs=images_output)


