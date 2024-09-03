import gradio as gr
from config import PORT

with gr.Blocks() as app:
    # Import the endpoint ( is equivalent to copy and paste the content of basic_example.py)
    from basic_example import *


# Launch the Gradio app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=PORT)

