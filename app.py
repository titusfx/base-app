import gradio as gr

# Define a simple function that will be the core of your Gradio app
def greet(name):
    return f"Hello, {name}!"

# Create the Gradio interface
# interface = gr.Interface(fn=greet, inputs="text", outputs="text").api(name="/greet")
# btn = gr.Button("Execute")
# btn.click(fn=greet, inputs=inp, outputs=out)

with gr.Blocks() as app:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=greet, inputs=[inp], outputs=[out])

    
# Launch the Gradio app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)

