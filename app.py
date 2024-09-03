import gradio as gr

# Define a simple function that will be the core of your Gradio app
def greet(name):
    return f"Hello, {name}!"

# Create the Gradio interface
interface = gr.Interface(fn=greet, inputs="text", outputs="text")

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)

