import gradio as gr

# This define logic (greet method) and UI in the same file, probably is better in separated files for complex UI

# LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC 
# Define a simple function that will be the core of your Gradio app
def greet(name):
    return f"Hello, {name}!"



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


