from gradio_client import Client
from .config import PORT, HOST, SCHEMA


# Initialize the client with the URL of the Gradio service
# client = Client("http://127.0.0.1:7860/")
client = Client(f"{SCHEMA}{HOST}:{PORT}/")

# Uncomment the next line to see api doc
#client.view_api() # Shows the api info
# Client.predict() Usage Info
# ---------------------------
# Named API endpoints: 1

#  - predict(name, api_name="/greet") -> value_4
#     Parameters:
#      - [Textbox] name: str (required)  
#     Returns:
#      - [Textbox] value_4: str 

# Call the Gradio interface with the input data
result = client.predict(name="Alice",api_name="/greet")  # Replace "Alice" with the input you want to test

# Print the result returned by the Gradio service
print(result)


