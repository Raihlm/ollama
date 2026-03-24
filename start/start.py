import requests
import json

url = "http://localhost:11434/api/generate"

data = {
    "model": "llama3.2",
    "prompt": "What is the capital of France?",
}

response = requests.post(
    url, json=data, stream=True
)

listModel = ollama.list()

chat = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

# get a content
print(chat["messages"][0]["content"])