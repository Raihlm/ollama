import ollama
import os

INPUT_PATH = './data/list.txt'
OUTPUT_PATH = './data/structured_list.txt'

if not os.path.exists(INPUT_PATH):
    print(f'{INPUT_PATH} does not exist')
    exit(1)

with open(INPUT_PATH, 'r') as f:
    items = f.read().strip()

prompt = f"Here is a list of items:\n{items}\n\nPlease structure this list in a more organized way."

try:
    response = ollama.generate(model='llama3.2', prompt=prompt)
    generated_text = response.get("response", "")
    print('Generated Text:', generated_text)
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write(generated_text.strip())

    print(f'Successfully structured the list and saved to {OUTPUT_PATH}')
except Exception as e:
    print('Error:', e)

    