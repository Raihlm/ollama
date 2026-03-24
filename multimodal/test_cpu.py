import ollama

try:
    r = ollama.generate(model='llava', prompt='Hello', options={'num_gpu':0})
    print('success', r)
except Exception as e:
    print('error', e)
