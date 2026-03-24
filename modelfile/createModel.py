import ollama

modelfile = """
You are a great mentor and very wisdom. you will always give related real life advice to the question.
 you will always give a detailed answer. 
  you will always give a detailed but concise answer. you will always give a related real life advice to the question.
"""

ollama.create_model(model='my-life-mentor', modelfile=modelfile)

res = ollama.generate(model='my-life-mentor', prompt='How to be good at learning?')

print(res)


# ollama.detail_model('my-life-mentor')
# ollama.delete_model('my-life-mentor')