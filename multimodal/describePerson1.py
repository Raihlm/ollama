"""
llava is a multimodal model that can understand and generate text based on
both textual and visual inputs(binary images). in this example, we use the model to describe a person
in an image. we provide a prompt asking the model to describe the person's appearance, clothing,
and accessrories, and we also provide the image of the person. we specify that we want to run the model on CPU to avoid CUDA OOM errors. 
the model processes the image and the prompt, and generates a detailed description.
"""

import ollama



prompt = (
    "Describe the person in this image with detailed information about their "
    "appearance, clothing, and any accessories they are wearing."
)

response = ollama.generate(
    model="llava",                  # multimodal model
    prompt=prompt,
    images=["../img/person1.png"],   # path (or base64) to the image
    options={"num_gpu": 0},         # run on CPU to avoid CUDA OOM
)

print(response.text)
