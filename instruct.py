import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "where in all the world can you catch the most colorful fish?"},
]
outputs = pipe(
    messages,
    max_new_tokens=512,
)
print(outputs[0]["generated_text"][-1])