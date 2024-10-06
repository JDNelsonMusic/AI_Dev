from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Device configuration
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load tokenizer and model
model_path = "./"
tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

# Test generation
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=200)
generated_text = tokenizer.decode(outputs[1], skip_special_tokens=True)

print("Generated text:")
print(generated_text)
