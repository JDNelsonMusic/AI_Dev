import torch
from transformers import AutoProcessor, AutoModelForPreTraining
from PIL import Image

# Specify the model name
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Load the processor and model
processor = AutoProcessor.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForPreTraining.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_auth_token=True
)

# Configure the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")

model = model.to(device)

def generate_response(text_prompt, image_path=None):
    if image_path:
        image = Image.open(image_path).convert("RGB")
        # Include image in a list
        inputs = processor(text=text_prompt, images=[image], return_tensors="pt")
    else:
        inputs = processor(text=text_prompt, return_tensors="pt")

    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=100,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("LLaMA 3.2 11B Vision-Text Model - Interactive Mode")
    print("Type 'exit' to stop the program.")

    while True:
        text_input = input("\nEnter your text prompt: ")
        if text_input.lower() == 'exit':
            print("Goodbye!")
            break

        use_image = input("Do you want to include an image? (yes/no): ").lower()
        image_path = None

        if use_image == 'yes':
            image_path = input("Enter the path to your image: ")

        response = generate_response(text_input, image_path)
        print("\nLLaMA Response:", response)

if __name__ == "__main__":
    main()