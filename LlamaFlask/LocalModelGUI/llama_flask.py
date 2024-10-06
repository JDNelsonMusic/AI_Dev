import torch
from transformers import AutoProcessor, AutoModelForPreTraining
from PIL import Image
from flask import Flask, request, render_template_string

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

def generate_response(text_prompt, image=None):
    if image:
        image = Image.open(image).convert("RGB")
        inputs = processor(text=text_prompt, images=[image], return_tensors="pt")
    else:
        inputs = processor(text=text_prompt, return_tensors="pt")

    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=1500,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

app = Flask(__name__)

# Define the HTML template
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>LLaMA 3.2 11B Vision-Text Chat</title>
</head>
<body>
    <h1>LLaMA 3.2 11B Vision-Text Chat</h1>
    <form action="/" method="post" enctype="multipart/form-data">
        <label for="prompt">Enter your text prompt:</label><br>
        <textarea id="prompt" name="prompt" rows="4" cols="50">{{ prompt }}</textarea><br><br>
        <label for="image">Select an image (optional):</label><br>
        <input type="file" id="image" name="image"><br><br>
        <input type="submit" value="Submit">
    </form>
    {% if response %}
    <h2>Response:</h2>
    <p>{{ response }}</p>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    response = ''
    prompt = ''
    if request.method == 'POST':
        prompt = request.form['prompt']
        image = request.files.get('image')

        # Include the image token if an image is attached
        if image and "<image>" not in prompt:
            prompt += " <image>"

        response = generate_response(prompt, image)
    return render_template_string(html_template, response=response, prompt=prompt)

if __name__ == '__main__':
    app.run(debug=True)
