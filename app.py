from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch
from werkzeug.utils import secure_filename
import os
from PIL import Image
import time

app = Flask('JDN_LocalModel_GUI')
app.secret_key = 'KEEx'  # Replace with a secure secret key

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model path
MODEL_PATH = '/Users/JDNelson/LlamaModels/Llama-3.2-11B-Vision-Instruct'

# Load the processor and model
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16,
)

# Configure the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")

model = model.to(device)

# Store conversation history
conversation_history = []

# To track if the context window is locked
context_window_locked = False

def generate_response(text_prompt, image_path=None, max_length=512):
    if image_path:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=text_prompt, images=[image], return_tensors="pt").to(device)
    else:
        inputs = processor(text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=max_length,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    global conversation_history, context_window_locked
    if request.method == 'POST':
        start_time = time.time()
        # Get form data
        message = request.form.get('message')
        context_window = request.form.get('context_window', type=int)
        num_tokens = request.form.get('num_tokens', type=int)
        image_file = request.files.get('image')

        # Lock context window after first message
        if not context_window_locked:
            if context_window:
                max_length = context_window
            else:
                max_length = 512
            context_window_locked = True
        else:
            max_length = 512  # Default value once the context is locked

        # Handle image upload
        image_path = None
        if image_file and image_file.filename != '':
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)

        # Generate response
        try:
            response_text = generate_response(
                message,
                image_path=image_path,
                max_length=num_tokens if num_tokens else max_length,
            )
        except Exception as e:
            response_text = f"An error occurred during generation: {e}"

        # Add to conversation history
        conversation_history.append({'message': message, 'response': response_text})

        end_time = time.time()
        elapsed_time = end_time - start_time

        return render_template(
            'main_page.html',
            response=response_text,
            message=message,
            image_path=image_path,
            elapsed_time=f"{elapsed_time:.2f} seconds",
            conversation_history=conversation_history
        )
    else:
        return render_template('main_page.html', conversation_history=conversation_history)

@app.route('/clear', methods=['POST'])
def clear():
    global conversation_history, context_window_locked
    conversation_history = []
    context_window_locked = False
    return render_template('main_page.html', conversation_history=conversation_history)

if __name__ == '__main__':
    app.run(debug=True)
