# app.py
from flask import Flask, render_template, request, redirect, url_for
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch
from werkzeug.utils import secure_filename
import os
from PIL import Image
import time

app = Flask('JDN_LocalModel_GUI')
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model paths
MODEL_PATHS = {
    'Llama3.2-11B-Vision-Instruct': '/Users/JDNelson/LlamaModels/Llama-3.2-11B-Vision-Instruct',
    'Llama-3.2-3B-Instruct': '/Users/JDNelson/LlamaModels/Llama-3.2-3B-Instruct',
    'Falcon8B': '/Users/JDNelson/LlamaModels/Falcon8B',  # Adjust this path
}

# Initialize models dictionary
models = {}
processors = {}

# Load the memory model (using the 3B model)
MEMORY_MODEL_NAME = 'Llama-3.2-3B-Instruct'
MEMORY_MODEL_PATH = MODEL_PATHS[MEMORY_MODEL_NAME]
memory_processor = AutoProcessor.from_pretrained(MEMORY_MODEL_PATH, local_files_only=True)
memory_model = AutoModelForCausalLM.from_pretrained(
    MEMORY_MODEL_PATH,
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

memory_model = memory_model.to(device)

# Memory bank initialization
memory_bank = "This is the initial memory bank."

# Store conversation history
conversation_history = []

# To track if the context window is locked
context_window_locked = False

def generate_memory_addendum(message):
    # Use the memory model to generate an addendum based on the memory bank and the user's message
    prompt = f"Memory Bank:\n{memory_bank}\nUser Message:\n{message}\nAddendum:"
    inputs = memory_processor(text=prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = memory_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=50,
            eos_token_id=memory_processor.tokenizer.eos_token_id,
            pad_token_id=memory_processor.tokenizer.pad_token_id,
        )
    generated_text = memory_processor.decode(outputs[0], skip_special_tokens=True)
    # Extract the addendum
    addendum = generated_text.split("Addendum:")[-1].strip()
    return addendum

def update_memory_bank(message, response):
    # Update the memory bank based on the latest exchange
    global memory_bank
    memory_bank += f"\nUser: {message}\nModel: {response}"

def generate_response(text_prompt, image_path=None, max_length=512, model_name='Llama3.2-11B-Vision-Instruct'):
    # Get the selected model and processor
    if model_name not in models:
        # Load the model if not already loaded
        model_path = MODEL_PATHS.get(model_name)
        if model_path is None:
            return f"Model {model_name} not found."
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to(device)
        processors[model_name] = processor
        models[model_name] = model
    else:
        processor = processors[model_name]
        model = models[model_name]

    # Generate the memory addendum
    memory_addendum = generate_memory_addendum(text_prompt)

    # Combine the user's message with the memory addendum
    combined_prompt = f"{text_prompt}\n{memory_addendum}"

    # Prepare inputs
    try:
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=combined_prompt, images=image, return_tensors="pt").to(device)
        else:
            inputs = processor(text=combined_prompt, return_tensors="pt").to(device)
    except Exception as e:
        print(f"Error processing inputs: {e}")
        inputs = processor(text=combined_prompt, return_tensors="pt").to(device)

    # Generate response
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
    global conversation_history, context_window_locked, memory_bank
    if request.method == 'POST':
        start_time = time.time()
        # Get form data
        message = request.form.get('message')
        context_window = request.form.get('context_window', type=int)
        num_tokens = request.form.get('num_tokens', type=int)
        image_file = request.files.get('image')
        model_name = request.form.get('model_name', 'Llama3.2-11B-Vision-Instruct')

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

        # Add the user's message to the conversation history
        conversation_history.append({'sender': 'User', 'message': message})

        # Generate response
        try:
            response_text = generate_response(
                message,
                image_path=image_path,
                max_length=num_tokens if num_tokens else max_length,
                model_name=model_name,
            )
        except Exception as e:
            response_text = f"An error occurred during generation: {e}"
            print(e)

        # Add the model's response to the conversation history
        conversation_history.append({'sender': 'Model', 'message': response_text})

        # Update the memory bank
        update_memory_bank(message, response_text)

        end_time = time.time()
        elapsed_time = end_time - start_time

        return render_template(
            'main_page_thread.html',
            elapsed_time=f"{elapsed_time:.2f} seconds",
            conversation_history=conversation_history,
            models_available=list(MODEL_PATHS.keys()),
            selected_model=model_name,
            context_window_locked=context_window_locked,
        )
    else:
        # GET request
        return render_template(
            'main_page_thread.html',
            conversation_history=conversation_history,
            models_available=list(MODEL_PATHS.keys()),
            selected_model='Llama3.2-11B-Vision-Instruct',
            context_window_locked=context_window_locked,
        )

@app.route('/clear', methods=['POST'])
def clear():
    global conversation_history, context_window_locked, memory_bank
    conversation_history = []
    context_window_locked = False
    memory_bank = "This is the initial memory bank."
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
