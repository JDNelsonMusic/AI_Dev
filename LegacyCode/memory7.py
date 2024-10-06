# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from werkzeug.utils import secure_filename
import os
import time

app = Flask('JDN_LocalModel_GUI')
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("conversations", exist_ok=True)

# Model paths
MODEL_INFO = {
    'Llama3.2-11B-Vision-Instruct': {
        'path': '/Users/JDNelson/LlamaModels/Llama-3.2-11B-Vision-Instruct',
        'type': 'vision',  # Indicates vision model
    },
    'Llama-3.2-3B-Instruct': {
        'path': '/Users/JDNelson/LlamaModels/Llama-3.2-3B-Instruct',
        'type': 'text',  # Indicates text model
    },
    'Falcon8B': {
        'path': '/Users/JDNelson/LlamaModels/Falcon8B',
        'type': 'text',  # Adjust this if Falcon8B is a vision model
    },
}

# Initialize pipelines dictionary
pipelines = {}

# Configure the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Memory bank initialization
memory_bank = "This is the initial memory bank."

# Store conversation history
conversation_history = []

# To track if the context window is locked
context_window_locked = False

def generate_memory_addendum(message):
    # Use the memory model to generate an addendum based on the memory bank and the user's message
    MEMORY_MODEL_NAME = 'Llama-3.2-3B-Instruct'
    MEMORY_MODEL_PATH = MODEL_INFO[MEMORY_MODEL_NAME]['path']
    if MEMORY_MODEL_NAME not in pipelines:
        # Load the model and tokenizer for the memory model
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MEMORY_MODEL_PATH,
                local_files_only=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MEMORY_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            pipelines[MEMORY_MODEL_NAME] = pipe
            print(f"Memory model '{MEMORY_MODEL_NAME}' loaded successfully.")
        except Exception as e:
            print(f"Error loading memory model '{MEMORY_MODEL_NAME}': {e}")
            return ""
    else:
        pipe = pipelines[MEMORY_MODEL_NAME]

    prompt = f"Memory Bank:\n{memory_bank}\nUser Message:\n{message}\nAddendum:"
    outputs = pipe(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id,
    )
    generated_text = outputs[0]["generated_text"]
    # Extract the addendum
    addendum = generated_text.split("Addendum:")[-1].strip()
    return addendum

def update_memory_bank(message, response):
    # Update the memory bank based on the latest exchange
    global memory_bank
    memory_bank += f"\nUser: {message}\nAssistant: {response}"

def generate_response(message, image_path=None, max_length=512, model_name='Llama3.2-11B-Vision-Instruct', temperature=0.7, top_p=0.9):
    # Get the selected model
    model_info = MODEL_INFO.get(model_name)
    if model_info is None:
        return f"Model {model_name} not found."

    if model_name not in pipelines:
        model_path = model_info['path']
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            pipelines[model_name] = pipe
            print(f"Pipeline for model '{model_name}' initialized successfully.")
        except Exception as e:
            print(f"Error initializing pipeline for model '{model_name}': {e}")
            return f"Error initializing pipeline for model '{model_name}': {e}"
    else:
        pipe = pipelines[model_name]

    # Generate the memory addendum
    memory_addendum = generate_memory_addendum(message)

    # Combine the user's message with the memory addendum
    combined_prompt = f"{memory_bank}\nUser: {message}\nAssistant:"

    # Generate response
    try:
        outputs = pipe(
            combined_prompt,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
        response = outputs[0]["generated_text"].split("Assistant:")[-1].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {e}"

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
        model_name = request.form.get('model_name', 'Llama3.2-11B-Vision-Instruct')
        temperature = float(request.form.get('temperature', 0.7))
        top_p = float(request.form.get('top_p', 0.9))

        # Lock context window after first message
        if not context_window_locked:
            if context_window:
                max_length = context_window
            else:
                max_length = 512
            context_window_locked = True
        else:
            max_length = 512  # Default value once the context is locked

        # Add the user's message to the conversation history
        conversation_history.append({'sender': 'User', 'message': message})

        # Generate response
        try:
            response_text = generate_response(
                message,
                max_length=num_tokens if num_tokens else max_length,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            response_text = f"An error occurred during generation: {e}"
            print(e)

        # Add the assistant's response to the conversation history
        conversation_history.append({'sender': 'Assistant', 'message': response_text})

        # Update the memory bank
        update_memory_bank(message, response_text)

        end_time = time.time()
        elapsed_time = end_time - start_time

        return render_template(
            'main_page_thread.html',
            elapsed_time=f"{elapsed_time:.2f} seconds",
            conversation_history=conversation_history,
            models_available=list(MODEL_INFO.keys()),
            selected_model=model_name,
            context_window_locked=context_window_locked,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        # GET request
        return render_template(
            'main_page_thread.html',
            conversation_history=conversation_history,
            models_available=list(MODEL_INFO.keys()),
            selected_model='Llama3.2-11B-Vision-Instruct',
            context_window_locked=context_window_locked,
            temperature=0.7,
            top_p=0.9,
        )

@app.route('/clear', methods=['POST'])
def clear():
    global conversation_history, context_window_locked, memory_bank
    conversation_history = []
    context_window_locked = False
    memory_bank = "This is the initial memory bank."
    return redirect(url_for('index'))

@app.route('/save_conversation', methods=['POST'])
def save_conversation():
    global conversation_history
    # Save the conversation history to a file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"conversation_{timestamp}.txt"
    filepath = os.path.join("conversations", filename)
    with open(filepath, "w") as f:
        for entry in conversation_history:
            f.write(f"{entry['sender']}: {entry['message']}\n")
    return jsonify({'message': f"Conversation saved as {filename}"})

if __name__ == '__main__':
    app.run(debug=True)