# app.py
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from werkzeug.utils import secure_filename
import os
from PIL import Image  # For image handling if needed

app = Flask('JDN_LocalModel_GUI')
app.secret_key = 'KEEx'  # Replace with a secure secret key

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the tokenizer and model
MODEL_PATH = '/Users/JDNelson/LlamaModels/Llama-3.2-11B-Vision-Instruct'

# Use AutoTokenizer and AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        message = request.form.get('message')
        context_window = request.form.get('context_window', type=int)
        num_tokens = request.form.get('num_tokens', type=int)
        image_file = request.files.get('image')

        # Handle image upload
        image_path = None
        if image_file and image_file.filename != '':
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)
            # For now, we'll not process the image

        # Prepare input for the model
        input_ids = tokenizer.encode(message, return_tensors='pt').to(device)

        # Adjust context window and num_tokens if specified
        max_length = context_window if context_window else 512
        num_tokens = num_tokens if num_tokens else 50

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return render_template('main_page.html', response=output_text, message=message, image_path=image_path)
    else:
        return render_template('main_page.html')

if __name__ == '__main__':
    app.run(debug=True)
