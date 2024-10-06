from flask import Flask, request
import subprocess
import threading

app = Flask(__name__)

# Define functions for each model
def llama3():
    process = subprocess.Popen(["olm", "run", "llama.3.2"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return process

def solar_pro():
    process = subprocess.Popen(["olm", "run", "solar-pro"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return process

def mistral_nemo():
    process = subprocess.Popen(["olm", "run", "mistral-nemo"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return process

# Launch models in separate threads upon app initialization
llama_thread = threading.Thread(target=llama3, args=[])
solar_pro_thread = threading.Thread(target=solar_pro, args=[])
mistral_nemo_thread = threading.Thread(target=mistral_nemo, args=[])

llama_thread.start()
solar_pro_thread.start()
mistral_nemo_thread.start()

# Define routes for each model
@app.route("/submit", methods=["POST"])
def submit():
    prompt = request.form["prompt"]

    # Pass the prompt to each model and return responses in separate threads
    llama_response = threading.Thread(target=lambda: handle_response("llama", prompt))
    solar_pro_response = threading.Thread(target=lambda: handle_response("solar-pro", prompt))
    mistral_nemo_response = threading.Thread(target=lambda: handle_response("mistral-nemo", prompt))

    llama_thread.join()
    solar_pro_thread.join()
    mistral_nemo_thread.join()

def handle_response(model, prompt):
    if model == "llama":
        response = llama3().communicate(prompt.encode())[0].decode()
    elif model == "solar-pro":
        response = solar_pro().communicate(prompt.encode())[0].decode()
    elif model == "mistral-nemo":
        response = mistral_nemo().communicate(prompt.encode())[0].decode()
    else:
        raise ValueError("Invalid model")

    return response
