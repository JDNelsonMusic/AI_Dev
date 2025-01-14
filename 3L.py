from flask import Flask, request, render_template
import subprocess
import threading
import requests
import time

app = Flask(__name__)

def load_models():
    # Start the Ollama server
    cmd = ['ollama', 'serve']
    subprocess.Popen(cmd)
    # Wait for the server to start
    time.sleep(2)  # Adjust the sleep time if necessary

@app.before_first_request
def startup():
    # Load the models upon launch
    threading.Thread(target=load_models).start()

@app.route('/', methods=['GET', 'POST'])
def index():
    responses = {}
    if request.method == 'POST':
        prompt = request.form['prompt']
        threads = []
        models = ['llama3.2', 'solar-pro', 'mistral-nemo']

        def get_response(model_name):
            url = 'http://localhost:11434/generate'
            data = {
                'model': model_name,
                'prompt': prompt
            }
            response = requests.post(url, json=data)
            responses[model_name] = response.text

        # Send prompt to each model in its own thread
        for model_name in models:
            thread = threading.Thread(target=get_response, args=(model_name,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

    return render_template('3L_GUI.html', responses=responses)

if __name__ == '__main__':
    app.run(debug=True)
