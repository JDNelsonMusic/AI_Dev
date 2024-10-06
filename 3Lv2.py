from flask import Flask, request, render_template
import subprocess
import threading

app = Flask(__name__)

def get_response(model_name, prompt, responses):
    try:
        process = subprocess.Popen(
            ['ollama', 'run', model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate(input=prompt)
        if error:
            responses[model_name] = f"Error: {error}"
        else:
            responses[model_name] = output
    except FileNotFoundError as e:
        responses[model_name] = f"Error: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    responses = {}
    if request.method == 'POST':
        prompt = request.form['prompt']
        threads = []
        models = ['llama-3.2', 'solar-pro', 'mistral-nemo']  # Update model names as needed

        for model_name in models:
            thread = threading.Thread(target=get_response, args=(model_name, prompt, responses))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    return render_template('3L_GUIv2.html', responses=responses)

if __name__ == '__main__':
    # Start the Ollama server
    subprocess.Popen(['ollama', 'serve'])
    app.run(debug=True)
