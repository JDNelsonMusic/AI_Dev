import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def load_falcon_model(device):
    print("Loading Falcon 7B model...")
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    return tokenizer, model

def generate_falcon_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(prompt):].strip()
    return response

def generate_llama_response(prompt):
    try:
        # Use subprocess to run the ollama command
        command = ['ollama', 'run', 'llama3']
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Send the prompt to the model
        out, err = process.communicate(input=prompt)
        if err:
            print(f"Error with Ollama: {err}")
            return None
        # Process the output to get the response
        response = out.strip()
        return response
    except Exception as e:
        print(f"Error running Ollama: {e}")
        return None

def main():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Load Falcon model
    falcon_tokenizer, falcon_model = load_falcon_model(device)

    # Initialize conversation
    conversation = []
    num_turns = int(input("Enter the number of turns for the conversation: "))
    topic = input("Enter the topic for the models to discuss: ")

    print("\nStarting conversation...\n")

    # Initial prompt
    initial_prompt = f"The following is a conversation between two AI models discussing '{topic}'.\n\n"
    conversation.append(initial_prompt)

    for turn in range(num_turns):
        if turn % 2 == 0:
            # Falcon's turn
            current_prompt = "".join(conversation)
            print("Falcon is generating a response...")
            falcon_response = generate_falcon_response(falcon_model, falcon_tokenizer, current_prompt)
            print(f"\nFalcon: {falcon_response}\n")
            conversation.append(f"Falcon: {falcon_response}\n")
        else:
            # Llama's turn
            current_prompt = "".join(conversation)
            print("Llama is generating a response...")
            llama_response = generate_llama_response(current_prompt)
            if llama_response:
                print(f"\nLlama: {llama_response}\n")
                conversation.append(f"Llama: {llama_response}\n")
            else:
                print("Llama did not provide a response.")
                break

        # Wait before next turn (optional)
        time.sleep(1)

    print("Conversation ended.")

if __name__ == "__main__":
    main()