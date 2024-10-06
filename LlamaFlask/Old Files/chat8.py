import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load the model and tokenizer
    model_name = "tiiuae/falcon-7b-instruct"
    print("Loading model. This may take a few minutes...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check for MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model.to(device)
        print("Using MPS backend for Apple Silicon.")
    else:
        device = torch.device("cpu")
        print("MPS backend not available. Using CPU.")

    print("\nModel loaded successfully. You can start chatting now. Type 'exit' to quit.")

    # Chat loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chat. Goodbye!")
                break

            # Prepare the input
            input_text = f"{user_input}"

            # Tokenize and generate response
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode and print the response
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response = output_text[len(user_input):].strip()

            print(f"Model: {response}")

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    main()