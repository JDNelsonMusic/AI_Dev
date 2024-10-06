import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load the model and tokenizer
    model_name = "apple/dclm-7b"  # Replace with the correct model name if different
    print("Loading model. This may take a few minutes...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

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
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break

        # Encode the user input and add end-of-sentence token
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the output and remove the user input from the response
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text[len(user_input):].strip()

        print(f"Model: {response}")

if __name__ == "__main__":
    main()