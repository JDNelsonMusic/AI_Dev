import tkinter as tk
from tkinter import filedialog, scrolledtext
import torch
from transformers import AutoProcessor, AutoModelForPreTraining
from PIL import Image

# Specify the model name
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Load the processor and model
processor = AutoProcessor.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForPreTraining.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_auth_token=True
)

# Configure the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")

model = model.to(device)

def generate_response(text_prompt, image_path=None):
    if image_path:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=text_prompt, images=[image], return_tensors="pt")
    else:
        inputs = processor(text=text_prompt, return_tensors="pt")

    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=100,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

# GUI Application
class LlamaApp:
    def __init__(self, master):
        self.master = master
        master.title("LLaMA 3.2 11B Vision-Text Chat")

        self.text_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=80, height=20)
        self.text_area.pack(padx=10, pady=10)

        self.entry_frame = tk.Frame(master)
        self.entry_frame.pack(padx=10, pady=5)

        self.entry = tk.Entry(self.entry_frame, width=60)
        self.entry.pack(side=tk.LEFT, padx=5)

        self.image_button = tk.Button(self.entry_frame, text="Attach Image", command=self.load_image)
        self.image_button.pack(side=tk.LEFT)

        self.send_button = tk.Button(self.entry_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=5)

        self.image_path = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename(title="Select an Image",
                                                     filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if self.image_path:
            self.text_area.insert(tk.END, "Image attached: {}\n".format(self.image_path))
            self.text_area.see(tk.END)

    def send_message(self):
        user_input = self.entry.get()
        if user_input.strip() == "":
            return

        # Include the image token if an image is attached
        if self.image_path and "<image>" not in user_input:
            user_input += " <image>"

        self.text_area.insert(tk.END, "You: {}\n".format(user_input))
        self.text_area.see(tk.END)
        self.entry.delete(0, tk.END)

        response = generate_response(user_input, self.image_path)
        self.text_area.insert(tk.END, "LLaMA: {}\n\n".format(response))
        self.text_area.see(tk.END)

        # Reset the image path after sending the message
        self.image_path = None

# Run the application
root = tk.Tk()
app = LlamaApp(root)
root.mainloop()