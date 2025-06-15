import gradio as gr
import os
import shutil
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from gradio import update
import time  

# --- Config ---
NICHE_DATASETS = {
    "Finance": "datasets/finance.txt",
    "Legal": "datasets/legal.txt",
    "Medical": "datasets/medical.txt",
    "Education": "datasets/education.txt"
}
OUTPUT_DIR = "lora_adapter"

chat_tokenizer = None
chat_model = None

# --- Train Functions ---
def train_from_niche(niche, base_model):
    dataset_path = NICHE_DATASETS[niche]
    return run_lora_train(base_model, dataset_path)

def train_from_file(file_path, base_model):
    return run_lora_train(base_model, file_path.name)
def load_finetuned_model():
    time.sleep(3)
    return "✅ Chat interface ready! (Simulated)"


def run_lora_train(base_model, dataset_path):
    # Simulate training delay
    time.sleep(6)  # Wait for 6 seconds to mimic training time

    # Create dummy adapter directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Write dummy adapter files
    with open(os.path.join(OUTPUT_DIR, "adapter_config.json"), "w") as f:
        f.write('{"dummy": true}')
    
    with open(os.path.join(OUTPUT_DIR, "adapter_model.bin"), "w") as f:
        f.write("DUMMY MODEL WEIGHTS")

    # Create dummy zip file
    shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)

    return "✅ Fine-tuning complete!", OUTPUT_DIR + ".zip"
def chat_with_model(message, history):
    global chat_model, chat_tokenizer
    if chat_model is None or chat_tokenizer is None:
        return "⚠️ Model not loaded. Click 'Load Fine-Tuned Model' first."
    input_ids = chat_tokenizer.encode(message + chat_tokenizer.eos_token, return_tensors="pt")
    with torch.no_grad():
        output = chat_model.generate(input_ids, max_length=150, pad_token_id=chat_tokenizer.eos_token_id)
    response = chat_tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# --- UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🔧 LLM Tuner: On-Device Fine-Tuner for Niche Domains")

    with gr.Column(visible=True) as home_screen:
        gr.Markdown("### Welcome! Choose how you'd like to fine-tune:")
        b1 = gr.Button("Select Niche Domain Dataset")
        b2 = gr.Button("Upload Custom Dataset (.txt)")
        b3 = gr.Button("💬 Chat with Fine-Tuned Model")

    with gr.Column(visible=False) as niche_screen:
        gr.Markdown("### Select a Niche Domain and Base Model")
        niche_dropdown = gr.Dropdown(choices=list(NICHE_DATASETS.keys()), label="Choose Domain")
        model_dropdown_niche = gr.Dropdown(choices=["gpt2", "distilgpt2"], label="Choose Base Model")
        train_niche_button = gr.Button("Train on Selected Niche Dataset")
        output_niche = gr.Textbox(label="Training Status")
        download_button_niche = gr.File(label="Download LoRA Adapter", interactive=False)
        back_button1 = gr.Button("🔙 Back to Home")

    with gr.Column(visible=False) as upload_screen:
        gr.Markdown("### Upload Custom Dataset and Choose Base Model")
        file_upload = gr.File(label="Upload a .txt dataset file", file_types=[".txt"])
        model_dropdown_file = gr.Dropdown(choices=["gpt2", "distilgpt2"], label="Choose Base Model")
        train_file_button = gr.Button("Train on Uploaded Dataset")
        output_file = gr.Textbox(label="Training Status")
        download_button_file = gr.File(label="Download LoRA Adapter", interactive=False)
        back_button2 = gr.Button("🔙 Back to Home")

    with gr.Column(visible=False) as chat_screen:
        gr.Markdown("### 💬 Chat with Your Fine-Tuned Model")
        chatbot = gr.ChatInterface(fn=chat_with_model)
        load_chat_button = gr.Button("Load Fine-Tuned Model")
        chat_status = gr.Textbox(label="Status")
        back_button_chat = gr.Button("🔙 Back to Home")

    # --- Routing Functions ---
    def show_niche_screen(): return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    def show_upload_screen(): return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    def show_home_screen(): return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    def show_chat_screen(): return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    # --- Bind Buttons ---
    b1.click(fn=show_niche_screen, outputs=[home_screen, niche_screen, upload_screen, chat_screen])
    b2.click(fn=show_upload_screen, outputs=[home_screen, niche_screen, upload_screen, chat_screen])
    b3.click(fn=show_chat_screen, outputs=[home_screen, niche_screen, upload_screen, chat_screen])

    back_button1.click(fn=show_home_screen, outputs=[home_screen, niche_screen, upload_screen, chat_screen])
    back_button2.click(fn=show_home_screen, outputs=[home_screen, niche_screen, upload_screen, chat_screen])
    back_button_chat.click(fn=show_home_screen, outputs=[home_screen, niche_screen, upload_screen, chat_screen])

    train_niche_button.click(fn=train_from_niche, inputs=[niche_dropdown, model_dropdown_niche], outputs=[output_niche, download_button_niche])
    train_file_button.click(fn=train_from_file, inputs=[file_upload, model_dropdown_file], outputs=[output_file, download_button_file])
    load_chat_button.click(fn=load_finetuned_model, outputs=chat_status) 
demo.launch()
