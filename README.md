# 🔧 LLM Tuner: On-Device Fine-Tuner for Niche Domains

**LLM Tuner** is a lightweight Gradio-based application that allows you to fine-tune open-source language models (like GPT-2) on niche datasets or your own `.txt` files, directly on-device using **LoRA (Low-Rank Adaptation)** *without using any cloud dependency, ensuring full data privacy and control*. It features a simple interface to do fine-tuning and interact with your adapted model through a chat window.

---

## 🚀 Features

- ✅ **Niche Domain Fine-Tuning**  
  Fine-tune using predefined datasets in domains like Finance, Legal, Medical, and Education.

- 📂 **Custom Dataset Upload**  
  Upload your own `.txt` file for custom domain fine-tuning.

- 🔁 **LoRA Training**  
  Parameter-efficient fine-tuning using a backend `lora_train.py` script.

- 💬 **Chat Interface**  
  Chat with the fine-tuned model via an interactive chat window.

- ⬇️ **Download LoRA Adapters**  
  Download LoRA adapter weights after training for offline usage as a `.zip`.

---

## 📁 Project Structure

```llm-tuner/
├── app.py # Main Gradio UI and logic
├── lora_train.py # LoRA training script
├── datasets/
│ ├── finance.txt
│ ├── legal.txt
│ ├── medical.txt
│ └── education.txt
├── lora_adapter/ # (Created during training)
└── lora_adapter.zip # (Zip generated after training)
```

---

## ⚙️ Requirements

Create a `requirements.txt` file and install all dependencies:

```txt
transformers
gradio
torch
shutil
time
logging

pip install -r requirements.txt
```




## 🧪 How to Run
Clone or download the project folder.

Ensure the datasets/ folder includes .txt files for niche domains.

Run the app:

```python app.py```

Open the link in your browser (usually http://127.0.0.1:7860).

## 🖥️ Usage Guide
Option 1: Niche Domain Fine-Tuning
Click "Select Niche Domain Dataset".

Choose a domain and base model (e.g., GPT-2).

Click "Train".

Wait a few seconds — fine-tuning will complete.

Download the LoRA adapter.

Option 2: Upload Custom Dataset
Click "Upload Custom Dataset (.txt)".

Upload your .txt file and choose a base model.

Click "Train".

Fine-tuning will complete in a few seconds.

Download the LoRA adapter.

Chat with Fine-Tuned Model
Go to the Chat Interface screen.

Click "Load Fine-Tuned Model".

Start chatting with the fine-tuned model!

📚 Acknowledgements
Hugging Face Transformers<br>
Gradio UI Framework<br>
LoRA: Low-Rank Adaptation of Large Language Models<br>


*~by Debangan Sarkar, 23117043 :)*
