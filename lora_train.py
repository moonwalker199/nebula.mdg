import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)

# Choose model here (you can change this or accept as CLI arg or Gradio input)
BASE_MODEL = os.getenv("BASE_MODEL", "gpt2")  # fallback to GPT2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# === Dataset Setup ===
def get_dataset():
    data_path = "toy_dataset.txt"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [f"This is sample sentence number {i}." for i in range(25)]
        with open(data_path, "w") as f:
            f.write("\n".join(texts))
    return Dataset.from_dict({"text": texts})

# === Tokenization ===
def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)
    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


# === Training Logic ===
def train_lora_model():
    logging.info(f"Loading model and tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(DEVICE)

    logging.info("Applying LoRA config...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "q_proj", "v_proj"],  # customize per model
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = get_dataset()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir="lora-output",
        per_device_train_batch_size=4,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    logging.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    logging.info(f"Sample tokenized input: {tokenized_dataset[0]}")

    trainer.train()
    model.save_pretrained("lora-output")
    tokenizer.save_pretrained("lora-output")
    logging.info("Training completed and model saved to 'lora-output/'.")

# === Entry Point ===
if __name__ == "__main__":
    logging.info("Starting LoRA training script...")
    train_lora_model()
    logging.info("LoRA training script completed.")