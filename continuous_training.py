# continuous_training.py
import time
import sqlite3
import os
import json
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import Dataset
from datetime import datetime

# Import secure configuration
from core.config_loader import config

# --- Configuration ---
# Use configuration from environment (aligned with main Sylana_AI.py)
MODEL_NAME = config.MODEL_NAME  # Llama 2 7B Chat
HF_TOKEN = config.HF_TOKEN
CHECKPOINT_DIR = config.CHECKPOINT_DIR
TRAINING_DATA_FILE = "training_data.jsonl"
DATABASE_PATH = config.DB_PATH
POLL_INTERVAL = 300  # seconds (5 minutes)
ENABLE_TRAINING = config.ENABLE_FINE_TUNING  # Safety flag

# --- Dataset Class ---
class ConversationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                # Concatenate prompt and completion for fine-tuning.
                text = sample["prompt"] + sample["completion"]
                # Adjust max_length as needed. Llama models typically allow longer sequences.
                tokenized = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
                input_ids = tokenized.input_ids.squeeze()  # tensor of shape [seq_length]
                # For causal LM training, labels are the same as input_ids.
                self.samples.append({
                    "input_ids": input_ids,
                    "labels": input_ids.clone()
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# --- Functions for Data Extraction ---
def load_new_conversation_data(last_finetune_timestamp):
    """
    Extract conversation pairs added after last_finetune_timestamp.
    Assumes your memory table has a 'timestamp' field in the format '%Y-%m-%d %H:%M:%S'.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    query = "SELECT user_input, sylana_response, timestamp FROM memory"
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    
    training_data = []
    new_timestamp = last_finetune_timestamp
    for user_input, sylana_response, ts in data:
        # Convert the string timestamp to an integer epoch time
        ts_epoch = int(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp())
        if ts_epoch > last_finetune_timestamp:
            prompt = f"User: {user_input}\nSylana:"
            completion = f" {sylana_response}"
            training_data.append({"prompt": prompt, "completion": completion})
            if ts_epoch > new_timestamp:
                new_timestamp = ts_epoch
    return training_data, new_timestamp

def save_training_file(training_data, filename=TRAINING_DATA_FILE):
    """
    Saves training data to a JSONL file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sample in training_data:
            f.write(json.dumps(sample) + "\n")

def load_model_and_tokenizer():
    """Load model from checkpoint or base model using secure token"""
    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN not configured! Please set up your .env file.")
        print("   See SECURITY_NOTICE.md for instructions.")
        exit(1)

    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    if os.path.exists(CHECKPOINT_DIR) and os.path.exists(config_path):
        print("Loading model from checkpoint...")
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, token=HF_TOKEN)
    else:
        print("No valid checkpoint found. Loading base model from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    return model, tokenizer


# --- Continuous Training Loop ---
def continuous_training_loop():
    """Main continuous learning loop - only runs if ENABLE_FINE_TUNING=true"""
    if not ENABLE_TRAINING:
        print("⚠️  Fine-tuning is DISABLED. Set ENABLE_FINE_TUNING=true in .env to activate.")
        print("   This is a safety feature to prevent accidental model modification.")
        return

    print("✅ Fine-tuning ENABLED. Starting continuous learning loop...")
    model, tokenizer = load_model_and_tokenizer()
    
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,  # Fine-tune for one epoch per cycle
        per_device_train_batch_size=2,  # Adjust batch size based on available GPU memory
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
    )
    
    # Initialize last_finetune_timestamp. In production, persist this value if needed.
    last_finetune_timestamp = 0
    
    while True:
        print("Checking for new conversation data...")
        training_data, new_timestamp = load_new_conversation_data(last_finetune_timestamp)
        if not training_data:
            print("No new data for fine-tuning. Waiting...")
        else:
            print(f"Found {len(training_data)} new training examples. Starting fine-tuning...")
            save_training_file(training_data, TRAINING_DATA_FILE)
            train_dataset = ConversationDataset(TRAINING_DATA_FILE, tokenizer)
            if len(train_dataset) == 0:
                print("Training dataset is empty after processing. Skipping fine-tuning.")
            else:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                )
                trainer.train()
                # Update last_finetune_timestamp to the latest timestamp from new data.
                last_finetune_timestamp = new_timestamp
                # Save the updated model and tokenizer.
                model.save_pretrained(CHECKPOINT_DIR)
                tokenizer.save_pretrained(CHECKPOINT_DIR)
                print("Model fine-tuning complete. Model updated with new data.")
        
        # Sleep before polling again.
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    continuous_training_loop()
