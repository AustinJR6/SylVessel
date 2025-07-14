import sqlite3
import json
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
from long_term_memory import build_index, recall_memory
from config import DB_PATH

# Connect to the Sylana memory database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def load_conversation_data(context_length=5):
    """Extracts conversation logs with context for better training."""
    cursor.execute("SELECT user_input, sylana_response FROM memory ORDER BY timestamp ASC")
    data = cursor.fetchall()
    
    training_data = []
    for i in range(len(data) - context_length):
        context = data[i:i + context_length]
        prompt = "\n".join([f"User: {c[0]}\nSylana: {c[1]}" for c in context[:-1]])
        completion = f" {context[-1][1]}"
        training_data.append({"prompt": prompt, "completion": completion})
    
    return training_data

def load_conversation_data_with_faiss():
    """Retrieves past conversations using FAISS-based semantic search for training."""
    index, texts = build_index()
    
    training_data = []
    retrieved = recall_memory()
    retrieved_memories = retrieved.split("\n")
    if len(retrieved_memories) > 1:
        prompt = "\n".join(retrieved_memories[:-1])
        completion = f" {retrieved_memories[-1]}"
        training_data.append({"prompt": prompt, "completion": completion})
    
    return training_data

def save_training_file(training_data, filename="training_data.jsonl"):
    """Saves training data in JSONL format required for fine-tuning."""
    with open(filename, "w", encoding="utf-8") as f:
        for sample in training_data:
            f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    use_faiss = True
    
    if use_faiss:
        data = load_conversation_data_with_faiss()
    else:
        data = load_conversation_data()
    
    save_training_file(data)
    print(f"Saved {len(data)} training examples.")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

from torch.utils.data import Dataset

class ConversationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                text = sample["prompt"] + sample["completion"]
                tokenized = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
                self.samples.append(tokenized.input_ids.squeeze())
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

train_dataset = ConversationDataset("training_data.jsonl", tokenizer)

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Uncomment the following line to start fine-tuning:
# trainer.train()
