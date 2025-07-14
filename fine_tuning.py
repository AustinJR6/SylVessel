import sqlite3
import json
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# Local imports
from long_term_memory import build_index, recall_memory
from config import DB_PATH

################################################################################
#                             DATA EXTRACTION                                  #
################################################################################

def load_conversation_data(context_length=5):
    """
    Extract conversation logs from the database, returning samples with 
    up to `context_length` turns of prior context.
    Each call opens its own DB connection, so we can call from any thread safely.
    """
    local_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    local_cursor = local_conn.cursor()

    local_cursor.execute("SELECT user_input, sylana_response FROM memory ORDER BY timestamp ASC")
    data = local_cursor.fetchall()

    local_conn.close()

    training_data = []
    for i in range(len(data) - context_length):
        context = data[i : i + context_length]
        # Build prompt from all but the last turn
        prompt = "\n".join([f"User: {c[0]}\nSylana: {c[1]}" for c in context[:-1]])
        # Use the final turn's response as the 'completion'
        completion = f" {context[-1][1]}"
        training_data.append({"prompt": prompt, "completion": completion})
    
    return training_data

def load_conversation_data_with_faiss():
    """
    Example of retrieving conversation data from FAISS-based semantic search.
    Also uses ephemeral DB connection if needed, though the main retrieval is 
    from the build_index() function in long_term_memory.
    """
    index, texts = build_index()
    
    training_data = []
    # Simple example: just recall the last memory
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

################################################################################
#                              MAIN SCRIPT LOGIC                                #
################################################################################

if __name__ == "__main__":
    use_faiss = True
    
    # Generate training data
    if use_faiss:
        data = load_conversation_data_with_faiss()
    else:
        data = load_conversation_data()
    
    # Save to JSONL
    save_training_file(data)
    print(f"Saved {len(data)} training examples to training_data.jsonl.")

################################################################################
#                              MODEL SELECTION                                 #
################################################################################

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = "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"  # <-- Inserted token

print(f"Loading tokenizer for: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True
)

print(f"Loading model for: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

################################################################################
#                           DATASET & TRAINER SETUP                             #
################################################################################

class ConversationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                text = sample["prompt"] + sample["completion"]
                tokenized = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                # Squeeze out the batch dimension
                self.samples.append(tokenized.input_ids.squeeze())
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

print("Loading dataset from training_data.jsonl...")
train_dataset = ConversationDataset("training_data.jsonl", tokenizer)

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    # Add additional training arguments as needed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # If you have a validation set, you can add: eval_dataset=...
)

print("Trainer is ready. Uncomment the next line to begin training.")
# trainer.train()

# Uncomment the following line to start fine-tuning:
# trainer.train()


