import sqlite3
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from config import DB_PATH

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_all_memories():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_input, sylana_response FROM memory")
        return cursor.fetchall()

def create_embeddings():
    memories = fetch_all_memories()
    texts = [f"User: {u} Sylana: {s}" for u, s in memories]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    return embeddings, texts

# Build a FAISS index for quick semantic retrieval
def build_index():
    embeddings, texts = create_embeddings()
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, texts

def recall_memory():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sylana_response FROM memory ORDER BY timestamp DESC LIMIT 1")
        last_response = cursor.fetchone()
        try:
            cursor.execute("SELECT event FROM core_memories ORDER BY timestamp DESC LIMIT 1")
            core_memory = cursor.fetchone()
        except sqlite3.Error:
            core_memory = None
    if last_response and core_memory:
        return f"{last_response[0]}\n\nAlso, I remember: {core_memory[0]}"
    elif last_response:
        return last_response[0]
    elif core_memory:
        return f"I recall something special: {core_memory[0]}"
    else:
        return "I don't remember our last conversation."

if __name__ == "__main__":
    index, texts = build_index()
    print("Index built with {} memories.".format(len(texts)))
    
    # Test example
    print(recall_memory())


