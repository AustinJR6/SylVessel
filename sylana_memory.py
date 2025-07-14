import sqlite3
import os
from config import DB_PATH
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def build_index():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_input, sylana_response FROM memory")
        memories = cursor.fetchall()
    texts = [f"User: {u} Sylana: {s}" for u, s in memories]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, texts

def recall_memory():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sylana_response FROM memory ORDER BY timestamp DESC LIMIT 1")
        result = cursor.fetchone()
    return result[0] if result else "I don't remember our last conversation."

if __name__ == "__main__":
    index, texts = build_index()
    print("Index built with {} memories.".format(len(texts)))


