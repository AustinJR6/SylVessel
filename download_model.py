"""
Quick script to download the quantized model
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.quantized_model import download_quantized_model

if __name__ == "__main__":
    try:
        model_path = download_quantized_model()
        print(f"\nSuccess! Model downloaded to: {model_path}")
        sys.exit(0)
    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("Please try downloading manually from:")
        print("https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
        sys.exit(1)
