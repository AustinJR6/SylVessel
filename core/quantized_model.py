"""
Sylana Vessel - Quantized Model Loader
Uses llama.cpp for efficient CPU inference with quantized models
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class QuantizedModelLoader:
    """
    Loads and manages quantized Llama models using llama.cpp

    Benefits:
    - Much smaller file size (~6GB vs 13GB)
    - Faster CPU inference
    - Lower memory usage
    - No GPU required
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None):
        """
        Initialize quantized model loader

        Args:
            model_path: Path to .gguf model file
            n_ctx: Context window size (default: 2048)
            n_threads: Number of CPU threads (default: auto-detect)
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count()
        self.llm = None

    def load_model(self):
        """Load the quantized model"""
        try:
            from llama_cpp import Llama

            logger.info(f"Loading quantized model from: {self.model_path}")
            logger.info(f"Context size: {self.n_ctx}, Threads: {self.n_threads}")

            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"Please download a quantized .gguf model first."
                )

            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=0,  # CPU only
                verbose=False
            )

            logger.info("Quantized model loaded successfully!")
            return self.llm

        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with:\n"
                "pip install llama-cpp-python"
            )
        except Exception as e:
            logger.error(f"Error loading quantized model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.9,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            Generated text
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Default stop sequences for chat models
        if stop is None:
            stop = ["User:", "Human:", "\n\n\n"]

        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False
            )

            # Extract generated text
            generated_text = output['choices'][0]['text'].strip()
            return generated_text

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow calling the loader directly like a function"""
        return self.generate(prompt, **kwargs)


def download_quantized_model(output_dir: str = "./models") -> str:
    """
    Helper function to download a quantized Llama 2 model

    Args:
        output_dir: Directory to save model

    Returns:
        Path to downloaded model
    """
    import urllib.request
    from tqdm import tqdm

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Recommended model: Llama-2-7B-Chat Q4_K_M (best quality/size balance)
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    model_filename = "llama-2-7b-chat.Q4_K_M.gguf"
    output_path = os.path.join(output_dir, model_filename)

    # Check if already downloaded
    if os.path.exists(output_path):
        logger.info(f"Model already exists: {output_path}")
        return output_path

    logger.info(f"Downloading quantized model (~6GB)...")
    logger.info(f"From: {model_url}")
    logger.info(f"To: {output_path}")

    # Download with progress bar
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve(
            model_url,
            output_path,
            reporthook=t.update_to
        )

    logger.info(f"Model downloaded successfully: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test the quantized model loader
    print("Quantized Model Loader Test")
    print("-" * 60)

    # Example: Download and load model
    model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"

    if not os.path.exists(model_path):
        print("Model not found. To download:")
        print("python -c \"from core.quantized_model import download_quantized_model; download_quantized_model()\"")
    else:
        loader = QuantizedModelLoader(model_path, n_ctx=512)
        loader.load_model()

        # Test generation
        prompt = "Hello! How are you today?"
        response = loader.generate(prompt, max_tokens=50)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
