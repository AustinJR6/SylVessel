"""
Sylana Vessel - CTransformers Model Loader (Alternative to llama-cpp)
Uses CTransformers which has pre-built wheels (no compiler needed!)
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CTransformersModelLoader:
    """
    Loads and manages quantized Llama models using CTransformers

    Benefits over llama-cpp-python:
    - Pre-built wheels (no C++ compiler needed!)
    - Easy installation on Windows
    - Same performance
    - Works on ancient laptops
    """

    def __init__(self, model_path: str, context_length: int = 2048, threads: Optional[int] = None):
        """
        Initialize model loader

        Args:
            model_path: Path to .gguf model file
            context_length: Context window size
            threads: Number of CPU threads (None = auto)
        """
        self.model_path = model_path
        self.context_length = context_length
        self.threads = threads or os.cpu_count()
        self.llm = None

    def load_model(self):
        """Load the quantized model"""
        try:
            from ctransformers import AutoModelForCausalLM

            logger.info(f"Loading quantized model from: {self.model_path}")
            logger.info(f"Context: {self.context_length}, Threads: {self.threads}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"Please download it first."
                )

            # Load with CTransformers
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type='llama',
                context_length=self.context_length,
                threads=self.threads,
                gpu_layers=0  # CPU only
            )

            logger.info("Quantized model loaded successfully!")
            return self.llm

        except ImportError:
            raise ImportError(
                "ctransformers not installed. Install with:\n"
                "pip install ctransformers"
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
            top_p: Nucleus sampling
            stop: Stop sequences

        Returns:
            Generated text
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if stop is None:
            stop = ["User:", "Human:", "\n\n\n"]

        try:
            # Generate with ctransformers
            output = self.llm(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=False
            )

            return output.strip()

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow calling directly"""
        return self.generate(prompt, **kwargs)


if __name__ == "__main__":
    print("CTransformers Model Loader Test")
    print("-" * 60)

    model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please download the model first.")
    else:
        loader = CTransformersModelLoader(model_path, context_length=512)
        loader.load_model()

        prompt = "Hello! How are you?"
        response = loader.generate(prompt, max_tokens=50)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
