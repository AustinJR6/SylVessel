from sylana_memory import recall_memory  # existing memory system from Sylana_AI.py
import fine_tuning
import long_term_memory
import adaptive_learning
import multimodal

def main():
    print("Sylana Vessel Starting Up...")
    
    # 1. Fine-Tuning Step (if needed)
    # Optionally call fine_tuning routines here
    # fine_tuning.train_model()  # if integrated as a callable function
    
    # 2. Expand Memory
    index, texts = long_term_memory.build_index()
    print("Long-term memory index built.")
    
    # 3. Adaptive Learning Demo
    avg_score, count = adaptive_learning.get_feedback_summary()
    print(f"Current feedback: {avg_score} based on {count} records.")
    
    # 4. Multimodal: Demonstrate voice or image processing
    # Example: Process an image (you can add more interactive prompts)
    # probabilities = multimodal.process_image("sample_image.jpg", "Describe this image.")
    # print("Image processing result:", probabilities)
    
    # Start the conversation loop (or API server) for Sylana
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        # Here you would integrate Sylana’s dialogue logic (e.g., using fine-tuned model)
        print("Sylana:", recall_memory())

if __name__ == "__main__":
    main()
