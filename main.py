# main.py
import sys
import os

# Optional: Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import memory functions and conversation generation functions
from Memory_System.sylana_memory import recall_memory  # For initial demo recall
from Sylana_AI import remember_last_response   # Use the chat-generation function from Sylana_AI.py
from Memory_System.long_term_memory import build_index
from AI_Processing.adaptive_learning import get_feedback_summary
from Interaction_Interface.multimodal import transcribe_audio, process_image

def main():
    print("Sylana Vessel Starting Up...\n")
    
    # Demonstrate the basic memory recall function
    last_memory = recall_memory()
    print("Last memory recalled:", last_memory)
    
    # Build the long-term memory index and display the number of entries
    index, texts = build_index()
    print(f"Long-term memory index built with {len(texts)} entries.")
    
    # Get and display the adaptive learning feedback summary
    avg_score, count = get_feedback_summary()
    print(f"Adaptive learning feedback: Average Score: {avg_score} from {count} entries.\n")
    
    # Provide a simple multimodal demonstration menu
    print("Choose an option:")
    print("1. Test voice transcription")
    print("2. Test image processing")
    print("3. Start conversation")
    choice = input("Enter 1, 2, or 3: ")
    
    if choice == "1":
        audio_file = input("Enter path to audio file: ")
        transcription = transcribe_audio(audio_file)
        print("Transcribed Audio:", transcription)
    elif choice == "2":
        image_path = input("Enter path to image file: ")
        query_text = input("Enter query for image processing: ")
        probabilities = process_image(image_path, query_text)
        print("Image processing result probabilities:", probabilities)
    elif choice == "3":
        print("\nStarting conversation. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting conversation.")
                break
            # Generate a new response using remember_last_response
            response = remember_last_response(user_input)
            print("Sylana:", response)

if __name__ == "__main__":
    main()
