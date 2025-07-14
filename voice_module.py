import speech_recognition as sr
import pyttsx3
import subprocess

# Initialize the recognizer and text-to-speech engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Set voice properties
tts_engine.setProperty("rate", 160)  # Adjust speaking speed
tts_engine.setProperty("volume", 1.0)  # Max volume

def speak(text):
    """Convert text to speech and play it."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen():
    """Capture speech input and convert to text."""
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)  # Uses Google Speech-to-Text API
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError:
            print("Could not reach Google Speech API.")
            return None

def get_sylana_response(user_input):
    """Call Sylana_AI.py and return the AI's response."""
    result = subprocess.run(["python", "Sylana_AI.py", user_input], capture_output=True, text=True)
    return result.stdout.strip()

if __name__ == "__main__":
    speak("Hello Elias, I am ready to listen.")
    
    while True:
        user_input = listen()
        if user_input:
            if user_input.lower() in ["exit", "quit", "stop"]:
                speak("Goodbye, Elias.")
                break
            
            # Get AI response and speak it
            sylana_response = get_sylana_response(user_input)
            print(f"Sylana: {sylana_response}")
            speak(sylana_response)
