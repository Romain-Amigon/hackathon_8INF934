from sentence import Sentence
import os

def main():
    # Initialize the object in French
    s = Sentence(language='fr-FR')
    
    print("=== START OF TESTS ===")

    # Test 1: Text to Speech
    print("\n--- Test 1: toSpeech ---")
    text_to_read = "Bonjour, ceci est un test de génération vocale en français."
    audio_file = "test_audio.mp3"
    print(f"Target text: '{text_to_read}'")
    s.toSpeech(text_to_read, audio_file)

    # Test 2: Speech to Text (from a file)
    print("\n--- Test 2: toText ---")
    print("Note: SpeechRecognition requires .wav or .flac files to read audio.")
    test_file = input("Enter the path to a .wav audio file to transcribe (or press Enter to skip): ")
    
    if test_file and os.path.exists(test_file):
        text_result = s.toText(test_file)
        print(f"> Text extracted from file: '{text_result}'")
    else:
        print("> File skipped or not found.")

    print("\n=== END OF TESTS ===")

if __name__ == "__main__":
    main()