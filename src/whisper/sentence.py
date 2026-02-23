import speech_recognition as sr
from gtts import gTTS
import os

class Sentence:
    def __init__(self, language='fr-FR'):
        self.language = language
        # 'fr-FR' for SpeechRecognition, 'fr' pour gTTS
        self.gtts_lang = language.split('-')[0] 
        self.recognizer = sr.Recognizer()

    def toSpeech(self, text, output_file="output.mp3"):
        """Transform text to speech and save it as an audio file."""
        try:
            tts = gTTS(text=text, lang=self.gtts_lang)
            tts.save(output_file)
            print(f"[Success] file save to : {output_file}")
            return output_file
        except Exception as e:
            print(f"[Error] : {e}")
            return None

    def toText(self, audio_file):
        """Transform an audio recording (.wav, .flac) into text."""
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                # Utilisation de l'API Google Web Speech (gratuite par défaut)
                text = self.recognizer.recognize_google(audio_data, language=self.language)
                return text
        except sr.UnknownValueError:
            return "[Error] not able to understand the audio."
        except sr.RequestError as e:
            return f"[Error] : {e}"
        except Exception as e:
            return f"[Error toText] : {e}"