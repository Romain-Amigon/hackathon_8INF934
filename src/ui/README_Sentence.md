# Text-to-Speech et Speech-to-Text en Français

Ce projet fournit une interface simple via un objet `Sentence` permettant de générer de l'audio à partir de texte, de transcrire un fichier audio en texte, et de faire de la dictée vocale en temps réel via le microphone, le tout en français.

## `testSentence.py`

1. **Prérequis** : Assurez-vous d'avoir Python installé (version 3.7 ou supérieure).
2. **Installation des dépendances** :
   Ouvrez un terminal dans le dossier du projet et lancez :

```bash
pip install -r requirements.txt
```

* Sur Linux : Faites `sudo apt-get install python3-pyaudio portaudio19-dev` avant le pip install.
* Sur Mac : Faites `brew install portaudio` puis `pip install pyaudio`.

3. **Lancement du test** :
```bash
python testSentence.py
```


## Modèles choisis et Justifications

* **Pour le Text-to-Speech (TTS) : `gTTS` (Google Text-to-Speech)**

gTTS interroge l'API de Google Translate. Il est extrêmement facile à mettre en place, ne nécessite aucun téléchargement de modèle local lourd, et offre une voix française très naturelle et fluide par rapport aux bibliothèques hors-ligne standard.


* **Pour le Speech-to-Text (STT) : `SpeechRecognition` avec le moteur Google Web Speech**

C'est le standard de l'industrie pour les projets Python simples. Le moteur de Google utilisé par défaut (`recognize_google`) est gratuit, ne requiert pas de clé d'API pour un usage basique, gère la ponctuation de base et excelle dans la reconnaissance des accents francophones en direct.


### Alternatives pour le Text-to-Speech (TTS)

| Modèle | Avantages | Inconvénients |
| --- | --- | --- |
| **gTTS (Choisi)** | Voix fluide, très léger, simple. | Nécessite internet, génère du mp3 (parfois dur à manipuler). |
| **pyttsx3** | 100% Hors-ligne, très rapide. | Voix souvent très robotiques (dépend du système d'exploitation). |
| **ElevenLabs / OpenAI (API)** | Réalisme bluffant, clones de voix. | Payant, nécessite une clé API. |
| **Bark / Coqui TTS** | Open-source, haute qualité, hors-ligne. | Très lourd à installer, nécessite une bonne carte graphique (GPU). |

### Alternatives pour le Speech-to-Text (STT)

| Modèle | Avantages | Inconvénients |
| --- | --- | --- |
| **Google Web Speech (Choisi)** | Précis en français, ultra-léger en local. | Nécessite une connexion internet. |
| **Whisper (OpenAI)** | La référence absolue (Open-source), robuste au bruit, hors-ligne. | Très lourd, lent sans GPU, overkill pour de la petite dictée. *(Intégrable via `recognize_whisper()` si désiré).* |
| **Vosk** | 100% hors-ligne, rapide, léger. | Précision inférieure à Whisper/Google, surtout sur le vocabulaire complexe. |

## Misc

Convertir un .m4a en .wac avec ffmpeg :

```bash
ffmpeg -i input.m4a output.wav
```