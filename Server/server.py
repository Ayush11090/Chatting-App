from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
import os
from pydub import AudioSegment
import torchaudio
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the text classification model and tokenizer
text_model_name = 'ayushbillade/OffensiveChatDefender'
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name)
text_model.eval()

# Load the audio classification model
audio_model_name = 'D:/Semister 7/Major Project/Project/Messaging App/Server/Model'
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model_name)
audio_model.eval()

# Ensure models are on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model.to(device)
audio_model.to(device)

import speech_recognition as sr

def wav_to_text(wav_file):
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Load the WAV file
    with sr.AudioFile(wav_file) as source:
        print("Listening to the audio...")
        audio_data = recognizer.record(source)  # Read the entire audio file
    
    try:
        # Recognize and convert speech to text
        print("Converting speech to text...")
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Speech was unintelligible."
    except sr.RequestError as e:
        return f"Could not request results; {e}"


# Helper function for text prediction
def classify_text(content):
    variations = [content, content.upper(), content.lower()]
    for text_variant in variations:
        inputs = text_tokenizer(text_variant, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = text_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            if predicted_class == 1:  # Offensive content
                return 1
    return 0  # All variations are safe

# Helper function for audio prediction
def classify_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    audio = waveform.squeeze().numpy()
    
    inputs = audio_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    # Move inputs to the same data type as the model weights
    dtype = next(audio_model.parameters()).dtype
    input_values = inputs["input_values"].to(device).to(dtype)

    with torch.no_grad():
        logits = audio_model(input_values).logits
        predicted_class_id = logits.argmax().item()

    return predicted_class_id

def save_audio_from_base64(encoded_audio, filename):
    try:
        audio_data = base64.b64decode(encoded_audio)
        temp_path = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as audio_file:
            audio_file.write(audio_data)
        return temp_path
    except Exception as e:
        raise ValueError(f"Error decoding Base64 audio: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        predicted_class = 0;
        data = request.json
        print(data)

        if isinstance(data.get("text"), str):
            # Text classification
            predicted_class = predicted_class | classify_text(data["text"])
            if predicted_class== 1 :
                return jsonify({"predicted_class": predicted_class})

        data = request.get_json()
        predicted_class = 0
        if "audio" in data and "filename" in data:
            # Decode and save audio file
            audio_path = save_audio_from_base64(data["audio"], data["filename"])
            
            # Check if the file is an MP3
            if audio_path.endswith(".mp3"):
                wav_path = os.path.splitext(audio_path)[0] + ".wav"
                audio = AudioSegment.from_mp3(audio_path)
                audio.export(wav_path, format="wav")
                os.remove(audio_path)  # Remove the MP3 file after conversion
                audio_path = wav_path
            
            print(audio_path)
            # Transcribe the audio
            transcription = wav_to_text(audio_path)
            print("Transcription:", transcription)
            
            # Perform text classification
            predicted_class = classify_audio(audio_path)
            
            if predicted_class== 1 :
                return jsonify({"predicted_class": predicted_class})
              
            predicted_class = 0
            # Perform audio classification
            predicted_class = classify_text(transcription)
            
            # Clean up the temporary file
            os.remove(audio_path)


        return jsonify({"predicted_class": predicted_class})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)
