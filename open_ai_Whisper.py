from flask import Flask, request, jsonify
import requests
import tempfile
import openai
import os

app = Flask(__name__)

openai.api_key = os.getenv("API_KEY")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    audio_url = data.get("audioUrl")
    if not audio_url:
        return jsonify({"error": "Missing audioUrl"}), 400

    # Download the audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name

    # Send it to Whisper API
    with open(tmp_path, "rb") as audio_file:
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"  # optional
        )

    # Clean up and return text
    os.remove(tmp_path)
    return jsonify({"text": transcript.text})
