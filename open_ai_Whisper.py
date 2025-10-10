from flask import Flask, request, jsonify
import requests
import tempfile
import os
from openai import OpenAI

app = Flask(__name__)

# Use the API key stored in your GitHub Actions or Render environment (API_KEY)
client = OpenAI(api_key=os.getenv("API_KEY"))

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    audio_url = data.get("audioUrl")
    if not audio_url:
        return jsonify({"error": "Missing audioUrl"}), 400

    try:
        # Download the audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        # Send it to Whisper API for transcription
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"  # optional
            )

        # Clean up the temporary file
        os.remove(tmp_path)

        return jsonify({
            "text": transcript.text,
            "status": "success",
            "source": audio_url
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == "__main__":
    # Render assigns a dynamic port to your app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
