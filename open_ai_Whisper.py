from flask import Flask, request, jsonify
import requests
import tempfile
import os
import traceback
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client using the API key from Render/GitHub environment
client = OpenAI(api_key=os.getenv("API_KEY"))

@app.route("/", methods=["GET"])
def home():
    """Simple health check route."""
    return jsonify({
        "message": "Whisper transcription API is live. Use POST /transcribe to upload audio."
    })

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Download an audio file from a URL and transcribe it using OpenAI Whisper."""
    data = request.get_json()
    audio_url = data.get("audioUrl") if data else None

    if not audio_url:
        return jsonify({"error": "Missing audioUrl in request body."}), 400

    tmp_path = None
    try:
        # --- Step 1: Download audio file ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        # --- Step 2: Send to Whisper model ---
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"  # optional: specify language if known
            )

        # --- Step 3: Return result ---
        return jsonify({
            "text": transcript.text,
            "status": "success",
            "source": audio_url
        })

    except Exception as e:
        print("=== ERROR IN TRANSCRIPTION ===")
        traceback.print_exc()
        print("==============================")
        return jsonify({
            "error": f"Error code: 500 - {str(e)}",
            "status": "failed"
        }), 500

    finally:
        # Always clean up the temporary file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    # Render assigns a dynamic port to the app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
