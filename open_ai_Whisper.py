from flask import Flask, request, jsonify
import requests
import tempfile
import os
import traceback
from openai import OpenAI

app = Flask(__name__)

# Initialize the OpenAI client using the API key from Render/GitHub environment
client = OpenAI(api_key=os.getenv("API_KEY"))

@app.route("/", methods=["GET"])
def home():
    """Simple health check route."""
    return jsonify({
        "message": "Whisper transcription API is live. Use POST /transcribe to transcribe audio files."
    })

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accepts JSON with an 'audioUrl' field, downloads the audio file,
    validates it, and sends it to OpenAI Whisper for transcription.
    """
    data = request.get_json()
    audio_url = data.get("audioUrl") if data else None

    if not audio_url:
        return jsonify({"error": "Missing audioUrl"}), 400

    tmp_path = None
    try:
        # --- Step 1: Download the audio file ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            # Follow redirects to reach the actual audio file
            response = requests.get(audio_url, stream=True, allow_redirects=True)
            response.raise_for_status()

            # Validate that the response is audio
            content_type = response.headers.get("Content-Type", "")
            print(f"Downloaded from {audio_url} with content type: {content_type}")

            if "audio" not in content_type.lower():
                raise ValueError(f"URL did not return audio content. Got: {content_type}")

            # Write the audio stream to a temporary file
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.flush()
            tmp_path = tmp.name

        # Check that file size is reasonable (avoid empty or corrupted downloads)
        file_size = os.path.getsize(tmp_path)
        if file_size < 1000:
            raise ValueError(f"Downloaded file too small ({file_size} bytes). Invalid or incomplete audio.")

        print(f"Downloaded {file_size} bytes to {tmp_path}")

        # --- Step 2: Transcribe using Whisper ---
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"  # optional
            )

        # --- Step 3: Return transcription result ---
        return jsonify({
            "text": transcript.text,
            "status": "success",
            "source": audio_url
        })

    except Exception as e:
        # Log detailed error information to Render logs
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
            print(f"Deleted temp file: {tmp_path}")

if __name__ == "__main__":
    # Render dynamically assigns a port; default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
