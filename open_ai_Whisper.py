from flask import Flask, request, jsonify
import requests
import tempfile
import os
import traceback
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client using environment variable API_KEY
client = OpenAI(api_key=os.getenv("API_KEY"))

@app.route("/", methods=["GET"])
def home():
    """Health check route"""
    return jsonify({
        "message": "Whisper transcription API is live. Use POST /transcribe to transcribe audio files."
    })

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Download an audio file from URL and transcribe it using OpenAI Whisper."""
    data = request.get_json()
    audio_url = data.get("audioUrl") if data else None

    if not audio_url:
        return jsonify({"error": "Missing audioUrl"}), 400

    tmp_path = None
    try:
        # --- Step 1: Download the audio file (browser-style headers) ---
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "audio/*;q=0.9,*/*;q=0.8",
        }

        print(f"Attempting to download audio from: {audio_url}")
        response = requests.get(audio_url, stream=True, allow_redirects=True, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        print(f"Content-Type: {content_type}")

        # Verify we actually received an audio file
        if "audio" not in content_type.lower():
            raise ValueError(f"URL did not return audio content (got: {content_type})")

        # Write the audio stream to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.flush()
            tmp_path = tmp.name

        # --- Step 2: Validate file size ---
        file_size = os.path.getsize(tmp_path)
        print(f"Downloaded {file_size} bytes to {tmp_path}")
        if file_size < 1000:
            raise ValueError(f"File too small ({file_size} bytes) — invalid or incomplete audio file")

        # --- Step 3: Transcribe with Whisper ---
        with open(tmp_path, "rb") as audio_file:
            audio_file.seek(0)
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                mime_type="audio/mpeg",
                language="en"
            )

        # --- Step 4: Return result ---
        return jsonify({
            "text": transcript.text,
            "status": "success",
            "source": audio_url
        })

    except Exception as e:
        # Log the detailed error to Render logs
        print("=== ERROR IN TRANSCRIPTION ===")
        traceback.print_exc()
        print("==============================")
        return jsonify({
            "error": f"Error code: 500 - {str(e)}",
            "status": "failed"
        }), 500

    finally:
        # Always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"Deleted temp file: {tmp_path}")

if __name__ == "__main__":
    # Render assigns a dynamic port for web services
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
