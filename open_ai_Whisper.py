from flask import Flask, request, jsonify
import requests
import tempfile
import os
import traceback
from openai import OpenAI
import socket
import ssl

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
    """Download an audio file from a URL and transcribe it using OpenAI Whisper."""
    print("\n=== /transcribe REQUEST RECEIVED ===")
    print("Headers:", dict(request.headers))
    print("Content-Type received:", request.content_type)
    print("Request Method:", request.method)
    print("Request Path:", request.path)
    print("Raw Body (first 500 chars):", request.get_data(as_text=True)[:500])
    print("===================================\n")

    # --- Step 1: Parse request data safely ---
    data = None
    try:
        data = request.get_json(force=True)
        print("Parsed JSON data:", data)
    except Exception as e:
        print("Failed to parse JSON:", e)

    audio_url = None
    if data and "audioUrl" in data:
        audio_url = data["audioUrl"]
    elif "audioUrl" in request.form:
        audio_url = request.form["audioUrl"]
    elif "audioUrl" in request.args:
        audio_url = request.args.get("audioUrl")

    if not audio_url:
        return jsonify({
            "error": "Missing 'audioUrl' in body, form, or query.",
            "received_content_type": request.content_type
        }), 400

    tmp_path = None
    try:
        # --- Step 2: DNS check ---
        try:
            print("DNS Lookup:", socket.gethostbyname("www.learningcontainer.com"))
        except Exception as e:
            print("DNS lookup failed:", e)

        # --- Step 3: Download audio with robust headers ---
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "*/*",
        }

        print(f"Attempting to download audio from: {audio_url}")
        response = requests.get(
            audio_url,
            stream=True,
            allow_redirects=True,
            headers=headers,
            timeout=15,
            verify=True  # Change to False temporarily if SSL fails
        )

        print(f"Response status code: {response.status_code}")
        if response.status_code != 200:
            print("Response headers:", response.headers)
            print("Response text (first 200 chars):", response.text[:200])
            return jsonify({
                "error": f"Failed to download file. HTTP {response.status_code}",
                "url": audio_url
            }), 400

        content_type = response.headers.get("Content-Type", "")
        print(f"Downloaded file Content-Type: {content_type}")

        if "audio" not in content_type.lower():
            raise ValueError(f"URL did not return audio content (got: {content_type})")

        # --- Step 4: Save to temp file ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.flush()
            tmp_path = tmp.name

        file_size = os.path.getsize(tmp_path)
        print(f"Downloaded {file_size} bytes to {tmp_path}")

        if file_size < 1000:
            raise ValueError(f"File too small ({file_size} bytes) — invalid or incomplete audio file")

        # --- Step 5: Transcribe with Whisper ---
        with open(tmp_path, "rb") as audio_file:
            audio_file.seek(0)
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )

        print("Transcription successful!")
        return jsonify({
            "text": transcript.text,
            "status": "success",
            "source": audio_url
        })

    except requests.exceptions.RequestException as e:
        print(f"RequestException while downloading: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Download failed: {str(e)}",
            "status": "failed"
        }), 400

    except Exception as e:
        print("=== ERROR IN TRANSCRIPTION ===")
        traceback.print_exc()
        print("==============================")
        return jsonify({
            "error": f"{type(e).__name__}: {str(e)}",
            "status": "failed"
        }), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"Deleted temp file: {tmp_path}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
