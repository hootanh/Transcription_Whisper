# Transcription_Whisper
# Whisper Transcription API

A simple Flask app that downloads an audio file from a URL and sends it to OpenAI’s Whisper API for transcription.

## Endpoints
**POST /transcribe**
```json
{
  "audioUrl": "https://example.com/audio.mp3"
}
