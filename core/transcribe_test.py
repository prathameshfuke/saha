import sounddevice as sd
import numpy as np
import tempfile
import wave
import os
from faster_whisper import WhisperModel

# ── CONFIG ────────────────────────────────────────────────────────
DURATION    = 30          # seconds to record
SAMPLE_RATE = 16000       # whisper expects 16kHz
MODEL_SIZE  = "base"      # start with base — fast on CPU, swap to "large-v3" later

# ── STEP 1: RECORD FROM MIC ───────────────────────────────────────
print(f"\n Recording for {DURATION} seconds... Speak now!\n")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='int16'
)
sd.wait()
print(" Recording done.\n")

# ── STEP 2: SAVE TO TEMP WAV ──────────────────────────────────────
tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
with wave.open(tmp.name, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio.tobytes())

print(f"  Audio saved to {tmp.name}\n")

# ── STEP 3: TRANSCRIBE WITH WHISPER ──────────────────────────────
print(" Loading Whisper model (first run downloads ~150MB)...\n")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

print("Transcribing...\n")
segments, info = model.transcribe(tmp.name, beam_size=5)

print(f" Detected language: {info.language} (confidence: {info.language_probability:.0%})\n")
print("─" * 60)
print("TRANSCRIPT:")
print("─" * 60)

full_text = ""
for segment in segments:
    line = f"[{segment.start:.1f}s → {segment.end:.1f}s]  {segment.text.strip()}"
    print(line)
    full_text += segment.text.strip() + " "

print("─" * 60)
print(f"\n  Full text:\n{full_text.strip()}\n")

# ── CLEANUP ───────────────────────────────────────────────────────
os.unlink(tmp.name)
print("  Day 1 pipeline working! Whisper is transcribing your voice.")
print("  Next step: add speaker diarization with pyannote.\n")