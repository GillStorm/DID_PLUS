import subprocess
import os
import sys

FFMPEG = r"C:\Users\ashis\Downloads\ffmpeg-2025-12-22-git-c50e5c7778-full_build\ffmpeg-2025-12-22-git-c50e5c7778-full_build\bin\ffmpeg.exe"

INPUT = "Recording (3).m4a"
OUTPUT = "verify_voice.wav"

if not os.path.exists(FFMPEG):
    print("[ERROR] ffmpeg.exe not found:", FFMPEG)
    sys.exit(1)

if not os.path.exists(INPUT):
    print("[ERROR] input audio not found:", INPUT)
    sys.exit(1)

cmd = [
    FFMPEG,
    "-y",
    "-i", INPUT,
    "-ac", "1",
    "-ar", "16000",
    "-sample_fmt", "s16",
    OUTPUT
]

print("Running:", " ".join(cmd))

subprocess.run(cmd, check=True)

if not os.path.exists(OUTPUT) or os.path.getsize(OUTPUT) == 0:
    print("[FATAL] voice.wav not created")
    sys.exit(1)

print("✅ voice.wav created successfully")
print("Size:", os.path.getsize(OUTPUT), "bytes")
