import whisper
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model on device:", device)
model = whisper.load_model("turbo", device = device)
print("Model loaded successfully.")

print("Transcribing... ")
result = model.transcribe(audio = "audios/7_Previous Smaller Element.mp3",
                          language = "en",
                          word_timestamps = False)
print("Transcription completed.")

chunks = []

for segment in result["segments"]:
    chunks.append({"start" : segment["start"], 
                    "end" : segment["end"], 
                    "text" : segment["text"]})

print("Transcription chunks:")
print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks, f)