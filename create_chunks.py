import whisper
import torch
import json
import os

compute_device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model on device:", compute_device)
model = whisper.load_model("turbo", device = compute_device)
print("Model loaded successfully.")

audios = os.listdir("audios")

print("Transcribing started... ")
print()

for audio in audios:
    print(f"Transcribing {audio}.. .")
 
    number = audio.split("_")[0]
    title = audio.split("_")[1][ : -4]

    transcription_result = model.transcribe(audio = f"audios/{audio}", 
                                            language = "en", 
                                            word_timestamps = False)

    chunks = []

    for segment in transcription_result["segments"]:
        chunks.append({"number" : number,
                       "title" : title,
                       "start" : segment["start"], 
                        "end" : segment["end"], 
                        "text" : segment["text"]})

    chunks_with_metadata = {"chunks" : chunks, "text" : transcription_result["text"]}

    print(f"Transcription of {audio} completed.")
    print("Saving to json file...")
    print()
    with open(f"jsons/{audio}.json", "w") as f:
        json.dump(chunks_with_metadata, f)