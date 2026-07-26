import whisper

model = whisper.load_model("turbo", device="cuda")

result = model.transcribe(
    "audios/7_Previous Smaller Element.mp3",
    verbose=True
)

print(result["language"])